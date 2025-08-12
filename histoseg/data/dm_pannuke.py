"""
PanNuke DataModule for PyTorch Lightning.
"""

from pathlib import Path
from typing import Union, Optional
from torch.utils.data import DataLoader
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .base_datamodule import BaseDataModule


class PanNukeDataModule(BaseDataModule):
    """
    PanNuke dataset Lightning DataModule.
    
    Loads PanNuke dataset using HuggingFace datasets and sets up cross-validation
    splits for instance segmentation. One fold is used as test set, the remaining 
    two folds are split into train/val (default 80/20 split) for monitoring during training.
    
    Args:
        root (str): Not used for PanNuke (loaded from HF), kept for compatibility
        num_workers (int): Number of dataloader workers
        img_size (tuple[int, int]): Target image size (H, W)
        batch_size (int): Batch size
        num_classes (int): Number of classes (6 for PanNuke: background + 5 nucleus types)
        num_metrics (int): Number of metrics to track
        scale_range (tuple[float, float]): Scale range for augmentation
        ignore_idx (int): Index to ignore in loss computation
        test_fold (str): Fold to use for testing ("fold1", "fold2", or "fold3")
        val_split (float): Fraction of train+val data to use for validation
        random_seed (int): Random seed for train/val split reproducibility
    """

    def __init__(
            self,
            root: str = None,  # Not used for PanNuke
            num_workers: int = 4,
            img_size: tuple[int, int] = (256, 256),
            batch_size: int = 8,
            num_classes: int = 6,  # background + 5 nucleus types
            num_metrics: int = 1,
            scale_range: tuple[float, float] = (0.5, 2.0),
            ignore_idx: int = 255,
            test_fold: str = "fold3",
            val_split: float = 0.2,
            random_seed: int = 42,
            **kwargs) -> None:
        super().__init__(
            root=root or "/tmp/pannuke",  # Dummy path for compatibility
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            img_size=img_size,
        )
        self.save_hyperparameters()
        self.scale_range = scale_range
        self.test_fold = test_fold
        self.val_split = val_split
        self.random_seed = random_seed

        # Validate test_fold argument
        if test_fold not in ["fold1", "fold2", "fold3"]:
            raise ValueError(
                f"test_fold must be one of ['fold1', 'fold2', 'fold3'], got {test_fold}"
            )

        # Create Albumentations transforms
        self.train_transforms = A.Compose([
            A.RandomResizedCrop(height=img_size[0],
                                width=img_size[1],
                                scale=scale_range,
                                ratio=(0.75, 1.33),
                                p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),  # Common for histology
            A.RandomRotate90(p=0.5),  # Common for histology
            A.ColorJitter(brightness=0.125,
                          contrast=0.5,
                          saturation=0.5,
                          hue=0.05,
                          p=0.8),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ],
                    p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.val_transforms = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage: Union[str, None] = None) -> "PanNukeDataModule":
        """Setup train, validation, and test datasets."""
        if not hasattr(self, 'pannuke_data'):
            # Load PanNuke dataset from HuggingFace
            print("Loading PanNuke dataset...")
            self.pannuke_data = load_dataset("RationAI/PanNuke")
            print(
                f"Loaded PanNuke with folds: {list(self.pannuke_data.keys())}")

        # Determine train/val folds (exclude test fold)
        all_folds = ["fold1", "fold2", "fold3"]
        train_val_folds = [
            fold for fold in all_folds if fold != self.test_fold
        ]

        if stage == "fit" or stage is None:
            # Combine train/val fold datasets
            combined_datasets = []
            for fold in train_val_folds:
                combined_datasets.append(self.pannuke_data[fold])

            # Concatenate the datasets
            if len(combined_datasets) == 1:
                combined_data = combined_datasets[0]
            else:
                from datasets import concatenate_datasets
                combined_data = concatenate_datasets(combined_datasets)

            # Split into train and validation indices
            if self.val_split > 0:
                n_samples = len(combined_data)
                indices = list(range(n_samples))
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.val_split,
                    random_state=self.random_seed)
            else:
                train_indices = list(range(len(combined_data)))
                val_indices = []

            self.train_dataset = PanNukeDataset(
                dataset=combined_data,
                indices=train_indices,
                transforms=self.train_transforms,
                img_size=self.img_size,
                ignore_idx=self.ignore_idx)

            print(
                f"Train dataset: {len(self.train_dataset)} samples from folds {train_val_folds}"
            )

        if stage == "fit" or stage == "validate" or stage is None:
            if self.val_split > 0:
                self.val_dataset = PanNukeDataset(
                    dataset=combined_data,
                    indices=val_indices,
                    transforms=self.val_transforms,
                    img_size=self.img_size,
                    ignore_idx=self.ignore_idx)
            else:
                # Use a small subset of train data as validation if no val split
                val_indices = list(range(min(100, len(combined_data))))
                self.val_dataset = PanNukeDataset(
                    dataset=combined_data,
                    indices=val_indices,
                    transforms=self.val_transforms,
                    img_size=self.img_size,
                    ignore_idx=self.ignore_idx)

            print(f"Validation dataset: {len(self.val_dataset)} samples")

        if stage == "test" or stage is None:
            # Use the held-out test fold
            test_data = self.pannuke_data[self.test_fold]
            self.test_dataset = PanNukeDataset(
                dataset=test_data,
                indices=None,  # Use all samples
                transforms=self.val_transforms,  # No augmentation for test
                img_size=self.img_size,
                ignore_idx=self.ignore_idx)

            print(
                f"Test dataset ({self.test_fold}): {len(self.test_dataset)} samples"
            )

        return self

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def predict_dataloader(self):
        """Create prediction dataloader (same as test)."""
        return self.test_dataloader()


class PanNukeDataset:
    """
    PanNuke dataset wrapper for individual samples.
    
    Handles instance segmentation format for PanNuke dataset, providing
    instance masks and corresponding category labels for each instance.
    
    Args:
        dataset: HuggingFace Dataset object
        indices: List of indices to use from the dataset (None = use all)
        transforms: Albumentations transforms to apply
        img_size: Target image size
        ignore_idx: Index to ignore in loss computation
    """

    def __init__(self,
                 dataset,
                 indices=None,
                 transforms=None,
                 img_size=(256, 256),
                 ignore_idx=255):
        self.dataset = dataset
        self.indices = indices if indices is not None else list(
            range(len(dataset)))
        self.transforms = transforms
        self.img_size = img_size
        self.ignore_idx = ignore_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual dataset index
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]

        # Extract image and convert to numpy
        image = np.array(sample['image'])

        # Process instance masks and categories
        instance_masks, categories = self._process_instances(
            sample['instances'], sample['categories'])
        instance_masks = [
            np.array(mask).astype(np.uint8) for mask in instance_masks
        ]
        categories = [int(cat) for cat in categories]

        # Apply transforms to image only (masks will be transformed separately)
        if self.transforms:
            # For instance segmentation, we need to be careful with transforms
            # Apply transforms to image
            transformed = self.transforms(image=image, masks=instance_masks)
            image = transformed['image']
            masks = transformed["masks"]

        # Convert to format expected by base collate function
        # Convert instance masks to tensor format
        mask_labels = []
        class_labels = []

        for mask, category in zip(instance_masks, categories):
            # Convert mask to tensor
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
            mask_labels.append(mask_tensor)
            class_labels.append(torch.tensor(category, dtype=torch.long))

        return {
            'pixel_values': image,  # This should be a tensor from ToTensorV2
            'mask_labels': mask_labels,  # List of mask tensors
            'class_labels': class_labels,  # List of category tensors
            'image_id': f"pannuke_{dataset_idx}",
            'tissue_type': sample['tissue'],
        }
