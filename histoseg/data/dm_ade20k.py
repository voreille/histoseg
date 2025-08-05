"""
ADE20K DataModule for PyTorch Lightning.
Adapted from benchmark-vfm-ss repository for histoseg.
"""

from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader

from .base_datamodule import BaseDataModule
from .zip_dataset import ZipDataset
from .mappings import get_ade20k_mapping
from .transforms import SegmentationTransforms


class ADE20KDataModule(BaseDataModule):
    """
    ADE20K dataset Lightning DataModule.
    
    Expects the dataset to be in zip format (ADEChallengeData2016.zip)
    as downloaded from the official ADE20K challenge.
    
    Args:
        root (str): Path to directory containing ADEChallengeData2016.zip
        devices: Lightning device specification
        num_workers (int): Number of dataloader workers
        img_size (tuple[int, int]): Target image size (H, W)
        batch_size (int): Batch size
        num_classes (int): Number of classes (150 for ADE20K)
        num_metrics (int): Number of metrics to track
        scale_range (tuple[float, float]): Scale range for augmentation
        ignore_idx (int): Index to ignore in loss computation
    """

    def __init__(
        self,
        root: str,
        devices,
        num_workers: int,
        img_size: tuple[int, int] = (512, 512),
        batch_size: int = 1,
        num_classes: int = 150,
        num_metrics: int = 1,
        scale_range: tuple[float, float] = (0.5, 2.0),
        ignore_idx: int = 255,
        **kwargs
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            img_size=img_size,
        )
        self.save_hyperparameters()
        self.scale_range = scale_range

        # Create transforms
        self.train_transforms = SegmentationTransforms(
            img_size=img_size, 
            scale_range=scale_range,
            training=True
        )
        self.val_transforms = SegmentationTransforms(
            img_size=img_size, 
            scale_range=scale_range,
            training=False
        )

    def setup(self, stage: Union[str, None] = None) -> "ADE20KDataModule":
        """Setup train and validation datasets."""
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.root, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.root, "ADEChallengeData2016.zip"),
            "class_mapping": get_ade20k_mapping(),
            "ignore_idx": self.ignore_idx,
        }
        
        if stage == "fit" or stage is None:
            self.train_dataset = ZipDataset(
                img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
                target_folder_path_in_zip=Path(
                    "./ADEChallengeData2016/annotations/training"
                ),
                transforms=self.train_transforms,
                **dataset_kwargs,
            )
            
        if stage == "fit" or stage == "validate" or stage is None:
            self.val_dataset = ZipDataset(
                img_folder_path_in_zip=Path("./ADEChallengeData2016/images/validation"),
                target_folder_path_in_zip=Path(
                    "./ADEChallengeData2016/annotations/validation"
                ),
                transforms=self.val_transforms,
                **dataset_kwargs,
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
        """Create test dataloader (same as validation for ADE20K)."""
        return self.val_dataloader()

    def predict_dataloader(self):
        """Create prediction dataloader (same as validation for ADE20K)."""
        return self.val_dataloader()
