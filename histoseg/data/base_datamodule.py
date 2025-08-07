"""
Base LightningDataModule for histoseg datasets.
Adapted from benchmark-vfm-ss repository.
"""

from typing import Optional
import torch
import lightning as pl
# from lightning.fabric.utilities.device_parser import _parse_gpu_ids


class BaseDataModule(pl.LightningDataModule):
    """Base data module providing common functionality for all datasets."""
    
    def __init__(
        self,
        root: str,
        # devices,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        num_classes: int,
        num_metrics: int = 1,
        ignore_idx: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()

        self.root = root
        self.ignore_idx = ignore_idx
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_metrics = num_metrics

        # self.devices = (
        #     devices if devices == "auto" else _parse_gpu_ids(devices, include_cuda=True)
        # )

        self.dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "batch_size": batch_size,
        }

    @staticmethod
    def train_collate(batch):
        """Collate function for training batches."""
        pixel_values = []
        mask_labels = []
        class_labels = []

        for item in batch:
            pixel_values.append(item["pixel_values"])
            mask_labels.append(item["mask_labels"])
            class_labels.append(item["class_labels"])

        return {
            "pixel_values": torch.stack(pixel_values),
            "mask_labels": mask_labels,  # List of tensors
            "class_labels": class_labels,  # List of tensors
        }

    @staticmethod
    def eval_collate(batch):
        """Collate function for evaluation batches."""
        # Use same collation for evaluation
        return BaseDataModule.train_collate(batch)
