"""
Simple training script without Lightning CLI for easier debugging.
"""

import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger

from histoseg.training import Mask2FormerModule
from histoseg.data import ADE20KDataModule


def main():
    # Set precision and other global settings
    torch.set_float32_matmul_precision("medium")
    
    # Model configuration
    model = Mask2FormerModule(
        num_classes=150,
        image_size=(512, 512),
        learning_rate=1e-4,
        weight_decay=0.01
    )
    
    # Data configuration
    data = ADE20KDataModule(
        root="/home/val/workspaces/histoseg/data",
        batch_size=2,  # Small batch for debugging
        num_workers=0,  # No multiprocessing for easier debugging
        img_size=(512, 512),
        num_classes=150,
        ignore_idx=255,
        num_metrics=1
    )
    
    # Logger (you can comment this out for no logging)
    logger = TensorBoardLogger(
        save_dir="/home/val/workspaces/histoseg/logs",
        name="debug_run"
    )
    
    # Trainer configuration
    trainer = pl.Trainer(
        max_steps=100,  # Short run for debugging
        log_every_n_steps=1,
        val_check_interval=50,  # Validate every 50 steps
        limit_train_batches=10,  # Only 10 batches per epoch for debugging
        limit_val_batches=5,    # Only 5 validation batches
        precision="16-mixed",
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[ModelSummary(max_depth=2)],
        logger=logger,
        enable_model_summary=False,
        fast_dev_run=False,  # Set to True for super quick test
    )
    
    # Start training
    print("🚀 Starting training...")
    trainer.fit(model, datamodule=data)
    print("✅ Training completed!")


if __name__ == "__main__":
    main()
