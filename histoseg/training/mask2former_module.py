"""
Mask2Former Lightning Module for training and inference.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex

from ..models.encoder.vit_adapter import ViTAdapter
from ..models.decoder.mask2former_pixel_decoder import MSDeformAttnPixelDecoder


class Mask2FormerModule(pl.LightningModule):
    """
    PyTorch Lightning module for Mask2Former semantic segmentation.
    
    Args:
        num_classes (int): Number of segmentation classes
        image_size (Tuple[int, int]): Input image size (H, W)
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay for regularization
        backbone: Backbone encoder configuration
        pixel_decoder: Pixel decoder configuration  
        transformer_decoder: Transformer decoder configuration
    """
    
    def __init__(
        self,
        num_classes: int = 150,
        image_size: Tuple[int, int] = (512, 512),
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        backbone: Optional[Dict[str, Any]] = None,
        pixel_decoder: Optional[Dict[str, Any]] = None,
        transformer_decoder: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Build model components
        self._build_model()
        
        # Setup metrics
        self._setup_metrics()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    def _build_model(self):
        """Build the complete Mask2Former model."""
        # Build backbone
        self.backbone = ViTAdapter(
            model_name="vit_large_patch16_224",
            pretrain_size=224,
            conv_inplane=64,
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
            with_cffn=True,
            cffn_ratio=0.25,
            deform_num_heads=6,
            drop_path_rate=0.1
        )
        
        # Build pixel decoder
        input_shape = {
            "res2": 64,
            "res3": 128, 
            "res4": 256,
            "res5": 512
        }
        
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,
            norm="GN",
            transformer_in_features=["res2", "res3", "res4", "res5"]
        )
        
        # Simple segmentation head for now
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=1)
        )
        
    def _setup_metrics(self):
        """Setup evaluation metrics."""
        metrics = MetricCollection({
            'miou': MulticlassJaccardIndex(
                num_classes=self.num_classes,
                ignore_index=255,
                average='macro'
            ),
        })
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            
        Returns:
            torch.Tensor: Segmentation logits [B, num_classes, H, W]
        """
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Process through pixel decoder
        pixel_decoder_output = self.pixel_decoder(features)
        
        # Generate segmentation predictions
        mask_features = pixel_decoder_output.mask_features
        
        # Upsample to input resolution
        logits = self.seg_head(mask_features)
        logits = F.interpolate(
            logits, 
            size=x.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        targets = batch['mask']
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.train_metrics(preds, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch['image']
        targets = batch['mask']
        
        # Forward pass
        logits = self(images)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.val_metrics(preds, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for bias and norm layers
            if len(param.shape) == 1 or name.endswith('.bias') or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
