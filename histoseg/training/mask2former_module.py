"""
Mask2Former Lightning Module for training and inference.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex

from ..models.mask2former_model import Mask2FormerModel, create_mask2former


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
        self.model = create_mask2former(
            num_classes=self.num_classes,
            model_name="hf-hub:MahmoodLab/UNI2-h",
            hidden_dim=256,
            num_queries=100,
            num_decoder_layers=6,
            num_heads=8,
            dim_feedforward=2048,
            dropout=0.1,
            use_auxiliary_loss=True,
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
    
    def forward(self, pixel_values: torch.Tensor, mask_labels=None, class_labels=None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            pixel_values (torch.Tensor): Input tensor of shape [B, 3, H, W]
            mask_labels (List[torch.Tensor], optional): List of mask labels for training
            class_labels (List[torch.Tensor], optional): List of class labels for training
            
        Returns:
            torch.Tensor or Mask2FormerOutput: Segmentation logits or full output during training
        """
        # During training, use the full model with labels for loss computation
        if self.training and mask_labels is not None and class_labels is not None:
            return self.model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels
            )
        
        # During inference, use the model without labels
        output = self.model(pixel_values=pixel_values)
        
        # For semantic segmentation inference, convert query-based output to dense prediction
        class_logits = output.logits  # [B, Q, C+1]
        mask_logits = output.masks    # [B, Q, H, W]
        
        # Remove no-object class and get per-pixel classification
        class_probs = F.softmax(class_logits[..., :-1], dim=-1)  # [B, Q, C]
        mask_probs = F.sigmoid(mask_logits)  # [B, Q, H, W]
        
        # Combine class and mask predictions
        # Each query contributes to the final prediction weighted by its class confidence
        final_logits = torch.einsum('bqc,bqhw->bchw', class_probs, mask_probs)  # [B, C, H, W]
        
        # Upsample to input resolution if needed
        if final_logits.shape[-2:] != pixel_values.shape[-2:]:
            final_logits = F.interpolate(
                final_logits, 
                size=pixel_values.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        return final_logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        pixel_values = batch['pixel_values']
        mask_labels = batch['mask_labels'] 
        class_labels = batch['class_labels']
        
        # Forward pass with labels for loss computation
        output = self(pixel_values, mask_labels, class_labels)
        loss = output.loss
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # For metrics, we need to convert to per-pixel format
        # Get dense predictions for metrics computation
        with torch.no_grad():
            logits = self(pixel_values)  # Get dense predictions without labels
            preds = torch.argmax(logits, dim=1)
            
            # Convert masks and labels to dense format for metrics
            targets = self._convert_to_dense_targets(mask_labels, class_labels, logits.shape[-2:])
            self.train_metrics(preds, targets)
        
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        pixel_values = batch['pixel_values']
        mask_labels = batch['mask_labels'] 
        class_labels = batch['class_labels']
        
        # Forward pass with labels for loss computation  
        output = self(pixel_values, mask_labels, class_labels)
        loss = output.loss
        
        # Get dense predictions for metrics computation
        logits = self(pixel_values)  # Get dense predictions without labels
        preds = torch.argmax(logits, dim=1)
        
        # Convert masks and labels to dense format for metrics
        targets = self._convert_to_dense_targets(mask_labels, class_labels, logits.shape[-2:])
        self.val_metrics(preds, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def _convert_to_dense_targets(self, mask_labels, class_labels, target_size):
        """Convert list of masks and labels to dense segmentation targets."""
        batch_size = len(mask_labels)
        device = mask_labels[0].device if len(mask_labels) > 0 and len(mask_labels[0]) > 0 else self.device
        
        dense_targets = []
        
        for i in range(batch_size):
            # Create empty target
            target = torch.zeros(target_size, dtype=torch.long, device=device)
            
            masks = mask_labels[i]  # (N, H, W)
            labels = class_labels[i]  # (N,)
            
            if len(masks) > 0 and len(labels) > 0:
                # Resize masks if needed
                if masks.shape[-2:] != target_size:
                    masks = F.interpolate(
                        masks.float().unsqueeze(0), 
                        size=target_size, 
                        mode='nearest'
                    ).squeeze(0).bool()
                
                # Apply masks in order (later masks override earlier ones)
                for mask, label in zip(masks, labels):
                    target[mask] = label
            
            dense_targets.append(target)
        
        return torch.stack(dense_targets)
    
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
