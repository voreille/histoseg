"""
Mask2Former Lightning Module for training and inference.
"""

from typing import Dict, Optional, Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput, )

from ..models.configuration_mask2former import HistosegMask2FormerConfig
from ..models.mask2former_model import Mask2FormerForUniversalSegmentation


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

    def __init__(self,
                 num_classes: int = 150,
                 image_size: Tuple[int, int] = (512, 512),
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.05,
                 config: HistosegMask2FormerConfig = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Build model components
        if config is None:
            config = HistosegMask2FormerConfig()

        self.model = Mask2FormerForUniversalSegmentation(config)

        # Setup metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup evaluation metrics."""
        metrics = MetricCollection({
            'miou':
            MulticlassJaccardIndex(num_classes=self.num_classes,
                                   ignore_index=255,
                                   average='macro'),
        })

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[list[Tensor]] = None,
        class_labels: Optional[list[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerForUniversalSegmentationOutput:
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states,
            output_auxiliary_logits=output_auxiliary_logits,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

    def training_step(self, batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """Training step."""
        pixel_values = batch['pixel_values']
        mask_labels = batch['mask_labels']
        class_labels = batch['class_labels']

        # Forward pass with labels for loss computation
        output = self(pixel_values,
                      mask_labels=mask_labels,
                      class_labels=class_labels)
        loss = output.loss

        # Log training loss
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)

        # For metrics, we need to convert to per-pixel format
        # Get dense predictions for metrics computation
        with torch.no_grad():
            logits = self(pixel_values)  # Get dense predictions without labels
            preds = torch.argmax(logits, dim=1)

            # Convert masks and labels to dense format for metrics
            targets = self._convert_to_dense_targets(mask_labels, class_labels,
                                                     logits.shape[-2:])
            self.train_metrics(preds, targets)

        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
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
        targets = self._convert_to_dense_targets(mask_labels, class_labels,
                                                 logits.shape[-2:])
        self.val_metrics(preds, targets)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True)

        return loss

    def _convert_to_dense_targets(self, mask_labels, class_labels,
                                  target_size):
        """Convert list of masks and labels to dense segmentation targets."""
        batch_size = len(mask_labels)
        device = mask_labels[0].device if len(mask_labels) > 0 and len(
            mask_labels[0]) > 0 else self.device

        dense_targets = []

        for i in range(batch_size):
            # Create empty target
            target = torch.zeros(target_size, dtype=torch.long, device=device)

            masks = mask_labels[i]  # (N, H, W)
            labels = class_labels[i]  # (N,)

            if len(masks) > 0 and len(labels) > 0:
                # Resize masks if needed
                if masks.shape[-2:] != target_size:
                    masks = F.interpolate(masks.float().unsqueeze(0),
                                          size=target_size,
                                          mode='nearest').squeeze(0).bool()

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
            if len(param.shape) == 1 or name.endswith(
                    '.bias') or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [{
            'params': decay_params,
            'weight_decay': self.weight_decay
        }, {
            'params': no_decay_params,
            'weight_decay': 0.0
        }]

        optimizer = torch.optim.AdamW(param_groups,
                                      lr=self.learning_rate,
                                      betas=(0.9, 0.999),
                                      eps=1e-8)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
            div_factor=25.0,
            final_div_factor=10000.0)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
