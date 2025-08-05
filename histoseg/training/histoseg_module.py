"""
HistoSeg Lightning Module.

Main training module adapted from benchmark-vfm-ss repository.
Provides training, validation, and testing functionality for semantic segmentation.
"""

import pytorch_lightning as lightning
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np


class HistoSegModule(lightning.LightningModule):
    """
    HistoSeg Lightning Module for semantic segmentation.
    
    Args:
        img_size (tuple[int, int]): Input image size (H, W)
        freeze_encoder (bool): Whether to freeze encoder parameters
        network (nn.Module): The segmentation network (e.g., Mask2Former)
        weight_decay (float): Weight decay for optimizer
        lr (float): Learning rate
        lr_multiplier_encoder (float): Learning rate multiplier for encoder
        num_classes (int): Number of segmentation classes
        ignore_idx (int): Index to ignore in loss computation
        num_metrics (int): Number of metrics to track
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        freeze_encoder: bool,
        network: nn.Module,
        weight_decay: float,
        lr: float,
        lr_multiplier_encoder: float,
        num_classes: int = 150,
        ignore_idx: int = 255,
        num_metrics: int = 1,
    ):
        super().__init__()

        self.img_size = img_size
        self.network = network
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_multiplier_encoder = lr_multiplier_encoder
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx

        # Freeze encoder if requested
        for param in self.network.encoder.parameters():
            param.requires_grad = not freeze_encoder

        # Disable compiler for logging to avoid issues
        self.log = torch.compiler.disable(self.log)  # type: ignore

        # Initialize metrics
        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)

    def init_metrics_semantic(self, num_classes, ignore_idx, num_metrics):
        """Initialize semantic segmentation metrics."""
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_metrics)
            ]
        )

    @torch.compiler.disable
    def update_metrics(
        self, preds: list[torch.Tensor], targets: list[torch.Tensor], dataloader_idx
    ):
        """Update metrics with predictions and targets."""
        for i in range(len(preds)):
            self.metrics[dataloader_idx].update(
                preds[i][None, ...], targets[i][None, ...]
            )

    def forward(self, imgs):
        """Forward pass through the network."""
        # Normalize images to [0, 1] range
        x = imgs / 255.0

        output = self.network(x)

        # During inference, return the final output
        if not self.training and isinstance(output, tuple):
            return (y[-1] for y in self.network(x))

        return output

    def training_step(self, batch, batch_idx):
        """Training step."""
        imgs, targets = batch
        
        # Forward pass
        outputs = self(imgs)
        
        # Compute loss (assuming the network returns loss when training)
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        else:
            # If network doesn't compute loss, you'll need to add loss computation here
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Validation step."""
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def eval_step(self, batch, batch_idx, dataloader_idx, log_prefix):
        """Generic evaluation step for validation/test."""
        imgs, targets = batch
        
        # Forward pass
        with torch.no_grad():
            outputs = self(imgs)
            
        # Convert outputs to per-pixel predictions
        if isinstance(outputs, dict):
            # For Mask2Former-style outputs
            if "pred_masks" in outputs and "pred_logits" in outputs:
                per_pixel_logits = self.to_per_pixel_logits_semantic(
                    outputs["pred_masks"], outputs["pred_logits"]
                )
            else:
                # Fallback
                per_pixel_logits = outputs.get("logits", outputs)
        else:
            per_pixel_logits = outputs
            
        # Get per-pixel targets
        per_pixel_targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        
        # Convert to predictions
        preds = [torch.argmax(logits, dim=0) for logits in per_pixel_logits]
        
        # Update metrics
        self.update_metrics(preds, per_pixel_targets, dataloader_idx)
        
        # Log example visualization occasionally
        if batch_idx == 0 and hasattr(self.logger, "experiment"):
            try:
                plot_img = self.plot_semantic(
                    imgs[0], per_pixel_targets[0], per_pixel_logits[0]
                )
                self.logger.experiment.log({
                    f"{log_prefix}_visualization": plot_img
                })
            except Exception:
                pass  # Skip visualization if it fails
                
        return {"preds": preds, "targets": per_pixel_targets}

    def on_train_batch_start(self, batch, batch_idx):
        """Log learning rates at the start of each training batch."""
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(
                f"group_{i}_lr_{len(param_group['params'])}",
                param_group["lr"],
                on_step=True,
            )

    def on_save_checkpoint(self, checkpoint):
        """Clean up compiled model state dict for saving."""
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        self._on_eval_epoch_end_semantic("val")

    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        self._on_eval_epoch_end_semantic("test")

    def _on_eval_epoch_end_semantic(self, log_prefix):
        """Compute and log semantic segmentation metrics."""
        miou_per_dataset = []
        iou_per_dataset_per_class = []
        
        for metric_idx, metric in enumerate(self.metrics):
            iou_per_dataset_per_class.append(metric.compute())
            metric.reset()

            # Log per-class IoU
            for iou_idx, iou in enumerate(iou_per_dataset_per_class[-1]):
                self.log(
                    f"{log_prefix}_{metric_idx}_iou_{iou_idx}", iou, sync_dist=True
                )

            # Compute and log mIoU
            miou_per_dataset.append(float(iou_per_dataset_per_class[-1].mean()))
            self.log(
                f"{log_prefix}_{metric_idx}_miou", miou_per_dataset[-1], sync_dist=True
            )

    def configure_optimizers(self):
        """Configure optimizers with different learning rates for encoder and decoder."""
        encoder_param_names = {
            name for name, _ in self.network.encoder.named_parameters()
        }
        base_params = []
        encoder_params = []

        for name, param in self.named_parameters():
            if name.replace("network.encoder.", "") in encoder_param_names:
                encoder_params.append(param)
            else:
                base_params.append(param)

        return AdamW(
            [
                {"params": base_params, "lr": self.lr},
                {
                    "params": encoder_params,
                    "lr": self.lr * self.lr_multiplier_encoder,
                },
            ],
            weight_decay=self.weight_decay,
        )

    @torch.compiler.disable
    def plot_semantic(self, img, target, logits=None, cmap="tab20"):
        """Create visualization plot for semantic segmentation."""
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        # Plot original image
        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].set_title("Image")
        axes[0].axis("off")

        # Plot target
        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = None
        if logits is not None:
            preds = torch.argmax(logits, dim=0).cpu().numpy()
            unique_classes = np.unique(
                np.concatenate((unique_classes, np.unique(preds)))
            )

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        # Set ignore index to black
        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        # Plot target segmentation
        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].set_title("Target")
        axes[1].axis("off")

        # Plot prediction if available
        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].set_title("Prediction")
            axes[2].axis("off")

        # Add legend
        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]
        fig.legend(handles=patches, loc="upper left")

        # Convert to PIL Image
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="white")
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf)

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        """Convert mask and class logits to per-pixel logits."""
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(targets: list[dict], ignore_idx):
        """Convert instance targets to per-pixel targets."""
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets
