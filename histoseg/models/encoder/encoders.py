from typing import Dict, List

import torch
import torch.nn as nn
import timm

from histoseg.models.encoder.vit_adapter import ViTAdapter
from histoseg.models.configuration_mask2former import HistosegMask2FormerConfig

DEFAULT_HISTOSEG_BACKBONE_CONFIG = {
    "encoder_type": "vit_adapter",
    "embed_dim": 768,
    "out_indices": [0, 1, 2, 3],
    "freeze_encoder": True,
    "feature_channels": [64, 128, 256, 512],
    "timm_name": "hf-hub:MahmoodLab/UNI2-h",
    "timm_kwargs": {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
}


class BaseEncoder(nn.Module):
    """Base class for all encoders."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.out_indices = config["out_indices"]
        self.freeze_encoder = config.get("freeze_encoder", False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            List of feature tensors at different scales
        """
        raise NotImplementedError

    def freeze(self):
        """Freeze encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False


def create_encoder(config: HistosegMask2FormerConfig) -> BaseEncoder:
    """
    Factory function to create an encoder based on the configuration.

    Args:
        config (HistosegMask2FormerConfig): Configuration object containing encoder settings.

    Returns:
        nn.Module: The instantiated encoder model.
    """
    histoseg_backbone_config = config.get("histoseg_backbone_config",
                                          DEFAULT_HISTOSEG_BACKBONE_CONFIG)
    encoder_type = histoseg_backbone_config.get("encoder_type", None)

    if encoder_type == "vit_adapter":
        return ViTAdaptedEncoder(histoseg_backbone_config)
    elif encoder_type == "resnet":
        return ResNetEncoder(histoseg_backbone_config)
    elif encoder_type == "swin":
        return SwinEncoder(histoseg_backbone_config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class ViTAdaptedEncoder(BaseEncoder):
    """ViT-based encoder using ViTAdapter."""

    def __init__(self, config: Dict):
        super().__init__(config)

        # Create ViTAdapter with config
        vit_config = {
            k: v
            for k, v in config.items()
            if k not in ["encoder_type", "feature_channels", "freeze_encoder"]
        }
        self.encoder = ViTAdapter(**vit_config)

        if self.freeze_encoder:
            self.freeze()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through ViTAdapter."""
        return self.encoder(x)


class ResNetEncoder(BaseEncoder):
    """ResNet-based encoder."""

    def __init__(self, config: Dict):
        super().__init__(config)

        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision is required for ResNet encoder")

        model_name = config["encoder_name"]
        pretrained = config.get("pretrained", True)

        # Create ResNet model
        if "resnet18" in model_name.lower():
            self.backbone = models.resnet18(pretrained=pretrained)
        elif "resnet34" in model_name.lower():
            self.backbone = models.resnet34(pretrained=pretrained)
        elif "resnet50" in model_name.lower():
            self.backbone = models.resnet50(pretrained=pretrained)
        elif "resnet101" in model_name.lower():
            self.backbone = models.resnet101(pretrained=pretrained)
        elif "resnet152" in model_name.lower():
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ResNet variant: {model_name}")

        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Get feature dimensions for each stage
        self._setup_feature_extraction()

        if self.freeze_encoder:
            self.freeze()

    def _setup_feature_extraction(self):
        """Setup hooks for multi-scale feature extraction."""
        self.features = {}

        # Register hooks for different ResNet stages
        def get_activation(name):

            def hook(model, input, output):
                self.features[name] = output

            return hook

        # ResNet stages: conv1, layer1, layer2, layer3, layer4
        stages = [
            self.backbone[4],
            self.backbone[5],
            self.backbone[6],
            self.backbone[7],
        ]
        for i, stage in enumerate(stages):
            stage.register_forward_hook(get_activation(f"stage_{i + 1}"))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through ResNet."""
        self.features.clear()
        _ = self.backbone(x)

        # Return features in order [stage1, stage2, stage3, stage4]
        return [self.features[f"stage_{i + 1}"] for i in range(4)]


class SwinEncoder(BaseEncoder):
    """Swin Transformer-based encoder."""

    def __init__(self, config: Dict):
        super().__init__(config)

        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for Swin encoder")

        model_name = config["encoder_name"]
        pretrained = config.get("pretrained", True)

        # Create Swin model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        if self.freeze_encoder:
            self.freeze()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through Swin Transformer."""
        features = self.backbone(x)
        return features
