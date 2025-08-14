from typing import Dict, List, Optional

from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig


class HistosegMask2FormerConfig(Mask2FormerConfig):
    """
    Extended Mask2Former configuration that includes ViTAdapter settings.

    This config inherits from HuggingFace's Mask2FormerConfig and adds
    ViTAdapter-specific parameters.
    """

    def __init__(
        self,
        histoseg_backbone_config: Optional[Dict] = None,
        backbone_config: Optional[Dict] = None,
        backbone: Optional[str] = None,
        use_pretrained_backbone: bool = False,
        feature_size: int = 256,
        mask_feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_feedforward_dim: int = 1024,
        activation_function: str = "relu",
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
        enforce_input_projection: bool = False,
        common_stride: int = 4,
        ignore_value: int = 255,
        num_queries: int = 100,
        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        use_auxiliary_loss: bool = True,
        feature_strides: list[int] = [4, 8, 16, 32],
        output_auxiliary_logits: Optional[bool] = None,
        use_timm_backbone: bool = False,
        backbone_kwargs: Optional[dict] = None,
        **kwargs,
    ):

        # Initialize parent
        super().__init__(
            backbone_config=backbone_config,
            backbone=backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            feature_size=feature_size,
            mask_feature_size=mask_feature_size,
            hidden_dim=hidden_dim,
            encoder_feedforward_dim=encoder_feedforward_dim,
            activation_function=activation_function,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            pre_norm=pre_norm,
            enforce_input_projection=enforce_input_projection,
            common_stride=common_stride,
            ignore_value=ignore_value,
            num_queries=num_queries,
            no_object_weight=no_object_weight,
            class_weight=class_weight,
            mask_weight=mask_weight,
            dice_weight=dice_weight,
            train_num_points=train_num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            init_std=init_std,
            init_xavier_std=init_xavier_std,
            use_auxiliary_loss=use_auxiliary_loss,
            feature_strides=feature_strides,
            output_auxiliary_logits=output_auxiliary_logits,
            use_timm_backbone=use_timm_backbone,
            backbone_kwargs=backbone_kwargs,
            **kwargs,
        )
        self.histoseg_backbone_config = histoseg_backbone_config
