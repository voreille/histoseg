from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig

# Import HuggingFace components (required dependency)
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
    Mask2FormerLoss,
    Mask2FormerModelOutput,
    Mask2FormerPixelDecoder,
    Mask2FormerPixelLevelModuleOutput,
    Mask2FormerPreTrainedModel,
    Mask2FormerTransformerModule,
)

from histoseg.models.encoder.encoders import create_encoder


class HistosegMask2FormerConfig(Mask2FormerConfig):
    """
    Extended Mask2Former configuration that includes ViTAdapter settings.

    This config inherits from HuggingFace's Mask2FormerConfig and adds
    ViTAdapter-specific parameters.
    """

    def __init__(
        self,
        # Encoder configuration (more general than ViT-specific)
        encoder_type: str = "vit_adapter",  # "vit_adapter", "resnet", "swin", etc.
        encoder_name: str = "hf-hub:MahmoodLab/UNI2-h",
        encoder_embed_dim: int = None,  # Auto-detected based on encoder
        encoder_pretrain_size: int = 224,
        encoder_freeze: bool = False,  # Whether to freeze the encoder
        encoder_kwargs: Dict = None,  # Encoder-specific parameters
        # Multi-scale feature extraction settings
        encoder_out_indices: List[int] = None,  # Which layers to extract features from
        encoder_feature_channels: List[
            int
        ] = None,  # Expected output channels per scale
        # Override backbone-related defaults
        backbone_config: Optional[Dict] = None,
        backbone: Optional[str] = None,
        use_pretrained_backbone: bool = False,
        # Standard Mask2Former parameters with better defaults
        num_queries: int = 100,
        max_seq_length: int = 77,
        encoder_layers: int = 6,
        encoder_ffn_dim: int = 2048,
        encoder_attention_heads: int = 8,
        decoder_layers: int = 6,
        decoder_ffn_dim: int = 2048,
        decoder_attention_heads: int = 8,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_function: str = "relu",
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        use_auxiliary_loss: bool = True,
        feature_size: int = 256,
        mask_feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_feedforward_dim: int = 1024,
        common_stride: int = 4,
        feature_strides: List[int] = None,
        **kwargs,
    ):
        # Set defaults
        if encoder_out_indices is None:
            if encoder_type == "vit_adapter":
                encoder_out_indices = [
                    3,
                    5,
                    7,
                    11,
                ]  # Multi-scale output indices for ViT
            elif encoder_type == "resnet":
                encoder_out_indices = [1, 2, 3, 4]  # ResNet stages
            else:
                encoder_out_indices = [1, 2, 3, 4]  # Default

        if feature_strides is None:
            feature_strides = [4, 8, 16, 32]

        if encoder_kwargs is None:
            encoder_kwargs = {}

        if encoder_feature_channels is None:
            encoder_feature_channels = [256, 512, 1024, 2048]  # ResNet-like progression

        # Auto-detect encoder parameters based on encoder type and name
        encoder_embed_dim, encoder_config = self._get_encoder_config(
            encoder_type, encoder_name, encoder_kwargs
        )

        # Store encoder config
        self.encoder_type = encoder_type
        self.encoder_name = encoder_name
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_pretrain_size = encoder_pretrain_size
        self.encoder_freeze = encoder_freeze
        self.encoder_kwargs = encoder_config
        self.encoder_out_indices = encoder_out_indices
        self.encoder_feature_channels = encoder_feature_channels

        # Initialize parent
        super().__init__(
            backbone_config=backbone_config,
            backbone=backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            num_queries=num_queries,
            max_seq_length=max_seq_length,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_function=activation_function,
            init_std=init_std,
            init_xavier_std=init_xavier_std,
            use_auxiliary_loss=use_auxiliary_loss,
            feature_size=feature_size,
            mask_feature_size=mask_feature_size,
            hidden_dim=hidden_dim,
            encoder_feedforward_dim=encoder_feedforward_dim,
            common_stride=common_stride,
            feature_strides=feature_strides,
            **kwargs,
        )

    def get_encoder_config(self) -> Dict:
        """Get encoder configuration dictionary."""
        return {
            "encoder_type": self.encoder_type,
            "encoder_name": self.encoder_name,
            "embed_dim": self.encoder_embed_dim,
            "pretrain_size": self.encoder_pretrain_size,
            "out_indices": self.encoder_out_indices,
            "freeze_encoder": self.encoder_freeze,
            "feature_channels": self.encoder_feature_channels,
            **self.encoder_kwargs,
        }


class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: HistosegMask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://huggingface.co/papers/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()

        self.encoder = create_encoder(config)
        self.decoder = Mask2FormerPixelDecoder(
            config, feature_channels=self.encoder.channels
        )

    def forward(
        self, pixel_values: Tensor, output_hidden_states: bool = False
    ) -> Mask2FormerPixelLevelModuleOutput:
        # backbone_features = self.encoder(pixel_values).feature_maps
        backbone_features = self.encoder(pixel_values)
        decoder_output = self.decoder(
            backbone_features, output_hidden_states=output_hidden_states
        )

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features)
            if output_hidden_states
            else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


class Mask2FormerModel(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(
            in_features=config.feature_size, config=config
        )

        self.post_init()

    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerModelOutput:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones(
                (batch_size, height, width), device=pixel_values.device
            )

        pixel_level_module_output = self.pixel_level_module(
            pixel_values=pixel_values, output_hidden_states=output_hidden_states
        )

        transformer_module_output = self.transformer_module(
            multi_scale_features=pixel_level_module_output.decoder_hidden_states,
            mask_features=pixel_level_module_output.decoder_last_hidden_state,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = (
                pixel_level_module_output.decoder_hidden_states
            )
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = (
                transformer_module_output.intermediate_hidden_states
            )

        output = Mask2FormerModelOutput(
            encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)

        return output


class Mask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.model = Mask2FormerModel(config)

        self.weight_dict: dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        self.post_init()

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        loss_dict: dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: list[dict[str, Tensor]] = []

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append(
                {
                    "masks_queries_logits": aux_binary_masks,
                    "class_queries_logits": aux_classes,
                }
            )

        return auxiliary_logits

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

        """
        mask_labels (`list[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`list[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.
        output_auxiliary_logits (`bool`, *optional*):
            Whether or not to output auxiliary logits.

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs.transformer_decoder_intermediate_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(
            class_queries_logits, masks_queries_logits
        )

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = (
                outputs.transformer_decoder_hidden_states
            )

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits
            if output_auxiliary_logits is None
            else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss) + output
        return output


__all__ = [
    "Mask2FormerForUniversalSegmentation",
    "Mask2FormerModel",
]
