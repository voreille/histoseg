"""
Mask2Former Pixel Decoder implementation.

Based on the Hugging Face Transformers implementation with modifications
for compatibility with ViTAdapter and modern PyTorch.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelLevelModule,
    Mask2FormerPixelDecoderOutput
)


class ConvModule(nn.Module):
    """Simple ConvModule replacement when MMCV is not available."""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, 
                 norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=norm_cfg is None)
        
        if norm_cfg is not None:
            if norm_cfg['type'] == 'GN':
                self.norm = nn.GroupNorm(norm_cfg['num_groups'], out_channels)
            elif norm_cfg['type'] == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            else:
                self.norm = nn.Identity()
        else:
            self.norm = nn.Identity()
            
        if act_cfg is not None and act_cfg['type'] == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MSDeformAttnPixelDecoder(nn.Module):
    """
    Multi-scale deformable attention pixel decoder.
    
    This is the pixel-level module of Mask2Former that processes multi-scale features
    from the backbone and generates enhanced features for the transformer decoder.
    
    Args:
        input_shape (Dict[str, int]): Dictionary containing feature dimensions
        transformer_dropout (float): Dropout rate for transformer layers
        transformer_nheads (int): Number of attention heads
        transformer_dim_feedforward (int): FFN dimension
        transformer_enc_layers (int): Number of encoder layers
        conv_dim (int): Dimension for convolution layers
        mask_dim (int): Dimension for mask features
        norm (str): Normalization type
        transformer_in_features (List[str]): Input feature names
        common_stride (int): Common stride for features
    """
    
    def __init__(
        self,
        input_shape: Dict[str, int],
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_enc_layers: int = 6,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "GN",
        transformer_in_features: List[str] = ["res2", "res3", "res4", "res5"],
        common_stride: int = 4,
        **kwargs
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.transformer_in_features = transformer_in_features
        self.common_stride = common_stride
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        
        # Build lateral and output convolutions
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.use_bias = norm == ""
        
        for idx, in_feature in enumerate(transformer_in_features):
            if in_feature in input_shape:
                in_channels = input_shape[in_feature]
            else:
                # Default channel dimensions for different levels
                in_channels = [64, 128, 256, 512][idx] if idx < 4 else 512
                
            lateral_conv = ConvModule(
                in_channels, conv_dim, kernel_size=1,
                norm_cfg=dict(type=norm, num_groups=32) if norm else None,
                act_cfg=None
            )
            output_conv = ConvModule(
                conv_dim, conv_dim, kernel_size=3, padding=1,
                norm_cfg=dict(type=norm, num_groups=32) if norm else None,
                act_cfg=dict(type='ReLU')
            )
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        
        # Build positional encoding
        self.positional_encoding = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        
        # Build transformer encoder
        from ..ops.ms_deform_attn import MSDeformAttn
        
        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model=conv_dim,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            n_levels=len(transformer_in_features),
            n_points=4
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, transformer_enc_layers)
        
        # Build mask features projection
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor], masks: Optional[List[torch.Tensor]] = None) -> Mask2FormerPixelDecoderOutput:
        """
        Forward pass through the pixel decoder.
        
        Args:
            features (List[torch.Tensor]): Multi-scale features from backbone
            masks (Optional[List[torch.Tensor]]): Optional input masks
            
        Returns:
            Mask2FormerPixelDecoderOutput: Contains mask features and multi-scale features
        """
        # Build feature pyramid
        multi_scale_features = []
        for idx, (feature, lateral_conv, output_conv) in enumerate(zip(features, self.lateral_convs, self.output_convs)):
            # Lateral connection
            lateral_feature = lateral_conv(feature)
            
            # Upsample to common resolution if needed
            if idx == 0:
                output_feature = lateral_feature
            else:
                # Add upsampled feature from previous level
                prev_feature = F.interpolate(
                    multi_scale_features[-1], 
                    size=lateral_feature.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
                output_feature = lateral_feature + prev_feature
            
            # Apply output convolution
            output_feature = output_conv(output_feature)
            multi_scale_features.append(output_feature)
        
        # Prepare features for transformer encoder
        batch_size = features[0].shape[0]
        
        # Flatten spatial dimensions and concatenate multi-scale features
        encoder_inputs = []
        spatial_shapes = []
        
        for feature in multi_scale_features:
            h, w = feature.shape[-2:]
            spatial_shapes.append((h, w))
            # Flatten: (B, C, H, W) -> (B, H*W, C)
            feature_flat = feature.flatten(2).transpose(1, 2)
            encoder_inputs.append(feature_flat)
        
        # Concatenate all features: (B, sum(H*W), C)
        encoder_input = torch.cat(encoder_inputs, dim=1)
        
        # Generate positional encodings for each scale
        pos_embeds = []
        for feature in multi_scale_features:
            pos_embed = self.positional_encoding(feature)
            pos_embed_flat = pos_embed.flatten(2).transpose(1, 2)
            pos_embeds.append(pos_embed_flat)
        pos_embed = torch.cat(pos_embeds, dim=1)
        
        # Create spatial shapes and level start index tensors
        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=encoder_input.device)
        level_start_index = torch.cat([
            torch.tensor([0], dtype=torch.long, device=encoder_input.device),
            spatial_shapes_tensor.prod(1).cumsum(0)[:-1]
        ])
        
        # Apply transformer encoder
        encoder_output = self.encoder(
            encoder_input,
            spatial_shapes=spatial_shapes_tensor,
            level_start_index=level_start_index,
            pos_embed=pos_embed
        )
        
        # Reshape encoder output back to spatial format for each scale
        start_idx = 0
        enhanced_features = []
        for i, (h, w) in enumerate(spatial_shapes):
            end_idx = start_idx + h * w
            feature_output = encoder_output[:, start_idx:end_idx, :]  # (B, H*W, C)
            feature_output = feature_output.transpose(1, 2).reshape(batch_size, self.conv_dim, h, w)
            enhanced_features.append(feature_output)
            start_idx = end_idx
        
        # Generate mask features from the finest scale
        mask_features = self.mask_features(enhanced_features[0])
        
        return Mask2FormerPixelDecoderOutput(
            mask_features=mask_features,
            multi_scale_features=enhanced_features
        )


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * 3.14159
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    """Multi-scale deformable attention transformer encoder layer."""
    
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0, 
                 n_levels=4, n_points=4):
        super().__init__()
        
        # Multi-scale deformable attention
        from ..ops.ms_deform_attn import MSDeformAttn
        self.self_attn = MSDeformAttn(d_model, n_levels, nhead, n_points)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU(inplace=True)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, spatial_shapes, level_start_index, pos=None, padding_mask=None):
        # Generate reference points
        device = src.device
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None].repeat(1, 1, len(spatial_shapes), 1)
        
        # Self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), 
            reference_points, 
            src, 
            spatial_shapes, 
            level_start_index, 
            padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    """Multi-scale deformable attention transformer encoder."""
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, spatial_shapes, level_start_index, pos_embed=None, padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, spatial_shapes, level_start_index, pos_embed, padding_mask)
        return output
