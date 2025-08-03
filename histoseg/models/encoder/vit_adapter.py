"""
ViT Adapter for multi-scale feature extraction.

Adapted from InternImage and MMDetection implementations.
Supports TIMM and HuggingFace ViT models like UNI2.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange

try:
    from mmcv.cnn import build_norm_layer
    from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
    from mmcv.runner import BaseModule
    _has_mmcv = True
except ImportError:
    _has_mmcv = False
    BaseModule = nn.Module


class Extractor(nn.Module):
    """Feature extractor module for ViT."""
    
    def __init__(self, dim, num_heads, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0.0, drop_path=0.0, 
                 norm_layer=nn.LayerNorm, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        
        # Multi-scale deformable attention
        from ..ops.ms_deform_attn import MSDeformAttn
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio)
        
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = FFN(
                embed_dims=dim,
                feedforward_channels=int(dim * cffn_ratio),
                num_fcs=2,
                ffn_drop=drop,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path) if drop_path > 0 else None,
                act_cfg=dict(type='GELU'),
                add_identity=True,
            ) if _has_mmcv else self._simple_ffn(dim, int(dim * cffn_ratio), drop)

    def _simple_ffn(self, dim, hidden_dim, drop=0.0):
        """Simple FFN fallback when MMCV is not available."""
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query), reference_points,
                self.feat_norm(feat), spatial_shapes,
                level_start_index, None)
            query = query + attn

            if self.with_cffn:
                if _has_mmcv:
                    query = self.ffn(query, identity=query)
                else:
                    query = query + self.ffn(query)
            return query

        if self.with_cp and query.requires_grad:
            import torch.utils.checkpoint as cp
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InjectionMultiSum(nn.Module):
    """Multi-scale feature injection and summation."""
    
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=nn.LayerNorm, init_values=0.0, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        
        from ..ops.ms_deform_attn import MSDeformAttn
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio)
        
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query), reference_points,
                self.feat_norm(feat), spatial_shapes,
                level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            import torch.utils.checkpoint as cp
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class ViTAdapter(BaseModule):
    """
    Vision Transformer Adapter for multi-scale feature extraction.
    
    This module adapts a standard ViT backbone to produce multi-scale features
    suitable for dense prediction tasks like semantic segmentation.
    
    Args:
        pretrain_size (int): Input size during pretraining. Default: 224
        conv_inplane (int): Channel dimension. Default: 64
        n_points (int): Number of sampling points. Default: 4
        deform_ratio (float): Deformation ratio. Default: 1.0
        interaction_indexes (List[List[int]]): Interaction indexes for different levels
        with_cffn (bool): Whether to use CFFN. Default: True
        cffn_ratio (float): CFFN ratio. Default: 0.25
        deform_num_heads (int): Number of deformable attention heads. Default: 6
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        extractor_dropout (float): Dropout rate in extractor. Default: 0.0
        drop_path_rate (float): Drop path rate. Default: 0.0
        with_cp (bool): Use checkpoint to save memory. Default: False
        model_name (str): Name of the ViT model from TIMM. Default: 'vit_large_patch16_224'
    """
    
    def __init__(self, 
                 pretrain_size: int = 224,
                 conv_inplane: int = 64,
                 n_points: int = 4,
                 deform_ratio: float = 1.0,
                 interaction_indexes: Optional[List[List[int]]] = None,
                 with_cffn: bool = True,
                 cffn_ratio: float = 0.25,
                 deform_num_heads: int = 6,
                 norm_layer: nn.Module = nn.LayerNorm,
                 extractor_dropout: float = 0.0,
                 drop_path_rate: float = 0.0,
                 with_cp: bool = False,
                 model_name: str = 'vit_large_patch16_224',
                 **kwargs):
        super().__init__()
        
        # Default interaction indexes for 4 levels
        if interaction_indexes is None:
            interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
        
        self.interaction_indexes = interaction_indexes
        self.num_levels = len(interaction_indexes)
        self.with_cp = with_cp
        
        # Load ViT backbone
        if 'hf-hub:' in model_name:
            # HuggingFace model via TIMM
            self.backbone = timm.create_model(model_name, pretrained=True, features_only=False)
        else:
            # Standard TIMM model
            self.backbone = timm.create_model(model_name, pretrained=True, features_only=False)
        
        # Get backbone dimensions
        if hasattr(self.backbone, 'embed_dim'):
            embed_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, 'num_features'):
            embed_dim = self.backbone.num_features
        else:
            # Try to infer from a forward pass
            dummy_input = torch.randn(1, 3, pretrain_size, pretrain_size)
            with torch.no_grad():
                dummy_output = self.backbone.forward_features(dummy_input)
                embed_dim = dummy_output.shape[-1]
        
        self.embed_dim = embed_dim
        self.pretrain_size = (pretrain_size, pretrain_size)
        
        # Level embeddings for multi-scale features
        self.level_embeds = nn.Parameter(torch.zeros(self.num_levels, embed_dim))
        self.spm = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(self.num_levels)
        ])
        
        # Interaction modules
        self.interactions = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        
        for i in range(self.num_levels):
            self.interactions.append(
                InjectionMultiSum(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    n_levels=1,
                    deform_ratio=deform_ratio,
                    norm_layer=norm_layer,
                    init_values=0.0,
                    with_cp=with_cp
                )
            )
            
            self.up_projs.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                ) if i < self.num_levels - 1 else nn.Identity()
            )
        
        # Feature projection layers for different scales
        self.conv_outplanes = [conv_inplane * (2 ** i) for i in range(self.num_levels)]
        self.conv_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, outplane, kernel_size=1)
            for outplane in self.conv_outplanes
        ])
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.level_embeds)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _get_reference_points(self, spatial_shapes, device):
        """Generate reference points for deformable attention."""
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
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through ViTAdapter.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            List[torch.Tensor]: Multi-scale features [c1, c2, c3, c4]
        """
        B, C, H, W = x.shape
        
        # Forward through ViT backbone to get intermediate features
        features = []
        
        # Extract patches and add positional encoding
        if hasattr(self.backbone, 'patch_embed'):
            x = self.backbone.patch_embed(x)
            if self.backbone.cls_token is not None:
                cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        else:
            # Alternative for different ViT implementations
            x = self.backbone.forward_features(x, return_all_tokens=True)
        
        # Extract features at different layers
        for i, layer in enumerate(self.backbone.blocks if hasattr(self.backbone, 'blocks') else self.backbone.layers):
            x = layer(x)
            
            # Collect features at interaction indexes
            for level_idx, layer_indexes in enumerate(self.interaction_indexes):
                if i in layer_indexes:
                    # Remove cls token if present
                    feat = x[:, 1:] if self.backbone.cls_token is not None else x
                    
                    # Reshape to spatial format
                    patch_size = self.backbone.patch_embed.patch_size[0] if hasattr(self.backbone, 'patch_embed') else 16
                    feat_h = feat_w = int(math.sqrt(feat.shape[1]))
                    feat = rearrange(feat, 'b (h w) c -> b c h w', h=feat_h, w=feat_w)
                    
                    # Apply level-specific processing
                    feat = feat + self.level_embeds[level_idx].view(1, -1, 1, 1)
                    feat = self.spm[level_idx](feat)
                    
                    if len(features) <= level_idx:
                        features.append([])
                    features[level_idx].append(feat)
        
        # Process multi-scale features
        outs = []
        for level_idx in range(self.num_levels):
            if len(features[level_idx]) > 1:
                # Combine features from multiple layers
                feat_tokens = []
                for feat in features[level_idx]:
                    feat_tokens.append(rearrange(feat, 'b c h w -> b (h w) c'))
                feat_tokens = torch.cat(feat_tokens, dim=1)
                
                # Apply deformable attention
                spatial_shapes = torch.tensor([[feat.shape[2], feat.shape[3]] for feat in features[level_idx]], 
                                            dtype=torch.long, device=x.device)
                level_start_index = torch.cat([
                    torch.tensor([0], dtype=torch.long, device=x.device),
                    spatial_shapes.prod(1).cumsum(0)[:-1]
                ])
                
                reference_points = self._get_reference_points(spatial_shapes, x.device)
                reference_points = reference_points.expand(B, -1, -1, -1)
                
                # Take the first feature as query
                query = rearrange(features[level_idx][0], 'b c h w -> b (h w) c')
                
                enhanced_feat = self.interactions[level_idx](
                    query, reference_points, feat_tokens, spatial_shapes, level_start_index
                )
                
                # Reshape back to spatial format
                feat_h, feat_w = features[level_idx][0].shape[2], features[level_idx][0].shape[3]
                enhanced_feat = rearrange(enhanced_feat, 'b (h w) c -> b c h w', h=feat_h, w=feat_w)
            else:
                enhanced_feat = features[level_idx][0]
            
            # Upsample if needed
            enhanced_feat = self.up_projs[level_idx](enhanced_feat)
            
            # Project to output channels
            enhanced_feat = self.conv_projs[level_idx](enhanced_feat)
            outs.append(enhanced_feat)
        
        return outs
