"""
ViT Adapter for multi-scale feature extraction.

Adapted from InternImage and MMDetection implementations.
Supports TIMM and HuggingFace ViT models like UNI2.
"""

import math
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn.init import normal_, trunc_normal_

from .adapter_modules import (
    InteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)
from .vit_backbone import TIMMVisionTransformer
from ..ops import MSDeformAttn


class ViTAdapter(nn.Module):
    """
    Vision Transformer Adapter for multi-scale feature extraction.

    Args:
        pretrain_size (int): Input size during pretraining. Default: 224
        num_heads (int): Number of heads in backbone ViT. Default: 12
        conv_inplane (int): Channel dimension. Default: 64
        n_points (int): Number of sampling points. Default: 4
        deform_num_heads (int): Number of deformable attention heads. Default: 6
        init_values (float): Init value for gamma. Default: 0.
        interaction_indexes (List[List[int]]): Interaction indexes for different levels
        with_cffn (bool): Whether to use CFFN. Default: True
        cffn_ratio (float): CFFN ratio. Default: 0.25
        deform_ratio (float): Deformation ratio. Default: 1.0
        add_vit_feature (bool): Whether to add ViT features. Default: True
        pretrained (str): Pretrained model path. Default: None
        use_extra_extractor (bool): Whether to use extra extractor. Default: True
        with_cp (bool): Use checkpoint to save memory. Default: False
    """

    def __init__(
        self,
        pretrain_size=224,
        num_heads=12,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        pretrained=None,
        use_extra_extractor=True,
        with_cp=False,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        model_name=None,
        timm_kwargs=None,
    ):
        super().__init__()

        # Default interaction indexes
        if interaction_indexes is None:
            interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]

        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.num_block = depth
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.with_cp = with_cp

        # Initialize ViT backbone
        if model_name and "hf-hub:" in model_name:
            # Load from HuggingFace via TIMM
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                features_only=False,
                **timm_kwargs,
            )
            embed_dim = getattr(self.backbone, "embed_dim", embed_dim)
        elif model_name:
            # Load standard TIMM model
            self.backbone = timm.create_model(model_name,
                                              pretrained=True,
                                              features_only=False)
            embed_dim = getattr(self.backbone, "embed_dim", embed_dim)
        else:
            # Create custom ViT
            self.backbone = TIMMVisionTransformer(
                img_size=pretrain_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                pretrained=pretrained,
                with_cp=with_cp,
            )

        self.embed_dim = embed_dim

        # Remove cls_token for dense prediction
        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token = None

        # Level embeddings and spatial prior module
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim,
                                      with_cp=False)

        # Interaction blocks
        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) -
                                  1 else False) and use_extra_extractor),
                with_cp=with_cp,
            ) for i in range(len(interaction_indexes))
        ])

        # Upsampling and normalization
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        # Initialize weights
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(1, self.pretrain_size[0] // 16,
                                      self.pretrain_size[1] // 16,
                                      -1).permute(0, 3, 1, 2)
        pos_embed = (F.interpolate(pos_embed,
                                   size=(H, W),
                                   mode="bicubic",
                                   align_corners=False).reshape(1, -1,
                                                                H * W).permute(
                                                                    0, 2, 1))
        return pos_embed

    def _init_deform_weights(self, m):

        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through ViTAdapter.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            List[torch.Tensor]: Multi-scale features [f1, f2, f3, f4]
        """
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        if hasattr(self.backbone, "patch_embed"):
            x, H, W = self.backbone.patch_embed(x)
            bs, n, dim = x.shape
            pos_embed = self._get_pos_embed(self.backbone.pos_embed, H, W)
            x = self.backbone.pos_drop(x + pos_embed)
        else:
            # Fallback for different ViT implementations
            x = self.backbone.forward_features(x)
            bs, n, dim = x.shape
            H = W = int(math.sqrt(n))

        # Get ViT blocks
        if hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
        else:
            # Fallback for different implementations
            blocks = []

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if len(blocks) > 0:
                interaction_blocks = blocks[indexes[0]:indexes[-1] + 1]
            else:
                interaction_blocks = []
            x, c = layer(x, c, interaction_blocks, deform_inputs1,
                         deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1,
                               scale_factor=4,
                               mode="bilinear",
                               align_corners=False)
            x2 = F.interpolate(x2,
                               scale_factor=2,
                               mode="bilinear",
                               align_corners=False)
            x4 = F.interpolate(x4,
                               scale_factor=0.5,
                               mode="bilinear",
                               align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
