"""
Multi-Scale Deformable Attention CUDA operations.

This module provides access to the original Mask2Former/Deformable-DETR
CUDA implementations for MSDeformAttn.
"""

# Import the original Mask2Former ops
from .modules import MSDeformAttn
from .functions import MSDeformAttnFunction

# Check if CUDA ops are available
try:
    from .functions.ms_deform_attn_func import CUDA_AVAILABLE
    _has_ops = CUDA_AVAILABLE
except ImportError:
    _has_ops = False

__all__ = ['MSDeformAttn', 'MSDeformAttnFunction']
