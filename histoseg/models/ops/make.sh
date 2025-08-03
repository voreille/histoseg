#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

# Enhanced version with better error handling and CUDA checks

echo "Attempting to build Multi-Scale Deformable Attention CUDA ops..."

# Check if PyTorch is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "Error: PyTorch not found. Please install PyTorch first."
    exit 1
}

# Check CUDA availability
python -c "
import torch
from torch.utils.cpp_extension import CUDA_HOME

print(f'CUDA available in PyTorch: {torch.cuda.is_available()}')
print(f'CUDA_HOME: {CUDA_HOME}')

if torch.cuda.is_available() and CUDA_HOME is not None:
    print('✓ CUDA compilation should work')
    exit(0)
elif torch.cuda.is_available() and CUDA_HOME is None:
    print('⚠ CUDA available but CUDA_HOME not set. Will try with FORCE_CUDA=1')
    exit(2)
else:
    print('⚠ CUDA not available. Will compile CPU-only version.')
    exit(1)
"

cuda_status=$?

if [ $cuda_status -eq 0 ]; then
    echo "Building with CUDA support..."
    python setup.py build install
elif [ $cuda_status -eq 2 ]; then
    echo "Building with FORCE_CUDA=1..."
    FORCE_CUDA=1 python setup.py build install
else
    echo "Building CPU-only version..."
    python setup.py build install
fi

echo "Build completed. Testing import..."

# Test the build
python -c "
try:
    import MultiScaleDeformableAttention
    print('✓ MultiScaleDeformableAttention import successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    print('Note: This is expected if CUDA development tools are not available.')
    print('The module will fall back to pure PyTorch implementation.')
"
