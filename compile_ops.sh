#!/bin/bash

# Script to compile MSDeformAttn CUDA operations

set -e

echo "Compiling MSDeformAttn CUDA operations..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "NVCC not found. Please install CUDA toolkit."
    exit 1
fi

# Check if PyTorch is installed
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "PyTorch not found. Please install PyTorch first."
    exit 1
}

# Check if CUDA is available in PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'" || {
    echo "CUDA not available in PyTorch. Please install PyTorch with CUDA support."
    exit 1
}

# Build the extensions
echo "Building CUDA extensions..."
python setup.py build_ext --inplace

echo "MSDeformAttn CUDA operations compiled successfully!"
