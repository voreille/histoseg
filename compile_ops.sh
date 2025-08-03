#!/bin/bash

# Script to compile MSDeformAttn CUDA operations using the original Mask2Former ops

set -e

echo "Compiling MSDeformAttn CUDA operations from original Mask2Former repo..."

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

# Navigate to ops directory and compile
echo "Building CUDA extensions using original Mask2Former setup..."
cd histoseg/models/ops
./make.sh

echo "MSDeformAttn CUDA operations compiled successfully!"

# Test the compilation
echo "Testing compilation..."
python -c "
try:
    import MultiScaleDeformableAttention
    print('✓ MultiScaleDeformableAttention import successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo "✓ CUDA ops compilation and test completed successfully!"
