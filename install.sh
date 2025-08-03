#!/bin/bash

# Installation script for histoseg
# Handles dependencies in the correct order

set -e

echo "Installing histoseg..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required. Found $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Install pybind11 first (required for setup.py)
echo "Installing pybind11..."
pip install pybind11

# Install other requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing histoseg in development mode..."
pip install -e .

echo "Installation complete! ✓"

# Check if CUDA is available for compilation
echo ""
echo "Checking CUDA availability..."
python -c "
import torch
if torch.cuda.is_available():
    print('CUDA available ✓')
    print('You can compile CUDA operations with: ./compile_ops.sh')
else:
    print('CUDA not available - will use PyTorch fallback')
"

echo ""
echo "Quick test:"
python -c "
try:
    import histoseg
    print('histoseg import successful ✓')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
"

echo ""
echo "Installation successful! You can now:"
echo "  - Run training: python cli.py fit --config configs/ade20k.yaml"
echo "  - Prepare data: python scripts/prepare_ade20k.py"
echo "  - Compile CUDA ops: ./compile_ops.sh (if CUDA available)"
