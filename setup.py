import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup

# Try to import pybind11, install if not available
try:
    from pybind11 import get_cmake_dir
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    _has_pybind11 = True
except ImportError:
    print("pybind11 not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
    try:
        from pybind11 import get_cmake_dir
        from pybind11.setup_helpers import Pybind11Extension, build_ext
        _has_pybind11 = True
    except ImportError:
        print("Warning: Could not import pybind11, CUDA extensions will be skipped")
        _has_pybind11 = False
        # Fallback for when pybind11 is not available
        from setuptools import Extension as Pybind11Extension
        from setuptools.command.build_ext import build_ext


def get_extensions():
    """Build MSDeformAttn CUDA extensions."""
    extensions = []
    
    # Skip CUDA extensions for now to avoid setup issues
    # Users can compile manually with ./compile_ops.sh if needed
    print("Skipping CUDA extensions in setup.py (compile manually with ./compile_ops.sh if needed)")
    
    return extensions


class CustomBuildExt(build_ext):
    """Custom build extension to handle CUDA compilation."""
    
    def build_extensions(self):
        # Skip extensions in setup.py
        print("Skipping CUDA extension compilation in setup.py")
        return


setup(
    name="histoseg",
    version="0.1.0",
    description="A modular and extensible codebase for semantic segmentation in histopathology",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/voreille/histoseg",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1",
        "pytorch-lightning>=2.2",
        "timm",
        "transformers",
        "wandb",
        "omegaconf",
        "hydra-core",
        "einops",
        "opencv-python",
        "pillow",
        "numpy",
        "matplotlib",
        "tqdm",
        "scipy",
        "scikit-learn",
        "albumentationsx",
        "pybind11",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
