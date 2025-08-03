"""IO utilities for data loading and saving."""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import numpy as np
from PIL import Image


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_image(filepath: Union[str, Path], mode: str = 'RGB') -> Image.Image:
    """Load image file."""
    return Image.open(filepath).convert(mode)


def save_image(image: Union[Image.Image, np.ndarray, torch.Tensor], 
               filepath: Union[str, Path]) -> None:
    """Save image to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    image.save(filepath)


def ensure_dir(dirpath: Union[str, Path]) -> None:
    """Ensure directory exists."""
    os.makedirs(dirpath, exist_ok=True)


def list_files(dirpath: Union[str, Path], extensions: List[str] = None) -> List[str]:
    """List files in directory with optional extension filtering."""
    dirpath = Path(dirpath)
    if not dirpath.exists():
        return []
    
    files = []
    for file in dirpath.iterdir():
        if file.is_file():
            if extensions is None or file.suffix.lower() in extensions:
                files.append(str(file))
    
    return sorted(files)
