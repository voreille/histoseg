import re
import json
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple
import torch
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import pandas as pd


class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transforms = transforms

        # Load image and annotation file paths
        self.image_files = sorted(self.image_dir.glob("*.jpg"))
        self.annotation_files = sorted(self.annotation_dir.glob("*.png"))
        self

        if len(self.image_files) != len(self.annotation_files):
            raise ValueError("Number of images and annotations do not match.")

