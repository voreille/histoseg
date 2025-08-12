import json
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F


class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transforms = transforms

        # Load image and annotation file paths
        image_files = sorted(self.image_dir.glob("*.jpg"))
        image_ids = []
        annotation_files = []
        for f in image_files:
            image_id = f.stem
            annotation_file = self.annotation_dir / f"{image_id}.png"
            if not annotation_file.exists():
                raise FileNotFoundError(
                    f"Annotation file {annotation_file} does not exist."
                )
            image_ids.append(image_id)
            annotation_files.append(annotation_file)

        self.dataset = pd.DataFrame(
            {
                "image_id": image_ids,
                "image_path": [str(f) for f in image_files],
                "annotation_path": [str(f) for f in annotation_files],
            }
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, slice):
            idx = idx.start if idx.start is not None else 0

        row = self.dataset.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        annotation = Image.open(row["annotation_path"])
        instance_seg = np.array(annotation)[:, :, 1]  # green channel encodes instances
        class_id_map = np.array(annotation)[:, :, 0]  # red channel encodes semantic cat
        class_labels = np.unique(class_id_map)

        if self.transforms:
            image = self.transforms(image=image)
            annotation = self.transforms(annotation)

        return F.to_tensor(image), F.to_tensor(annotation)