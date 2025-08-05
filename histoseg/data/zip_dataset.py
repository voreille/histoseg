"""
Generic dataset class for zip-based datasets.
Adapted from benchmark-vfm-ss repository.
"""

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


class ZipDataset(torch.utils.data.Dataset):
    """
    Generic dataset that loads images and targets from zip files.
    Supports various dataset formats including ADE20K, Cityscapes, etc.
    """

    def __init__(
        self,
        zip_path: Path,
        img_suffix: str,
        target_suffix: str,
        ignore_idx: Optional[int] = None,
        transforms: Optional[Callable] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_zip_path: Optional[Path] = None,
        img_folder_path_in_zip: Path = Path("./"),
        target_folder_path_in_zip: Path = Path("./"),
        annotations_json_path_in_zip: Optional[Path] = None,
        target_zip_path_in_zip: Optional[Path] = None,
        class_mapping: Optional[dict] = None,
    ):
        self.zip_path = zip_path
        self.target_zip_path = target_zip_path
        self.target_folder_path_in_zip = target_folder_path_in_zip
        self.target_zip_path_in_zip = target_zip_path_in_zip
        self.ignore_idx = ignore_idx
        self.transforms = transforms
        self.class_mapping = class_mapping

        self.zip = None
        self.target_zip = None
        img_zip, target_zip = self._load_zips()

        self.annotations_dict = {}
        if annotations_json_path_in_zip is not None:
            with zipfile.ZipFile(target_zip_path or zip_path) as outer_target_zip:
                with outer_target_zip.open(
                    str(annotations_json_path_in_zip), "r"
                ) as file:
                    annotations_json = json.load(file)

            self.class_mapping = {
                category["id"]: idx
                for idx, category in enumerate(annotations_json["categories"])
            }

            for annotation in annotations_json["annotations"]:
                self.annotations_dict[annotation["file_name"]] = {}
                for segment_info in annotation["segments_info"]:
                    self.annotations_dict[annotation["file_name"]][
                        segment_info["id"]
                    ] = segment_info["category_id"]

        self.imgs = []
        self.targets = []

        for img_info in sorted(img_zip.infolist(), key=self._sort_key):
            if not self.valid_member(
                img_info, img_folder_path_in_zip, img_stem_suffix, img_suffix
            ):
                continue

            rel_path = Path(img_info.filename).relative_to(img_folder_path_in_zip)
            target_parent = target_folder_path_in_zip / rel_path.parent
            target_stem = rel_path.stem.replace(img_stem_suffix, target_stem_suffix)
            target_filename = str(target_parent / (target_stem + target_suffix))

            if self.annotations_dict:
                if not self.annotations_dict.get(target_stem + target_suffix):
                    continue
            else:
                if target_filename not in target_zip.namelist():
                    continue

                with target_zip.open(target_filename) as target_file:
                    min_val, max_val = Image.open(target_file).getextrema()
                    if min_val == max_val:
                        continue

            self.imgs.append(img_info.filename)
            self.targets.append(target_filename)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img_zip, target_zip = self._load_zips()

        with img_zip.open(self.imgs[index]) as img:
            img = tv_tensors.Image(Image.open(img).convert("RGB"))

        with target_zip.open(self.targets[index]) as target:
            target = tv_tensors.Mask(Image.open(target))

        if img.shape[-2:] != target.shape[-2:]:
            target = F.resize(
                target, list(img.shape[-2:]), interpolation=F.InterpolationMode.NEAREST
            )

        if target.shape[0] == 3:
            target = target.long()
            target = target[0, :, :] + target[1, :, :] * 256 + target[2, :, :] * 256**2
            segments_dict = self.annotations_dict[Path(self.targets[index]).name]
        else:
            target = target[0]
            segments_dict = {}

        masks, labels = [], []
        unique_labels = torch.unique(target)

        for label_id in unique_labels:
            class_id = label_id.item()

            if segments_dict:
                if class_id not in segments_dict:
                    continue
                class_id = segments_dict[class_id]

            if self.class_mapping is not None:
                if class_id not in self.class_mapping:
                    continue
                class_id = self.class_mapping[class_id]

            if class_id != self.ignore_idx:
                masks.append(target == label_id)
                labels.append(torch.tensor([class_id]))

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.cat(labels),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _load_zips(self) -> Tuple[zipfile.ZipFile, zipfile.ZipFile]:
        """Load zip files, handling multiprocessing."""
        worker = get_worker_info()
        worker = worker.id if worker else None

        if self.zip is None or (worker is not None and worker != getattr(self.zip, "_worker_id", None)):
            self.zip = zipfile.ZipFile(self.zip_path, "r")
            if worker is not None:
                self.zip._worker_id = worker

        if self.target_zip is None or (worker is not None and worker != getattr(self.target_zip, "_worker_id", None)):
            target_zip_path = self.target_zip_path or self.zip_path
            self.target_zip = zipfile.ZipFile(target_zip_path, "r")
            if worker is not None:
                self.target_zip._worker_id = worker

        return self.zip, self.target_zip

    def _sort_key(self, info):
        """Sort key for zip file members."""
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split(r'(\d+)', text)]

        return natural_keys(info.filename)

    def valid_member(self, member, folder_path_in_zip, stem_suffix, suffix):
        """Check if zip member is a valid image file."""
        if member.is_dir():
            return False

        member_path = Path(member.filename)
        
        try:
            member_path.relative_to(folder_path_in_zip)
        except ValueError:
            return False

        if not member_path.name.endswith(suffix):
            return False

        if stem_suffix and not member_path.stem.endswith(stem_suffix):
            return False

        return True
