"""
Data transforms for histoseg datasets.
Adapted from benchmark-vfm-ss repository.
"""

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torch import nn


class SegmentationTransforms(nn.Module):
    """
    Transforms for semantic segmentation datasets.
    Supports random augmentations during training.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        scale_range: tuple[float, float] = (0.5, 2.0),
        max_brightness_delta: float = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: float = 18,
        training: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.training = training

        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()

        self.scale_jitter = T.ScaleJitter(
            target_size=img_size,
            scale_range=scale_range,
            antialias=True,
        )

        self.random_crop = T.RandomCrop(img_size)

    def random_factor(self, factor, center=1.0):
        """Generate random factor for augmentations."""
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def brightness(self, img):
        """Apply random brightness adjustment."""
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, brightness_factor=self.random_factor(self.max_brightness_factor)
            )
        return img

    def contrast(self, img):
        """Apply random contrast adjustment."""
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(
                img, contrast_factor=self.random_factor(self.max_contrast_factor)
            )
        return img

    def saturation(self, img):
        """Apply random saturation adjustment."""
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, saturation_factor=self.random_factor(self.max_saturation_factor)
            )
        return img

    def hue(self, img):
        """Apply random hue adjustment."""
        if torch.rand(()) < 0.5:
            img = F.adjust_hue(
                img, hue_factor=self.random_factor(self.max_hue_delta, center=0.0)
            )
        return img

    def forward(self, img, target):
        """Apply transforms to image and target."""
        if self.training:
            # Geometric augmentations
            img, target = self.random_horizontal_flip(img, target)
            img, target = self.scale_jitter(img, target)
            img, target = self.random_crop(img, target)

            # Color augmentations (only on image)
            img = self.brightness(img)
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            # Only resize for validation/test
            img = F.resize(img, self.img_size, antialias=True)
            target["masks"] = F.resize(
                target["masks"], 
                self.img_size, 
                interpolation=F.InterpolationMode.NEAREST
            )

        # Convert to float and normalize
        img = F.to_dtype(img, torch.float32, scale=True)
        
        # ImageNet normalization
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, target
