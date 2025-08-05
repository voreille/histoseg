"""
Dataset class mappings for different segmentation datasets.
Adapted from benchmark-vfm-ss repository.
"""

import torchvision


def get_ade20k_mapping():
    """
    Get ADE20K class mapping.
    Maps from 1-150 to 0-149 (standard zero-indexed format).
    """
    return {i: i - 1 for i in range(1, 151)}


def get_cityscapes_mapping():
    """Get Cityscapes class mapping from class ID to train ID."""
    return {
        class_.id: class_.train_id
        for class_ in torchvision.datasets.Cityscapes.classes
    }


def get_mapillary_mapping():
    """Get Mapillary Vistas class mapping for domain adaptation."""
    return {
        13: 0,
        24: 0,
        41: 0,
        2: 1,
        15: 1,
        17: 2,
        6: 3,
        3: 4,
        45: 5,
        47: 5,
        48: 6,
        50: 7,
        30: 8,
        29: 9,
        27: 10,
        19: 11,
        20: 12,
        21: 12,
        22: 12,
        55: 13,
        61: 14,
        54: 15,
        58: 16,
        57: 17,
        52: 18,
    }
