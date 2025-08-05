"""
Configuration example for ADE20K dataset with histoseg.

This shows how to configure the data module for different use cases.
"""

# Basic configuration for prototyping
PROTOTYPE_CONFIG = {
    "root": "./data/ade20k",
    "devices": "auto", 
    "num_workers": 2,
    "img_size": (512, 512),
    "batch_size": 2,
    "num_classes": 150,
    "scale_range": (0.8, 1.2),  # Mild augmentation for prototyping
    "ignore_idx": 255,
}

# Full training configuration
TRAINING_CONFIG = {
    "root": "./data/ade20k",
    "devices": "auto",
    "num_workers": 8,
    "img_size": (512, 512), 
    "batch_size": 8,
    "num_classes": 150,
    "scale_range": (0.5, 2.0),  # Full augmentation range
    "ignore_idx": 255,
}

# High resolution configuration
HIGH_RES_CONFIG = {
    "root": "./data/ade20k",
    "devices": "auto",
    "num_workers": 6,
    "img_size": (768, 768),
    "batch_size": 4,  # Smaller batch for high res
    "num_classes": 150,
    "scale_range": (0.5, 2.0),
    "ignore_idx": 255,
}

# Fast debugging configuration
DEBUG_CONFIG = {
    "root": "./data/ade20k", 
    "devices": "auto",
    "num_workers": 0,  # No multiprocessing for debugging
    "img_size": (256, 256),  # Smaller images
    "batch_size": 1,
    "num_classes": 150,
    "scale_range": (1.0, 1.0),  # No scale augmentation
    "ignore_idx": 255,
}


def get_config(config_name: str = "prototype") -> dict:
    """Get configuration by name."""
    configs = {
        "prototype": PROTOTYPE_CONFIG,
        "training": TRAINING_CONFIG, 
        "high_res": HIGH_RES_CONFIG,
        "debug": DEBUG_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]


if __name__ == "__main__":
    from histoseg.data import ADE20KDataModule
    
    # Example usage
    config = get_config("prototype")
    dm = ADE20KDataModule(**config)
    
    print("📋 Configuration: prototype")
    print(f"🖼️  Image size: {config['img_size']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"👥 Workers: {config['num_workers']}")
    print(f"📏 Scale range: {config['scale_range']}")
