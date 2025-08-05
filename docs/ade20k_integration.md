# ADE20K Dataset Integration

This document explains how to use the ADE20K dataset with histoseg, following the approach from the benchmark-vfm-ss repository.

## Overview

The ADE20K data module has been adapted from the benchmark-vfm-ss repository to provide:

- **Zip-based loading**: Efficient loading directly from ADEChallengeData2016.zip
- **Modular transforms**: Separate training and validation augmentation pipelines  
- **Lightning integration**: Full PyTorch Lightning DataModule compatibility
- **Class mapping**: Proper ADE20K class mapping (1-150 → 0-149)
- **Multi-scale augmentation**: Scale jitter, color augmentation, spatial transforms

## Quick Start

### 1. Download ADE20K Dataset

```bash
# Download the official ADE20K dataset
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

# Create data directory
mkdir -p data/ade20k

# Move the zip file (don't extract it!)
mv ADEChallengeData2016.zip data/ade20k/
```

### 2. Basic Usage

```python
from histoseg.data import ADE20KDataModule

# Initialize data module
dm = ADE20KDataModule(
    root="./data/ade20k",           # Path to directory containing the zip
    devices="auto",                 # Lightning device specification  
    num_workers=4,                  # Number of dataloader workers
    img_size=(512, 512),           # Target image size (H, W)
    batch_size=8,                  # Training batch size
    num_classes=150,               # ADE20K has 150 classes
    scale_range=(0.5, 2.0),        # Scale augmentation range
    ignore_idx=255,                # Ignore index for loss computation
)

# Setup datasets
dm.setup("fit")

# Get dataloaders
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

# Use in training loop
for batch in train_loader:
    images, targets = batch
    # images: [B, 3, H, W] tensor
    # targets: List of dicts with 'masks' and 'labels' keys
```

### 3. Configuration Examples

See `examples/ade20k_configs.py` for predefined configurations:

```python
from examples.ade20k_configs import get_config
from histoseg.data import ADE20KDataModule

# Use prototype config for quick testing
config = get_config("prototype") 
dm = ADE20KDataModule(**config)

# Or training config for full training
config = get_config("training")
dm = ADE20KDataModule(**config)
```

## Key Features

### Zip-Based Loading
- Loads directly from `ADEChallengeData2016.zip` without extraction
- Efficient memory usage and faster I/O
- Handles multiprocessing correctly

### Smart Transforms
- **Training**: Scale jitter, random crop, horizontal flip, color augmentation
- **Validation**: Simple resize with normalization
- **Configurable**: Adjust augmentation strength via `scale_range`

### Proper Data Format
- **Images**: RGB tensors normalized with ImageNet statistics
- **Targets**: Dictionary format with masks and labels for Mask2Former compatibility
- **Class mapping**: ADE20K classes (1-150) mapped to (0-149)

## Advanced Usage

### Custom Transforms

```python
from histoseg.data import SegmentationTransforms

# Create custom transforms
custom_transforms = SegmentationTransforms(
    img_size=(768, 768),
    scale_range=(0.8, 1.2),
    max_brightness_delta=16,  # Reduce color augmentation
    training=True
)

dm = ADE20KDataModule(
    root="./data/ade20k",
    # ... other params
)

# Replace default transforms
dm.train_transforms = custom_transforms
```

### High Resolution Training

```python
# High-res config with smaller batches
dm = ADE20KDataModule(
    root="./data/ade20k",
    devices="auto",
    img_size=(768, 768),     # Higher resolution
    batch_size=4,            # Smaller batch size
    num_workers=6,
    scale_range=(0.5, 2.0),
)
```

### Debugging Setup

```python
# Debug config for fast iteration
dm = ADE20KDataModule(
    root="./data/ade20k", 
    devices="auto",
    img_size=(256, 256),     # Small images
    batch_size=1,            # Single sample
    num_workers=0,           # No multiprocessing
    scale_range=(1.0, 1.0),  # No augmentation
)
```

## Data Format

### Input Images
- **Format**: RGB JPEG images
- **Size**: Variable (resized to `img_size` during loading)
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Target Masks
- **Format**: Grayscale PNG with class indices
- **Classes**: 1-150 (mapped to 0-149)
- **Background**: 0 (mapped to ignore_idx=255)
- **Output**: Dictionary with 'masks' and 'labels' tensors

### Batch Structure
```python
images, targets = batch

# images: torch.Tensor of shape [B, 3, H, W]
# targets: List[Dict] with keys:
#   - 'masks': torch.Tensor of shape [N, H, W] (N = number of instances)  
#   - 'labels': torch.Tensor of shape [N] (class indices 0-149)
```

## Comparison with Reference

This implementation maintains compatibility with the benchmark-vfm-ss approach while adapting for histoseg:

| Feature | Reference Repo | Histoseg Adaptation |
|---------|---------------|-------------------|
| Base Architecture | ✅ Modular design | ✅ Same modular approach |
| Zip Loading | ✅ Direct zip access | ✅ Same zip loading |
| Transforms | ✅ v2 transforms | ✅ Same transform system |
| Lightning Integration | ✅ DataModule | ✅ Lightning 2.x compatible |
| Multi-dataset Support | ✅ Multiple datasets | 🔧 ADE20K focused |
| Class Mapping | ✅ Configurable | ✅ ADE20K specific |

## Testing

Run the test script to verify your setup:

```bash
python examples/test_ade20k_datamodule.py
```

This will test:
- Data module initialization
- Dataset setup and loading
- Dataloader creation
- Batch loading and format verification

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure `ADEChallengeData2016.zip` is in the correct directory
2. **Memory issues**: Reduce `batch_size` or `img_size` for limited memory
3. **Slow loading**: Increase `num_workers` (but not beyond CPU cores)
4. **Transform errors**: Check that `torchvision >= 0.15` for v2 transforms

### Performance Tips

- **Storage**: Keep zip file on fast storage (SSD) for better I/O
- **Workers**: Set `num_workers = min(8, cpu_cores)` for optimal performance  
- **Memory**: Use `pin_memory=True` (default) when training on GPU
- **Batch size**: Larger batches are more efficient but require more memory

## Next Steps

1. **Integration**: Use with your histoseg models for semantic segmentation
2. **Experimentation**: Try different `scale_range` values for augmentation
3. **Validation**: Monitor performance on ADE20K validation set
4. **Extension**: Adapt the pattern for other datasets (Cityscapes, etc.)

The data module is now ready for prototyping and training your histoseg models on ADE20K!
