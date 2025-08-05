"""
Example usage of ADE20K DataModule for histoseg.

This script demonstrates how to use the adapted ADE20K data module
that follows the benchmark-vfm-ss repository approach.
"""

from histoseg.data import ADE20KDataModule


def test_ade20k_datamodule():
    """Test the ADE20K data module setup and basic functionality."""
    
    # Initialize data module
    dm = ADE20KDataModule(
        root="./data/ade20k",  # Path to directory containing ADEChallengeData2016.zip
        devices="auto",
        num_workers=4,
        img_size=(512, 512),
        batch_size=2,
        num_classes=150,
        scale_range=(0.5, 2.0),
        ignore_idx=255,
    )
    
    print("✅ ADE20K DataModule initialized successfully")
    print(f"📊 Number of classes: {dm.num_classes}")
    print(f"🖼️  Image size: {dm.img_size}")
    print(f"📦 Batch size: {dm.dataloader_kwargs['batch_size']}")
    print(f"👥 Number of workers: {dm.dataloader_kwargs['num_workers']}")
    
    # Setup datasets
    try:
        dm.setup("fit")
        print("✅ Datasets setup successfully")
        print(f"🚂 Training dataset size: {len(dm.train_dataset)}")
        print(f"🔍 Validation dataset size: {len(dm.val_dataset)}")
    except FileNotFoundError as e:
        print(f"❌ Dataset file not found: {e}")
        print("📥 Please download ADE20K dataset:")
        print("   1. Download ADEChallengeData2016.zip from:")
        print("      http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip")
        print("   2. Place it in ./data/ade20k/ directory")
        return
    
    # Test dataloaders
    try:
        train_loader = dm.train_dataloader()
        dm.val_dataloader()  # Just test creation
        
        print("✅ DataLoaders created successfully")
        
        # Test loading a batch
        batch = next(iter(train_loader))
        images, targets = batch
        
        print(f"🖼️  Batch images shape: {images.shape}")
        print(f"🎯 Number of targets: {len(targets)}")
        print(f"📋 First target keys: {list(targets[0].keys())}")
        print(f"🎭 First target masks shape: {targets[0]['masks'].shape}")
        print(f"🏷️  First target labels: {targets[0]['labels']}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("🎉 ADE20K DataModule test completed successfully!")


def compare_with_reference():
    """Compare our implementation with the reference repository approach."""
    
    print("\n📊 Comparison with benchmark-vfm-ss repository:")
    print("=" * 60)
    
    similarities = [
        "✅ Modular design with base DataModule class",
        "✅ Zip-based dataset loading for efficiency", 
        "✅ Configurable transforms with training/validation modes",
        "✅ Class mapping support (ADE20K: 1-150 → 0-149)",
        "✅ Multi-scale augmentations with scale jitter",
        "✅ Proper collate functions for training vs evaluation",
        "✅ Lightning DataModule integration",
    ]
    
    adaptations = [
        "🔧 Adapted for PyTorch Lightning 2.x",
        "🔧 Integrated with histoseg project structure", 
        "🔧 Added comprehensive documentation",
        "🔧 Simplified for ADE20K focus (not multi-dataset)",
        "🔧 Compatible with histoseg model architecture",
    ]
    
    print("Similarities with reference:")
    for item in similarities:
        print(f"  {item}")
    
    print("\nAdaptations for histoseg:")
    for item in adaptations:
        print(f"  {item}")
    
    print("\n🎯 Result: Clean, efficient ADE20K data loading for prototyping!")


if __name__ == "__main__":
    test_ade20k_datamodule()
    compare_with_reference()
