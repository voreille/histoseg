
from pathlib import Path

from histoseg.data.dm_pannuke import PanNukeDataModule

project_dir = Path(__file__).parents[1].resolve()


def test_panuke_datamodule():
    """Test the PanNuke data module setup and basic functionality."""

    # Initialize data module
    dm = PanNukeDataModule(
        # devices="auto",
        num_workers=0,  # Set to 0 for testing purposes
        batch_size=2,
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
        print(
            "      http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        )
        print("   2. Place it in ./data/ade20k/ directory")
        return

    # Test dataloaders
    # try:
    train_loader = dm.train_dataloader()
    dm.val_dataloader()  # Just test creation

    print("✅ DataLoaders created successfully")

    # Test loading a batch
    batch = next(iter(train_loader))
    
    print(f"🖼️  Batch pixel_values shape: {batch['pixel_values'].shape}")
    print(f"🎯 Number of samples in batch: {len(batch['mask_labels'])}")
    print(f"📋 First sample mask_labels shape: {batch['mask_labels'][0].shape}")
    print(f"�️  First sample class_labels: {batch['class_labels'][0]}")
    print(f"� First sample number of masks: {len(batch['mask_labels'][0])}")

    # except Exception as e:
    #     print(f"❌ Error loading data: {e}")
    #     return

    print("🎉 ADE20K DataModule test completed successfully!")


if __name__ == "__main__":
    test_panuke_datamodule()
