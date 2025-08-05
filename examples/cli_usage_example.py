"""
Usage example for the new HistoSeg CLI.

This demonstrates how to use the adapted CLI from the benchmark-vfm-ss repository.
"""

import sys
from pathlib import Path


def run_training_example():
    """Example of how to run training with the new CLI."""
    
    print("🚀 HistoSeg Training with Reference Repository CLI")
    print("=" * 60)
    
    # Check if data exists
    data_path = Path("./data/ade20k/ADEChallengeData2016.zip")
    if not data_path.exists():
        print("❌ ADE20K dataset not found!")
        print("📥 Please download the dataset:")
        print("   wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip")
        print("   mkdir -p data/ade20k")
        print("   mv ADEChallengeData2016.zip data/ade20k/")
        return False
    
    print("✅ Dataset found")
    
    # Example training commands
    training_commands = [
        # Basic training
        [
            sys.executable, "cli.py", "fit",
            "--config", "configs/ade20k_reference.yaml",
            "--root", "./data/ade20k"
        ],
        
        # Training with custom parameters
        [
            sys.executable, "cli.py", "fit",
            "--config", "configs/ade20k_reference.yaml", 
            "--root", "./data/ade20k",
            "--trainer.max_epochs", "50",
            "--data.init_args.batch_size", "4",
            "--model.init_args.lr", "0.0002"
        ],
        
        # Training without torch.compile (for debugging)
        [
            sys.executable, "cli.py", "fit",
            "--config", "configs/ade20k_reference.yaml",
            "--root", "./data/ade20k", 
            "--no_compile"
        ]
    ]
    
    print("\n📋 Available training commands:")
    for i, cmd in enumerate(training_commands, 1):
        print(f"\n{i}. {' '.join(cmd)}")
    
    print("\n🔧 Key Features from Reference Repository:")
    features = [
        "✅ Automatic argument linking (data -> model)",
        "✅ torch.compile optimization support", 
        "✅ Global step-based validation",
        "✅ Weights & Biases code logging",
        "✅ Mixed precision training",
        "✅ Configurable learning rates for encoder/decoder",
        "✅ IoU metrics and visualization",
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    return True


def run_validation_example():
    """Example of how to run validation."""
    
    print("\n🔍 Validation Commands:")
    validation_commands = [
        [
            sys.executable, "cli.py", "validate",
            "--config", "configs/ade20k_reference.yaml",
            "--ckpt_path", "path/to/checkpoint.ckpt"
        ],
        [
            sys.executable, "cli.py", "test", 
            "--config", "configs/ade20k_reference.yaml",
            "--ckpt_path", "path/to/checkpoint.ckpt"
        ]
    ]
    
    for i, cmd in enumerate(validation_commands, 1):
        print(f"{i}. {' '.join(cmd)}")


def compare_old_vs_new():
    """Compare old vs new CLI approach."""
    
    print("\n📊 Comparison: Old vs New CLI")
    print("=" * 60)
    
    comparisons = [
        ("🏗️ Architecture", "Custom CLI", "Lightning CLI + Reference patterns"),
        ("🔗 Argument linking", "Manual", "Automatic linking"),
        ("⚡ Optimization", "Basic", "torch.compile + mixed precision"),
        ("📊 Metrics", "Manual setup", "Automatic IoU tracking"),
        ("🎯 Validation", "Epoch-based", "Global step-based"),
        ("📝 Logging", "Basic", "W&B + code logging"),
        ("🔧 Configuration", "Hydra", "Lightning CLI YAML"),
        ("🎨 Visualization", "None", "Automatic plots"),
    ]
    
    print(f"{'Aspect':<20} {'Old Approach':<20} {'New Approach'}")
    print("-" * 60)
    for aspect, old, new in comparisons:
        print(f"{aspect:<20} {old:<20} {new}")


if __name__ == "__main__":
    success = run_training_example()
    
    if success:
        run_validation_example()
        compare_old_vs_new()
        
        print("\n🎉 HistoSeg is now ready with the reference repository CLI!")
        print("🚀 Start training with: python cli.py fit --config configs/ade20k_reference.yaml --root ./data/ade20k")
    else:
        print("\n❌ Please set up the dataset first before training.")
