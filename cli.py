#!/usr/bin/env python3
"""
Main Lightning CLI for training and evaluation.

Usage:
    python cli.py fit --config configs/ade20k.yaml
    python cli.py validate --ckpt_path path/to/checkpoint.ckpt
    python cli.py test --ckpt_path path/to/checkpoint.ckpt
"""

from pytorch_lightning.cli import LightningCLI

from histoseg.training.mask2former_module import Mask2FormerModule
from histoseg.data.dm_ade20k import ADE20KDataModule


class HistosegCLI(LightningCLI):
    """Custom Lightning CLI for histoseg."""
    
    def add_arguments_to_parser(self, parser):
        """Add custom arguments to the parser."""
        parser.link_arguments("model.init_args.num_classes", "data.init_args.num_classes")
        parser.link_arguments("model.init_args.image_size", "data.init_args.image_size")


def main():
    """Main entry point."""
    cli = HistosegCLI(
        model_class=Mask2FormerModule,
        datamodule_class=ADE20KDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
