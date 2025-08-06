"""
HistoSeg CLI module.

Provides command-line interface for training, validation, and testing.
Adapted from benchmark-vfm-ss repository.

Usage:
    python cli.py fit --config configs/ade20k.yaml
    python cli.py validate --ckpt_path path/to/checkpoint.ckpt
    python cli.py test --ckpt_path path/to/checkpoint.ckpt
"""

import logging
from types import MethodType

import torch
from gitignore_parser import parse_gitignore
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop

from histoseg.data import BaseDataModule
from histoseg.training import HistoSegModule


def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    """Custom validation check function for global step-based validation."""
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # Check validation based on global steps instead of batches
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class HistoSegCLI(cli.LightningCLI):
    """
    HistoSeg Lightning CLI with custom configuration and linking.
    
    Provides automated argument linking between trainer, model, and data modules.
    Supports torch.compile and Weights & Biases logging.
    """

    def __init__(self, *args, **kwargs):
        # Setup logging and torch optimizations
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.suppress_errors = True  # type: ignore

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        """Add custom arguments and create argument links."""
        # Data root directory
        parser.add_argument("--root", type=str, help="Root directory for dataset")
        parser.link_arguments("root", "data.init_args.root")
        parser.link_arguments("root", "trainer.logger.init_args.save_dir")

        # Compilation flag
        parser.add_argument("--no_compile", action="store_true", 
                          help="Disable torch.compile optimization")

        # Link trainer devices to data module
        parser.link_arguments("trainer.devices", "data.init_args.devices")

        # Link data module config to model
        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )
        parser.link_arguments(
            "data.init_args.num_metrics", "model.init_args.num_metrics"
        )
        parser.link_arguments("data.init_args.ignore_idx", "model.init_args.ignore_idx")
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )

    def fit(self, model, **kwargs):
        """Enhanced fit method with code logging and torch.compile."""
        # Log code to Weights & Biases if available
        if hasattr(self.trainer.logger.experiment, "log_code"):  # type: ignore
            is_gitignored = parse_gitignore(".gitignore")
            
            def include_fn(path):
                return path.endswith(".py") or path.endswith(".yaml")
            
            self.trainer.logger.experiment.log_code(  # type: ignore
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        # Install custom validation check function
        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        # Apply torch.compile if not disabled
        if not self.config[self.config["subcommand"]]["no_compile"]:  # type: ignore
            print("🚀 Applying torch.compile for optimized training...")
            model = torch.compile(model)

        # Start training
        self.trainer.fit(model, **kwargs)  # type: ignore


def cli_main():
    """Main CLI entry point."""
    HistoSegCLI(
        HistoSegModule,
        BaseDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=42,  # Reproducible training
        trainer_defaults={
            "precision": "16-mixed",      # Mixed precision training
            "log_every_n_steps": 1,       # Frequent logging
            "enable_model_summary": False,
            "callbacks": [ModelSummary(max_depth=2)],
            "devices": 1,                 # Single GPU by default
            "accumulate_grad_batches": 16, # Effective batch size scaling
            "max_epochs": 100,            # Default max epochs
            "val_check_interval": 1000,   # Validate every 1000 steps
        },
    )


if __name__ == "__main__":
    cli_main()
