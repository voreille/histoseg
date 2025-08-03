.PHONY: help install install-dev clean test lint format compile-ops train validate

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	pip install pybind11  # Install pybind11 first
	pip install -r requirements.txt
	pip install -e . --config-settings editable_mode=compat

install-dev:  ## Install package with development dependencies
	pip install pybind11  # Install pybind11 first
	pip install -r requirements.txt
	pip install -e . --config-settings editable_mode=compat
	pip install pytest pytest-cov black isort flake8 mypy

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

compile-ops:  ## Compile CUDA operations
	@echo "Compiling CUDA operations..."
	./compile_ops.sh

test:  ## Run tests
	pytest tests/ -v --cov=histoseg --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 histoseg/ --max-line-length=100 --ignore=E203,W503
	mypy histoseg/ --ignore-missing-imports

format:  ## Format code
	black histoseg/ scripts/ cli.py --line-length=100
	isort histoseg/ scripts/ cli.py --profile=black

# Training shortcuts
train:  ## Train model with default config
	python cli.py fit --config configs/ade20k.yaml

train-debug:  ## Train model in debug mode
	python cli.py fit --config configs/ade20k.yaml --trainer.fast_dev_run=true

train-uni2:  ## Train with UNI2 backbone
	python cli.py fit --config configs/ade20k.yaml \
		--model.backbone.model_name "hf-hub:MahmoodLab/UNI2-h"

validate:  ## Validate trained model
	@echo "Please specify checkpoint path: make validate CKPT=path/to/checkpoint.ckpt"

validate-with-ckpt:  ## Validate with specific checkpoint
	python cli.py validate --config configs/ade20k.yaml --ckpt_path $(CKPT)

# Data preparation
prepare-ade20k:  ## Download and prepare ADE20K dataset
	python scripts/prepare_ade20k.py --data_dir data/ade20k

# Development
dev-setup:  ## Complete development setup
	make install-dev
	make compile-ops
	make prepare-ade20k
	@echo "Development setup complete!"

# Inference
inference:  ## Run inference (requires MODEL and IMAGE variables)
	python scripts/inference.py --model_path $(MODEL) --image_path $(IMAGE)

# Examples
example-train:  ## Show example training commands
	@echo "Example training commands:"
	@echo "  Basic training:     make train"
	@echo "  Debug training:     make train-debug"
	@echo "  UNI2 training:      make train-uni2"
	@echo "  Custom config:      python cli.py fit --config my_config.yaml"

example-inference:  ## Show example inference commands
	@echo "Example inference commands:"
	@echo "  Basic inference:    make inference MODEL=model.ckpt IMAGE=test.jpg"
	@echo "  With custom output: python scripts/inference.py --model_path model.ckpt --image_path test.jpg --output_path result.png"

# Docker (if needed)
docker-build:  ## Build Docker image
	docker build -t histoseg:latest .

docker-run:  ## Run Docker container
	docker run --gpus all -it --rm -v $(PWD):/workspace histoseg:latest bash
