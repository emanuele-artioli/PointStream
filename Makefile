SHELL := /bin/bash

.PHONY: help install install-dev clean test lint format check-format type-check pre-commit setup-pre-commit

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check-system: ## Check system compatibility and installation
	python check_system.py

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev,mmlab]"

install-mmlab: ## Install MMlab dependencies (run after conda activation)
	mim install mmcv==2.1.0 mmdet==3.2.0 mmpose==1.3.2 mmpretrain
	pip uninstall mmcv -y
	FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" pip install mmcv==2.1.0 --no-cache-dir

clean: ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test: ## Run the test suite
	pytest tests/ -v

lint: ## Run linting checks
	flake8 pointstream/ tests/
	mypy pointstream/

format: ## Format code with black and isort
	black pointstream/ tests/
	isort pointstream/ tests/

check-format: ## Check if code is properly formatted
	black --check pointstream/ tests/
	isort --check-only pointstream/ tests/

type-check: ## Run type checking
	mypy pointstream/

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

setup-pre-commit: ## Install pre-commit hooks
	pre-commit install

# Development workflow targets
dev-setup: install-dev setup-pre-commit ## Complete development setup

ci-check: check-format lint type-check test ## Run all CI checks

# Environment management
conda-env: ## Create the main conda environment
	conda env create -f environment.yml

conda-env-training: ## Create the training conda environment  
	conda env create -f environment-training.yml

# Quick commands for common operations
server: ## Run the server pipeline (requires video path: make server VIDEO=/path/to/video.mp4)
	python -m pointstream.scripts.run_server --input-video $(VIDEO)

client: ## Run the client reconstruction (requires JSON path: make client JSON=/path/to/results.json)
	python -m pointstream.scripts.run_client --input-json $(JSON)

train: ## Run training script (requires data path: make train DATA=/path/to/data)
	python -m pointstream.scripts.train --data_path $(DATA)
