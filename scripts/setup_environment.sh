#!/bin/bash

# This script fully automates the setup of the PointStream project.
# It will remove any existing 'pointstream' conda environment and create
# a fresh one from the 'environment.yml' file.
#
# Usage:
# 1. Make the script executable: chmod +x scripts/setup_environment.sh
# 2. Run from your project's root directory: ./scripts/setup_environment.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
ENV_NAME="pointstream"

# --- Step 1: Deactivate and Remove Old Environment ---
echo ">>> Checking for and removing existing '$ENV_NAME' environment..."
# Deactivate if currently active (best effort, may not work in all shells)
if [[ "$(conda info --envs | grep '*' | awk '{print $1}')" == "$ENV_NAME" ]]; then
  conda deactivate
fi
# Remove the environment to ensure a clean slate
conda env remove -n $ENV_NAME --yes || true

# --- Step 2: Create New Conda Environment ---
echo ">>> Creating new conda environment '$ENV_NAME' from environment.yml..."
conda env create -f environment.yml

# --- Step 3: Install PointStream Project ---
echo ">>> Installing PointStream project in editable mode..."
conda run -n $ENV_NAME pip install -e .

# --- Final Verification ---
echo ">>> Verifying environment..."
conda run -n $ENV_NAME python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
conda run -n $ENV_NAME python -c "import cv2; print('OpenCV version:', cv2.__version__)"


echo ""
echo "--- Setup Complete! ---"
echo "To activate the new environment, run: conda activate $ENV_NAME"
