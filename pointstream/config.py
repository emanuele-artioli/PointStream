"""
Centralized configuration for the Pointstream project.
"""
from pathlib import Path
import torch
import os

# --- General Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "pipeline_output"
MODEL_DIR = PACKAGE_ROOT / "models" / "weights"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Stage 1: Scene Analysis ---
SCENE_DETECTOR_THRESHOLD = 27.0
MOTION_CLASSIFIER_THRESHOLD = 10.0  # Temporarily increased to allow more scenes through
MOTION_DOWNSAMPLE_FACTOR = 0.25

# --- Stage 2: Object Detection ---
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# FIX: Create a model registry to handle multiple student models
MODEL_REGISTRY = {
    "general": "yolo11n.pt", # The pre-trained general model
    "sports": str(ARTIFACTS_DIR / "training" / "sports_model.pt") # Path to our trained sports model
}

# --- Stage 4: Foreground Representation ---
MMPOSE_HUMAN_MODEL_ALIAS = 'human'
MMPOSE_ANIMAL_MODEL_ALIAS = 'animal'

# --- Device Configuration ---
# Allow forcing CPU usage with environment variable
FORCE_CPU = os.getenv('POINTSTREAM_FORCE_CPU', 'false').lower() == 'true'
DEVICE = 'cpu' if FORCE_CPU else ('cuda:0' if torch.cuda.is_available() else 'cpu')