"""
Centralized configuration for the Pointstream project.
"""
from pathlib import Path
import torch

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
MOTION_CLASSIFIER_THRESHOLD = 0.5
MOTION_DOWNSAMPLE_FACTOR = 0.25

# --- Stage 2: Object Detection ---
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# FIX: Create a model registry to handle multiple student models
MODEL_REGISTRY = {
    "general": "yolo12n.pt", # The pre-trained general model
    "sports": str(MODEL_DIR / "yolo_sports_student.pt") # Path to our future custom model
}

# --- Stage 4: Foreground Representation ---
MMPOSE_HUMAN_MODEL_ALIAS = 'human'
MMPOSE_ANIMAL_MODEL_ALIAS = 'animal'

# --- Device Configuration ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'