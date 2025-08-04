"""
Centralized configuration for the Pointstream project.
"""
from pathlib import Path
import torch

# --- General Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = Path(__file__).parent

# FIX: Point to the new, centralized artifacts directory
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "pipeline_output" # For main pipeline results

MODEL_DIR = PACKAGE_ROOT / "models" / "weights"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Stage 1: Scene Analysis ---
SCENE_DETECTOR_THRESHOLD = 27.0
MOTION_CLASSIFIER_THRESHOLD = 0.5
MOTION_DOWNSAMPLE_FACTOR = 0.25

# --- Stage 2: Object Detection ---
STUDENT_MODEL_PATH = MODEL_DIR / "yolo12n.pt"
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# --- Stage 4: Foreground Representation ---
MMPOSE_HUMAN_MODEL_ALIAS = 'human'
MMPOSE_ANIMAL_MODEL_ALIAS = 'animal'

# --- Device Configuration ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'