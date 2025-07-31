"""
Centralized configuration for the Pointstream project.
"""
from pathlib import Path
import torch

# --- General Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = Path(__file__).parent

OUTPUT_DIR = PROJECT_ROOT / "output_scenes"
MODEL_DIR = PACKAGE_ROOT / "models" / "weights"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Stage 1: Scene Analysis ---
SCENE_DETECTOR_THRESHOLD = 27.0
MOTION_CLASSIFIER_THRESHOLD = 0.5
MOTION_DOWNSAMPLE_FACTOR = 0.25

# --- Stage 2: Object Detection ---
STUDENT_MODEL_PATH = "yolov8n.pt"
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# --- Stage 4: Foreground Representation ---
# Use the model's config name alias for simplicity and robustness.
MMPOSE_MODEL_ALIAS = 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192'

# --- Device Configuration ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- Logging ---
LOG_LEVEL = "INFO"