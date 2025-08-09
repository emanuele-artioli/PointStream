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
MAX_SCENE_SIZE = 100  # Maximum frames per scene to prevent memory issues

# --- Stage 2: Object Detection ---
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# FIX: Create a model registry to handle multiple content-specific models
MODEL_REGISTRY = {
    "general": "yolo11n.pt",  # The pre-trained general model
    "sports": str(ARTIFACTS_DIR / "training" / "sports_model.pt"),  # Path to our trained sports model
    "dance": str(ARTIFACTS_DIR / "training" / "dance_model.pt"),   # Path to dance model
    "automotive": str(ARTIFACTS_DIR / "training" / "automotive_model.pt"),  # Path to automotive model
}

# Content type definitions with specialized ontologies
CONTENT_TYPE_ONTOLOGIES = {
    "sports": {
        "classes": ["tennis_player", "tennis_racket", "skier", "basketball_player", "soccer_player"],
        "description": "Sports-related objects and athletes"
    },
    "dance": {
        "classes": ["dancer", "ballet_dancer", "contemporary_dancer"],
        "description": "Dance and performance-related content"
    },
    "automotive": {
        "classes": ["car", "truck", "motorcycle", "bus"],
        "description": "Vehicles and automotive content"
    },
    "general": {
        "classes": ["person", "vehicle", "animal"],
        "description": "General-purpose object detection"
    }
}

# Annotations cache directory for reusing SAM annotations
ANNOTATIONS_CACHE_DIR = ARTIFACTS_DIR / "annotations_cache"
ANNOTATIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache configuration
CACHE_CONFIG = {
    "max_size_gb": 10.0,  # Maximum cache size in GB
    "cleanup_older_than_days": 30,  # Remove annotations older than this
    "auto_cleanup": True,  # Automatically clean up old annotations
}

# --- Stage 4: Foreground Representation ---
MMPOSE_HUMAN_MODEL_ALIAS = 'human'
MMPOSE_ANIMAL_MODEL_ALIAS = 'animal'

# --- Device Configuration ---
# Allow forcing CPU usage with environment variable
FORCE_CPU = os.getenv('POINTSTREAM_FORCE_CPU', 'false').lower() == 'true'
DEVICE = 'cpu' if FORCE_CPU else ('cuda:0' if torch.cuda.is_available() else 'cpu')