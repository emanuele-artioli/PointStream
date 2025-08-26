"""
PointStream Scripts Package

This package contains all the core PointStream processing components:

- server: Main processing pipeline 
- splitter: Video scene detection and splitting
- segmenter: Object detection and segmentation
- duplicate_filter: Duplicate detection filtering 
- semantic_classifier: Semantic classification of objects into human/animal/other
- stitcher: Panorama stitching
- keypointer: Keypoint detection
- saver: File saving utilities
- config: Configuration management
- decorators: Performance and logging decorators
"""

__version__ = "1.0.0"

# Import main components for easy access
# Note: PointStreamPipeline is in the parent server.py, not in scripts
from .splitter import VideoSceneSplitter
from .segmenter import Segmenter
from .duplicate_filter import DuplicateFilter
from .semantic_classifier import SemanticClassifier
from .stitcher import Stitcher
from .keypointer import Keypointer
from .saver import Saver
from utils import config

__all__ = [
    'VideoSceneSplitter',
    'Segmenter',
    'DuplicateFilter',
    'SemanticClassifier',
    'Stitcher', 
    'Keypointer',
    'Saver',
    'config'
]
