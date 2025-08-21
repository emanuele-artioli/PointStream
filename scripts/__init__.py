"""
PointStream Scripts Package

This package contains all the core PointStream processing components with single-word names:

- server: Main processing pipeline (formerly server_pipeline.py)
- splitter: Video scene detection and splitting (formerly video_scene_splitter.py)
- segmenter: Object detection and segmentation
- stitcher: Panorama stitching
- keypointer: Keypoint detection
- saver: File saving utilities
- config: Configuration management
- decorators: Performance and logging decorators

Note: client.py will be added later for client-side functionality.
"""

__version__ = "1.0.0"

# Import main components for easy access
from .server import PointStreamPipeline, PointStreamProcessor
from .splitter import VideoSceneSplitter
from .segmenter import Segmenter
from .stitcher import Stitcher
from .keypointer import Keypointer
from .saver import Saver
from . import config

__all__ = [
    'PointStreamPipeline',
    'PointStreamProcessor', 
    'VideoSceneSplitter',
    'Segmenter',
    'Stitcher', 
    'Keypointer',
    'Saver',
    'config'
]
