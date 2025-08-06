"""
PointStream: A Content-Aware Neural Video Codec

This package provides a semantic video codec that separates static backgrounds
from dynamic foreground objects, achieving lower bitrates through content-aware
compression and generative AI reconstruction.
"""

__version__ = "0.1.0"
__author__ = "PointStream Team"
__email__ = "contact@pointstream.dev"

# Core imports for convenience
from .config import *
from .utils.logging_utils import setup_logging, get_logger

__all__ = [
    "__version__",
    "setup_logging", 
    "get_logger",
]