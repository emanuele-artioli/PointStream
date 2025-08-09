"""
Warning suppression utilities for PointStream.

This module provides utilities to suppress known harmless warnings that 
create noise in the pipeline output without affecting functionality.
"""

import warnings
import logging
import functools
from typing import Callable, Any


def suppress_known_warnings():
    """
    Suppress known harmless warnings from dependencies.
    
    This includes:
    - MMEngine registry warnings (non-functional, uses fallback)
    - pkg_resources deprecation warnings (will be resolved by setuptools update)
    - PyTorch pytree deprecation warnings (resolved in newer transformers)
    """
    
    # Suppress MMEngine registry warnings
    warnings.filterwarnings(
        'ignore', 
        message='Failed to search registry with scope.*', 
        category=UserWarning,
        module='mmengine'
    )
    
    # Suppress pkg_resources deprecation warnings
    warnings.filterwarnings(
        'ignore',
        message='pkg_resources is deprecated.*',
        category=UserWarning
    )
    
    # Suppress PyTorch pytree warnings (if any remain)
    warnings.filterwarnings(
        'ignore',
        message='.*_pytree._register_pytree_node.*deprecated.*',
        category=FutureWarning
    )
    
    # Suppress other common OpenMMLab warnings
    warnings.filterwarnings(
        'ignore',
        message='.*The parameter.*pretrained.*deprecated.*',
        category=UserWarning
    )


def with_suppressed_warnings(func: Callable) -> Callable:
    """
    Decorator to suppress warnings for a specific function.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with warnings suppressed
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            suppress_known_warnings()
            return func(*args, **kwargs)
    return wrapper


class WarningFilter:
    """
    Context manager for temporarily suppressing warnings.
    
    Usage:
        with WarningFilter():
            # Code that might produce harmless warnings
            pass
    """
    
    def __init__(self, suppress_all: bool = False):
        self.suppress_all = suppress_all
        self.original_filters = None
    
    def __enter__(self):
        self.original_filters = warnings.filters[:]
        if self.suppress_all:
            warnings.simplefilter('ignore')
        else:
            suppress_known_warnings()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filters[:] = self.original_filters


def setup_clean_logging():
    """
    Setup logging to reduce noise from dependencies while keeping important messages.
    """
    # Reduce verbosity of specific loggers
    logging.getLogger('mmengine').setLevel(logging.ERROR)
    logging.getLogger('mmpose').setLevel(logging.WARNING)
    logging.getLogger('mmdet').setLevel(logging.WARNING)
    
    # Specifically suppress the registry warning by configuring mmengine logger
    mmengine_logger = logging.getLogger('mmengine')
    mmengine_logger.addFilter(lambda record: 'Failed to search registry' not in record.getMessage())
    
    # But keep our own messages
    logging.getLogger('pointstream').setLevel(logging.INFO)


# Apply global warning suppression when module is imported
suppress_known_warnings()
