#!/usr/bin/env python3
"""
Simplified decorators for the PointStream pipeline.
"""

import functools
import time
from typing import Callable, Any, Dict

# Global profiler instance (stub for compatibility)
class PerformanceProfiler:
    def get_overall_summary(self) -> Dict[str, Any]:
        return {}
        
    def log_scene_summary(self, scene_number: int):
        pass

profiler = PerformanceProfiler()


def track_performance(func: Callable) -> Callable:
    """
    A pass-through decorator that replaces the original performance tracking.
    It still adds 'processing_time' to dictionary results for compatibility.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.pop('verbose_logging', None)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if isinstance(result, dict):
            result['processing_time'] = execution_time
            
        return result
    
    return wrapper