#!/usr/bin/env python3
"""
Simplified decorators for logging and timing pipeline components.

This module provides unified decorators for the PointStream pipeline components:
- @track_performance: Combined logging and timing decorator
- PerformanceProfiler: Simplified performance tracking
"""

import logging
import time
import functools
from typing import Callable, Any, Dict
from collections import defaultdict


class PerformanceProfiler:
    """Simplified performance profiler for timing analysis."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        
    def record_timing(self, operation: str, duration: float):
        """Record timing for an operation."""
        self.timings[operation].append(duration)
        
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        summary = {}
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    'avg_time': sum(times) / len(times),
                    'total_time': sum(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'call_count': len(times)
                }
        return summary
    
    def log_scene_summary(self, scene_number: int):
        """Log a performance summary for the completed scene."""
        summary = self.get_overall_summary()
        if not summary:
            logging.info(f"🔄 Scene {scene_number}: No performance data available")
            return
        
        total_time = sum(data['total_time'] for data in summary.values())
        logging.info(f"📊 Performance Summary for Scene {scene_number} - Total: {total_time:.2f}s")
        
        # Sort operations by total time (longest first)
        sorted_ops = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for operation, data in sorted_ops[:3]:  # Show top 3 slowest operations
            percentage = (data['total_time'] / total_time * 100) if total_time > 0 else 0
            logging.info(f"   {operation:20s}: {data['avg_time']:6.2f}s avg ({percentage:5.1f}%) - {data['call_count']} calls")


# Global profiler instance
profiler = PerformanceProfiler()


def track_performance(func: Callable) -> Callable:
    """
    Unified decorator for logging and timing pipeline steps.
    
    Combines the functionality of @log_step and @time_step into a single decorator.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        operation_name = f"{func_name}_processing"
        
        # Extract scene information from arguments
        scene_info = ""
        if args and len(args) > 1:
            if hasattr(args[1], '__len__') and hasattr(args[1], '__getitem__'):
                scene_info = f" ({len(args[1])} items)"
        
        # Only log start for major operations (filter out small, repetitive operations)
        verbose_logging = kwargs.get('verbose_logging', False)
        is_minor_operation = func_name in ['classify_class_name', 'filter_duplicates']
        
        if not is_minor_operation or verbose_logging:
            logging.info(f"🚀 Starting {func_name}{scene_info}")
            
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record timing
            profiler.record_timing(operation_name, execution_time)
            
            # Log completion with timing only for major operations
            if not is_minor_operation or verbose_logging:
                logging.info(f"✅ {func_name} completed in {execution_time:.2f}s")
            
            # Add timing to result if it's a dictionary
            if isinstance(result, dict):
                result['processing_time'] = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"❌ {func_name} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    return wrapper


# Legacy decorators for backward compatibility
def log_step(func: Callable) -> Callable:
    """Legacy decorator - use @track_performance instead."""
    return track_performance(func)


def time_step(track_processing: bool = True, track_debugging: bool = False):
    """Legacy decorator - use @track_performance instead."""
    def decorator(func: Callable) -> Callable:
        return track_performance(func)
    return decorator