#!/usr/bin/env python3
"""
Decorators for logging and timing pipeline components.

This module provides decorators for the PointStream pipeline components:
- @log_step: Logs function execution with crucial information
- @time_step: Measures execution time, distinguishing between processing and debugging
- PerformanceProfiler: Detailed performance tracking and reporting
"""

import logging
import time
import functools
from typing import Callable, Any, Dict, List
from collections import defaultdict, deque


class PerformanceProfiler:
    """Global performance profiler for detailed timing analysis."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.global_timings = defaultdict(list)  # Keep global accumulation
        self.current_scene_timings = {}
        self.scene_count = 0
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.current_scene_timings[operation] = time.time()
        
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation in self.current_scene_timings:
            duration = time.time() - self.current_scene_timings[operation]
            self.timings[operation].append(duration)
            self.global_timings[operation].append(duration)  # Also store globally
            del self.current_scene_timings[operation]
            return duration
        return 0.0
        
    def log_scene_summary(self, scene_number: int):
        """Log a comprehensive performance summary for the completed scene."""
        summary = self.get_overall_summary()
        if not summary:
            logging.info(f"🔄 Scene {scene_number}: No performance data available")
            return
        
        total_time = sum(data['total_time'] for data in summary.values())
        
        logging.info(f"📊 Performance Summary for Scene {scene_number}")
        logging.info(f"   Total Processing Time: {total_time:.2f}s")
        
        # Sort operations by total time (longest first)
        sorted_ops = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for operation, data in sorted_ops:
            percentage = (data['total_time'] / total_time * 100) if total_time > 0 else 0
            logging.info(f"   {operation:20s}: {data['total_time']:6.2f}s ({percentage:5.1f}%) - {data['call_count']} calls")
        
        # Find bottlenecks
        if sorted_ops:
            slowest = sorted_ops[0]
            if slowest[1]['total_time'] > total_time * 0.5:
                logging.warning(f"⚠️ Performance bottleneck detected: {slowest[0]} takes {slowest[1]['total_time']:.1f}s ({slowest[1]['total_time']/total_time*100:.1f}% of total time)")
        
        # Clear data for next scene
        self.clear()
    
    def clear(self):
        """Clear all timing data."""
        self.timings.clear()


# Global profiler instance
        
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        summary = {}
        for operation, times in self.global_timings.items():  # Use global timings
            if times:
                summary[operation] = {
                    'avg_time': sum(times) / len(times),
                    'total_time': sum(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'call_count': len(times)
                }
        return summary

# Global profiler instance
profiler = PerformanceProfiler()


def log_step(func: Callable) -> Callable:
    """
    Decorator to log the execution of pipeline steps.
    
    Logs the function name, scene information, and any errors or warnings.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        # Try to extract scene information from arguments
        scene_info = ""
        if args:
            if hasattr(args[0], '__class__'):
                # Method call - args[0] is self
                if len(args) > 1 and isinstance(args[1], (list, tuple)):
                    scene_info = f" (processing {len(args[1])} frames)"
            elif isinstance(args[0], (list, tuple)):
                # Function call with frames as first argument
                scene_info = f" (processing {len(args[0])} frames)"
        
        logging.info(f"Starting {func_name}{scene_info}")
        
        try:
            result = func(*args, **kwargs)
            
            # Log additional information based on result
            if isinstance(result, dict):
                if 'scene_type' in result:
                    logging.info(f"{func_name} completed - Scene type: {result['scene_type']}")
                elif 'keypoints' in result:
                    logging.info(f"{func_name} completed - Found keypoints for {len(result.get('objects', []))} objects")
                elif 'prompt' in result:
                    logging.info(f"{func_name} completed - Generated prompt: '{result['prompt'][:100]}...'")
                elif 'inpainted_background' in result:
                    logging.info(f"{func_name} completed - Inpainted background generated")
                else:
                    logging.info(f"{func_name} completed successfully")
            else:
                logging.info(f"{func_name} completed successfully")
            
            return result
            
        except Exception as e:
            logging.error(f"{func_name} failed: {str(e)}")
            raise
    
    return wrapper


def time_step(track_processing: bool = True, track_debugging: bool = False):
    """
    Decorator to measure execution time of pipeline steps.
    
    Args:
        track_processing: Whether to track this as processing time (default: True)
        track_debugging: Whether to track this as debugging time (default: False)
    
    The decorator distinguishes between:
    - Processing time: Active model inference, image processing, etc.
    - Debugging time: File saving, visualization, optional operations
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            operation_name = f"{func_name}_{'processing' if track_processing else 'debugging'}"
            
            # Start profiler timer
            profiler.start_timer(operation_name)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                # End profiler timer
                profiler.end_timer(operation_name)
                
                # Log detailed timing information
                if track_processing:
                    logging.info(f"🚀 {func_name} processing: {execution_time:.3f}s")
                elif track_debugging:
                    logging.debug(f"🔧 {func_name} debugging: {execution_time:.3f}s")
                else:
                    logging.info(f"⏱️  {func_name} execution: {execution_time:.3f}s")
                
                # Add timing information to result if it's a dictionary
                if isinstance(result, dict):
                    timing_key = 'processing_time' if track_processing else 'debugging_time'
                    result[timing_key] = execution_time
                    result['operation_name'] = operation_name
                
                return result
                
            except Exception as e:
                # End timer even on error
                profiler.end_timer(operation_name)
                raise
        
        return wrapper
    return decorator


def gpu_context(gpu_id: int):
    """
    Decorator to set GPU context for model operations.
    
    Args:
        gpu_id: GPU device ID to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import os
            original_gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            try:
                # Set GPU context
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore original GPU context
                if original_gpu is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_gpu
                else:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        
        return wrapper
    return decorator