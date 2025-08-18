#!/usr/bin/env python3
"""
Decorators for logging and timing pipeline components.

This module provides decorators for the PointStream pipeline components:
- @log_step: Logs function execution with crucial information
- @time_step: Measures execution time, distinguishing between processing and debugging
"""

import logging
import time
import functools
from typing import Callable, Any, Dict


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
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Log timing information
                if track_processing:
                    logging.info(f"{func_name} processing time: {execution_time:.3f}s")
                elif track_debugging:
                    logging.debug(f"{func_name} debugging time: {execution_time:.3f}s")
                else:
                    logging.info(f"{func_name} execution time: {execution_time:.3f}s")
                
                # Add timing information to result if it's a dictionary
                if isinstance(result, dict):
                    timing_key = 'processing_time' if track_processing else 'debugging_time'
                    result[timing_key] = execution_time
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logging.error(f"{func_name} failed after {execution_time:.3f}s: {str(e)}")
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