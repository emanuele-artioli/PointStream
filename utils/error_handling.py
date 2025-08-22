#!/usr/bin/env python3
"""
Standardized Error Handling Utilities

This module provides consistent error handling patterns across all PointStream components.
Simplifies error logging and response formatting.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable
from functools import wraps


def safe_execute(operation_name: str, fallback_result: Any = None, 
                 log_traceback: bool = False):
    """
    Decorator for safe execution with standardized error handling.
    
    Args:
        operation_name: Name of the operation for logging
        fallback_result: Result to return on failure
        log_traceback: Whether to log full traceback
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{operation_name} failed: {str(e)}"
                
                if log_traceback:
                    logging.error(error_msg, exc_info=True)
                else:
                    logging.error(error_msg)
                
                # Return standardized error result
                if isinstance(fallback_result, dict):
                    result = fallback_result.copy()
                    result['error'] = str(e)
                    result['success'] = False
                    return result
                else:
                    return fallback_result
        
        return wrapper
    return decorator


def create_error_result(operation: str, error: Exception, 
                       additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized error result dictionary.
    
    Args:
        operation: Name of the operation that failed
        error: The exception that occurred
        additional_data: Optional additional data to include
        
    Returns:
        Standardized error result dictionary
    """
    result = {
        'success': False,
        'error': str(error),
        'operation': operation
    }
    
    if additional_data:
        result.update(additional_data)
    
    return result


def create_success_result(operation: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized success result dictionary.
    
    Args:
        operation: Name of the operation that succeeded
        data: Optional data to include
        
    Returns:
        Standardized success result dictionary
    """
    result = {
        'success': True,
        'operation': operation
    }
    
    if data:
        result.update(data)
    
    return result


def log_and_return_error(operation: str, error: Exception, 
                        fallback_result: Any = None,
                        log_traceback: bool = False) -> Any:
    """
    Log an error and return a fallback result.
    
    Args:
        operation: Name of the operation that failed
        error: The exception that occurred
        fallback_result: Result to return on failure
        log_traceback: Whether to log full traceback
        
    Returns:
        The fallback result
    """
    error_msg = f"{operation} failed: {str(error)}"
    
    if log_traceback:
        logging.error(error_msg, exc_info=True)
    else:
        logging.error(error_msg)
    
    if isinstance(fallback_result, dict):
        fallback_result['error'] = str(error)
        fallback_result['success'] = False
    
    return fallback_result
