#!/usr/bin/env python3
"""
Standardized Error Handling Utilities
"""

from typing import Any, Callable
from functools import wraps

def safe_execute(operation_name: str, fallback_result: Any = None, 
                 log_traceback: bool = False):
    """
    Decorator for safe execution. Returns a fallback result on failure.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Logging is removed.
                if isinstance(fallback_result, dict):
                    result = fallback_result.copy()
                    result['error'] = str(e)
                    result['success'] = False
                    return result
                else:
                    return fallback_result
        
        return wrapper
    return decorator
