from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _tag_function(func: F, tag_name: str) -> F:
    setattr(func, "_execution_tag", tag_name)
    return func


def cpu_bound(func: F) -> F:
    """Marks a callable as CPU-bound for orchestrator scheduling."""

    return _tag_function(func, "cpu")


def gpu_bound(func: F) -> F:
    """Marks a callable as GPU-bound for orchestrator scheduling."""

    return _tag_function(func, "gpu")
