"""Bounded process-tree CPU and monotonic wall interval recording for K5d."""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar('T')


class Timer:
    def __init__(self, path: Path):
        self.path = path
        self.events: list[dict[str, Any]] = []

    def flush(self) -> None:
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'scope': 'overlapping nested intervals; waited children, not detached descendants',
                                   'events': self.events}, allow_nan=False, indent=2) + '\n')
        tmp.replace(self.path)

    def relocate(self, path: Path) -> None:
        self.path = path
        self.flush()

    def measure(self, name: str, function: Callable[[], T]) -> T:
        start_wall = time.monotonic()
        own = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        completed = False
        try:
            value = function()
            completed = True
            return value
        finally:
            now = time.monotonic()
            after_own = resource.getrusage(resource.RUSAGE_SELF)
            after_child = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.events.append({'name': name, 'wall_seconds': now - start_wall,
                                'self_cpu_seconds': max(0., (after_own.ru_utime + after_own.ru_stime) - (own.ru_utime + own.ru_stime)),
                                'waited_children_cpu_seconds': max(0., (after_child.ru_utime + after_child.ru_stime) - (child.ru_utime + child.ru_stime)),
                                'completed': completed})
            # Event names may be nested; totals are not additive across nesting.
            self.flush()
