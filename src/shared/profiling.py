"""Profiling utilities for pipeline stage instrumentation.

This module provides the foundation for recording step-by-step pipeline timings.
Currently used by transport layer; can be extended to cover all pipeline stages.

## Usage Pattern:

```python
from src.shared.profiling import PipelineProfiler

profiler = PipelineProfiler()
with profiler.stage("actor_detection"):
    # ... detection code
with profiler.stage("segmentation"):
    # ... segmentation code

timings = profiler.get_timings()  # {"actor_detection": 0.5, "segmentation": 0.3, ...}
```

## Extending to All Pipeline Stages:

Future work should add profiling to:
1. Background modeling (src/encoder/background_modeler.py)
2. Actor extraction phases:
   - Detection (yolo or yoloe model inference)
   - Segmentation (separate or joint)
   - Pose estimation (yolo or dwpose)
   - Payload encoding
3. Reference extraction (crop extraction and JPEG encoding)
4. Ball extraction (detection or residual-based)
5. Residual calculation (synthesis + comparison + encoding)
6. Decoder rendering (genai_baseline or synthesis)

Thread-safety note: The simple dict-based approach works for sequential stages
but needs locking or thread-local storage for parallel/streaming execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Generator


@dataclass(frozen=False)
class PipelineProfiler:
    """Simple stage-based profiler for pipeline components."""

    _timings: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager for profiling a named pipeline stage."""
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self._timings[stage_name] = elapsed

    def get_timings(self) -> dict[str, float]:
        """Return all recorded timings as {stage_name: elapsed_sec}."""
        return dict(self._timings)

    def to_summary_dict(self) -> dict[str, float | None]:
        """Convert to JSON-serializable format for run_summary inclusion."""
        return {f"profile_{k}": v for k, v in self._timings.items()}
