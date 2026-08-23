"""The single run path: chunk loop, routing, accounting, evaluation.

``experiments/`` consumes this package as a library. This package does not
shell out and does not scrape stdout.
"""

from src.runner.accounting import SIZE_SUM_TOLERANCE, SizesBytes
from src.runner.run import ChunkResult, RunResult, run
from src.runner.routing import lattice_config_from

__all__ = [
    "SIZE_SUM_TOLERANCE",
    "ChunkResult",
    "RunResult",
    "SizesBytes",
    "lattice_config_from",
    "run",
]
