from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.multiprocessing as tmp


class BaseExecutionPool(ABC):
    @abstractmethod
    def execute(self, tag: str, func: Any, context: dict[str, Any], deps: dict[str, Any]) -> Any:
        """Execute one DAG node under the requested resource tag."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release pool resources."""


class InlineExecutionPool(BaseExecutionPool):
    def execute(self, tag: str, func: Any, context: dict[str, Any], deps: dict[str, Any]) -> Any:
        return func(context=context, deps=deps)

    def shutdown(self) -> None:
        return None


@dataclass
class WorkerConfig:
    cpu_workers: int = 1
    gpu_workers: int = 1
    start_method: str = "spawn"


class TaggedMultiprocessPool(BaseExecutionPool):
    """Stub worker pool that prepares torch.multiprocessing resources by tag.

    The current scaffold intentionally runs callables inline while preserving a
    hardware-aware API contract for future queue/process execution.
    """

    def __init__(self, config: WorkerConfig | None = None) -> None:
        self._config = config or WorkerConfig()
        self._ctx = tmp.get_context(self._config.start_method)
        self._cpu_queue = self._ctx.Queue()
        self._gpu_queue = self._ctx.Queue()
        self._cpu_dispatch_count = 0
        self._gpu_dispatch_count = 0

    def execute(self, tag: str, func: Any, context: dict[str, Any], deps: dict[str, Any]) -> Any:
        if tag == "gpu":
            self._gpu_queue.put(("queued", func.__name__))
            self._gpu_dispatch_count += 1
        else:
            self._cpu_queue.put(("queued", func.__name__))
            self._cpu_dispatch_count += 1
        return func(context=context, deps=deps)

    def shutdown(self) -> None:
        self._cpu_queue.close()
        self._gpu_queue.close()

    @property
    def cpu_dispatch_count(self) -> int:
        return self._cpu_dispatch_count

    @property
    def gpu_dispatch_count(self) -> int:
        return self._gpu_dispatch_count


def make_shared_cpu_tensor(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Allocate a CPU tensor in shared memory for process-safe handoff.

    Shape: [*shape]
    """

    return torch.zeros(*shape, dtype=dtype).share_memory_()
