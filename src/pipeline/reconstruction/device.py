"""Device choice and OOM fallback as one policy, not scattered ``try``.

The pre-rewrite engine caught ``RuntimeError`` inside panorama warp, rebuilt the
generator, and retried. A second CUDA path that OOMed independently had to copy
the same dance. Here the policy is the only place that decides, and every
device-using step *receives* a device rather than discovering one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc`` is a device memory failure, by type or message.

    Named so a policy can tell OOM from a real bug. Catching every
    ``RuntimeError`` and falling back to CPU would hide shape errors as
    "we ran on CPU instead".
    """
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "cudaErrorMemoryAllocation"}:
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


@dataclass(frozen=True)
class DeviceDecision:
    """Where an operation actually ran, and whether that was the preference."""

    device: str
    fell_back: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class DevicePolicy:
    """Preferred device, and whether an OOM may retry on the fallback.

    Args:
        preferred: Device string handed to the operation first (``"cpu"``,
            ``"cuda"``, ``"cuda:0"``). Opaque to this layer — backends interpret
            it.
        fallback: Device used after an OOM, when ``allow_fallback`` is set.
        allow_fallback: When false, OOM propagates. Tests and a machine with
            no spare host RAM want that, rather than a silent second attempt.
    """

    preferred: str = "cpu"
    fallback: str = "cpu"
    allow_fallback: bool = True

    def run(self, operation: Callable[[str], T]) -> tuple[T, DeviceDecision]:
        """Call ``operation(device)``, retrying on ``fallback`` after OOM only.

        ``operation`` must be safe to call twice: the first attempt may have
        allocated nothing usable. Reconstruction stages are written that way;
        a generator that mutates itself on failure is the runner's problem.
        """
        try:
            result = operation(self.preferred)
        except Exception as exc:
            if not self._may_fallback(exc):
                raise
            result = operation(self.fallback)
            return result, DeviceDecision(self.fallback, fell_back=True, reason=str(exc))
        return result, DeviceDecision(self.preferred)

    def _may_fallback(self, exc: BaseException) -> bool:
        if not self.allow_fallback:
            return False
        if self.fallback == self.preferred:
            return False
        return is_out_of_memory(exc)
