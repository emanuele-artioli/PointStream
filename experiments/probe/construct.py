"""Construct every registered backend, or record a stated reason it cannot.

The required-behaviour suite (PLAN.md §8) asks: every registered backend
constructs, or fails with a stated reason. A raw ``AttributeError`` from a
nested import is a wrapper bug, not a reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.components import all_registries
from src.contracts.registry import BackendSpec, Registry

# Exceptions that can carry a human-written limitation. An AttributeError or
# TypeError is a construction bug unless it matches a known environment block.
_STATED_TYPES = (
    RuntimeError,
    FileNotFoundError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    ValueError,
)

# PLAN.md §2.4: SAM3 cannot load because torch.nn.attention does not exist in
# torch 2.2.2. The detector/segmenter wrappers do not yet restate that; they
# leak the AttributeError. This helper names the known block so the invariant
# still distinguishes it from a silent crash. Reported as a leftover, not fixed
# here (detection/segmentation are not this stream's files).
_SAM3_ATTENTION_MARKERS = ("torch.nn.attention", "nn.attention")


@dataclass(frozen=True)
class ConstructRecord:
    """One backend's construction attempt."""

    axis: str
    name: str
    ok: bool
    reason: str | None = None
    exception_type: str | None = None


@dataclass(frozen=True)
class ConstructReport:
    records: tuple[ConstructRecord, ...]
    failures: tuple[ConstructRecord, ...] = field(default_factory=tuple)

    @property
    def constructed(self) -> int:
        return sum(1 for item in self.records if item.ok)

    @property
    def refused(self) -> int:
        return sum(1 for item in self.records if not item.ok and item.reason)


def stated_reason(exc: BaseException, *, axis: str, name: str) -> str | None:
    """Return the limitation the backend stated, or None if this looks like a bug."""
    message = str(exc).strip()
    lowered = message.lower()
    if any(marker in lowered for marker in _SAM3_ATTENTION_MARKERS) or (
        name == "sam3" and "attention" in lowered
    ):
        return (
            f"{axis}/{name} cannot load: torch.nn.attention is missing in this "
            "environment (torch 2.2.2). PLAN.md §2.4. The wrapper does not yet "
            "restate this; it currently leaks the nested AttributeError."
        )
    if not message:
        return None
    if isinstance(exc, _STATED_TYPES):
        return message
    return None


def construct_one(registry: Registry[object], spec: BackendSpec[object]) -> ConstructRecord:
    """Build ``spec`` on ``registry``. Never swallows a bug-shaped exception."""
    try:
        registry.build(spec.name)
    except Exception as exc:
        reason = stated_reason(exc, axis=registry.axis, name=spec.name)
        return ConstructRecord(
            axis=registry.axis,
            name=spec.name,
            ok=False,
            reason=reason,
            exception_type=type(exc).__name__,
        )
    return ConstructRecord(axis=registry.axis, name=spec.name, ok=True)


def construct_all(
    registries: dict[str, Registry[object]] | None = None,
) -> ConstructReport:
    """Try every registered backend on every axis."""
    tables = registries if registries is not None else all_registries()
    records: list[ConstructRecord] = []
    failures: list[ConstructRecord] = []
    for axis in sorted(tables):
        registry = tables[axis]
        for spec in registry:
            record = construct_one(registry, spec)
            records.append(record)
            if not record.ok and record.reason is None:
                failures.append(record)
    return ConstructReport(records=tuple(records), failures=tuple(failures))


def as_dicts(report: ConstructReport) -> list[dict[str, Any]]:
    return [
        {
            "axis": item.axis,
            "name": item.name,
            "ok": item.ok,
            "reason": item.reason,
            "exception_type": item.exception_type,
        }
        for item in report.records
    ]
