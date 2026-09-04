"""Cumulative wall budget for the BP53 diagnostic batch.

Consumed seconds are a lower bound of work already done: completed attempt
walls, control time, and the last heartbeat of an interrupted attempt.
Idle time between processes is never charged. Time after the last heartbeat
of a crashed process is unknown and must be flagged, not treated as zero.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WALL_BUDGET_S = 8 * 3600
POINT_RESERVE_S = 3500
HOURLY_LIMIT_S = 3600.0
HOURLY_CLEARANCE_S = 1.0
HEARTBEAT_INTERVAL_S = 60.0

Clock = Callable[[], float]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_budget() -> dict[str, Any]:
    return {
        "consumed_seconds": 0.0,
        "events": [],
        "wall_budget_seconds": WALL_BUDGET_S,
        "hourly_limit_seconds": HOURLY_LIMIT_S,
        "hourly_clearance_seconds": HOURLY_CLEARANCE_S,
        "longer_runs_operationally_cleared": False,
        "unknown_crash_interval": False,
        "consumed_seconds_is_lower_bound": False,
        "active_attempt": None,
        "note": (
            "consumed_seconds is controls plus attempt walls plus known "
            "interrupted elapsed. It does not include idle time between "
            "processes. Time after the last heartbeat of a crash is flagged, "
            "not invented."
        ),
    }


def load_budget(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_budget()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a budget object")
    merged = empty_budget()
    merged.update(payload)
    merged["consumed_seconds"] = float(merged.get("consumed_seconds") or 0.0)
    merged["events"] = list(merged.get("events") or [])
    merged["wall_budget_seconds"] = float(
        merged.get("wall_budget_seconds") or WALL_BUDGET_S
    )
    if "active_attempt" not in payload:
        merged["active_attempt"] = None
    return merged


def write_budget(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return state


def remaining_seconds(state: dict[str, Any]) -> float:
    return float(state["wall_budget_seconds"]) - float(state["consumed_seconds"])


def over_budget(state: dict[str, Any]) -> bool:
    return float(state["consumed_seconds"]) > float(state["wall_budget_seconds"])


def budget_alarms(state: dict[str, Any]) -> list[str]:
    if not over_budget(state):
        return []
    consumed = float(state["consumed_seconds"])
    limit = float(state["wall_budget_seconds"])
    return [
        f"consumed {consumed:.3f}s exceeds wall budget {limit:.3f}s; "
        "report must not be complete"
    ]


def longer_runs_cleared(max_checkpoint_gap_seconds: float) -> bool:
    """True when the worst gap is at least one second inside the hourly limit.

    A gap of 3599.72 s met ``gap <= 3600`` and still does not clear longer runs.
    """
    return float(max_checkpoint_gap_seconds) <= HOURLY_LIMIT_S - HOURLY_CLEARANCE_S


def _has_charge(state: dict[str, Any], *, kind: str, name: str) -> bool:
    return any(
        item.get("kind") == kind and item.get("name") == name
        for item in state.get("events") or []
    )


def add_consumed(
    path: Path,
    seconds: float,
    *,
    kind: str,
    name: str = "",
) -> dict[str, Any]:
    """Append a charge. Prefer finish_attempt / reconcile for point walls."""
    state = load_budget(path)
    added = float(seconds)
    if added < 0:
        raise ValueError("consumed seconds must be non-negative")
    state["consumed_seconds"] = float(state["consumed_seconds"]) + added
    state["events"].append(
        {
            "kind": kind,
            "name": name,
            "seconds": added,
            "at": _utc_now(),
        }
    )
    return write_budget(path, state)


def begin_attempt(path: Path, name: str, *, kind: str = "attempt") -> dict[str, Any]:
    """Record that expensive work is starting, before it runs."""
    state = load_budget(path)
    if state.get("active_attempt"):
        raise ValueError(
            "an attempt is already active; recover_interrupted before starting another"
        )
    stamp = _utc_now()
    state["active_attempt"] = {
        "kind": kind,
        "name": name,
        "started_at": stamp,
        "last_elapsed_seconds": 0.0,
        "last_heartbeat_at": stamp,
    }
    return write_budget(path, state)


def heartbeat_attempt(path: Path, elapsed_seconds: float) -> dict[str, Any]:
    """Persist known elapsed time of the in-process attempt. Not a charge."""
    state = load_budget(path)
    active = state.get("active_attempt")
    if not isinstance(active, dict):
        raise ValueError("heartbeat without an active attempt")
    known = max(0.0, float(elapsed_seconds))
    active["last_elapsed_seconds"] = known
    active["last_heartbeat_at"] = _utc_now()
    state["active_attempt"] = active
    return write_budget(path, state)


def finish_attempt(
    path: Path,
    name: str,
    seconds: float,
    *,
    kind: str = "attempt",
) -> dict[str, Any]:
    """Charge a completed attempt once and clear the active record."""
    state = load_budget(path)
    state["active_attempt"] = None
    if _has_charge(state, kind=kind, name=name):
        return write_budget(path, state)
    added = float(seconds)
    if added < 0:
        raise ValueError("consumed seconds must be non-negative")
    state["consumed_seconds"] = float(state["consumed_seconds"]) + added
    state["events"].append(
        {
            "kind": kind,
            "name": name,
            "seconds": added,
            "at": _utc_now(),
        }
    )
    return write_budget(path, state)


def recover_interrupted(path: Path, points_dir: Path | None = None) -> dict[str, Any]:
    """Adopt a leftover active attempt after a process death.

    If a point checkpoint already has an attempt wall, this is the
    checkpoint/charge crash window: keep the checkpoint as the charge source
    and do not add heartbeat elapsed. If there is no checkpoint, keep the last
    heartbeat as a lower bound and flag the unknown interval after it. Idle
    time since the crash is not added.
    """
    state = load_budget(path)
    active = state.get("active_attempt")
    if not isinstance(active, dict):
        return state
    name = str(active.get("name") or "")
    checkpoint_wall: float | None = None
    if points_dir is not None and name:
        from experiments.tier.low_rate_checkpoint import load_checkpoint

        row = load_checkpoint(points_dir, name)
        wall = None if row is None else row.get("attempt_wall_seconds")
        if isinstance(wall, (int, float)):
            checkpoint_wall = float(wall)
    if checkpoint_wall is not None:
        state["active_attempt"] = None
        state["events"].append(
            {
                "kind": "charge_window",
                "name": name,
                "seconds": 0.0,
                "at": _utc_now(),
                "note": (
                    "checkpoint had attempt_wall_seconds; crash was after "
                    "the durable row and before or during the budget charge"
                ),
            }
        )
        return write_budget(path, state)

    known = float(active.get("last_elapsed_seconds") or 0.0)
    state["consumed_seconds"] = float(state["consumed_seconds"]) + known
    state["unknown_crash_interval"] = True
    state["consumed_seconds_is_lower_bound"] = True
    state["events"].append(
        {
            "kind": "interrupted",
            "name": name,
            "seconds": known,
            "at": _utc_now(),
            "unknown_crash_interval": True,
            "last_heartbeat_at": active.get("last_heartbeat_at"),
            "note": (
                "known elapsed is the last heartbeat. Time after that "
                "heartbeat until the crash is unknown; idle since the crash "
                "is not charged."
            ),
        }
    )
    state["active_attempt"] = None
    return write_budget(path, state)


def reconcile_checkpoints(
    path: Path,
    points_dir: Path,
    names: Sequence[str],
) -> dict[str, Any]:
    """Charge each finished point checkpoint exactly once."""
    from experiments.tier.low_rate_checkpoint import load_checkpoint

    state = load_budget(path)
    changed = False
    for name in names:
        row = load_checkpoint(points_dir, name)
        if row is None:
            continue
        wall = row.get("attempt_wall_seconds")
        if not isinstance(wall, (int, float)):
            continue
        if _has_charge(state, kind="attempt", name=name):
            continue
        state["consumed_seconds"] = float(state["consumed_seconds"]) + float(wall)
        state["events"].append(
            {
                "kind": "attempt",
                "name": name,
                "seconds": float(wall),
                "at": _utc_now(),
                "source": "checkpoint",
            }
        )
        changed = True
    if changed:
        return write_budget(path, state)
    return state


def reconstruct_from_events(
    path: Path,
    events: list[dict[str, Any]],
    *,
    max_checkpoint_gap_seconds: float,
    note: str,
) -> dict[str, Any]:
    """Write a budget file from known walls. Does not invent missing control time."""
    state = empty_budget()
    state["events"] = list(events)
    state["consumed_seconds"] = sum(float(item["seconds"]) for item in events)
    state["longer_runs_operationally_cleared"] = longer_runs_cleared(
        max_checkpoint_gap_seconds
    )
    state["max_checkpoint_gap_seconds"] = float(max_checkpoint_gap_seconds)
    state["reconstructed"] = True
    state["consumed_seconds_is_lower_bound"] = True
    state["unknown_crash_interval"] = False
    state["note"] = note
    return write_budget(path, state)


class AttemptSession:
    """Begin an attempt, pulse elapsed while work runs, leave charging to finish."""

    def __init__(
        self,
        path: Path,
        name: str,
        *,
        kind: str = "attempt",
        interval_s: float = HEARTBEAT_INTERVAL_S,
        clock: Clock | None = None,
    ) -> None:
        self.path = path
        self.name = name
        self.kind = kind
        self.interval_s = float(interval_s)
        self.clock = clock or time.perf_counter
        self._started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> AttemptSession:
        self._started = self.clock()
        begin_attempt(self.path, self.name, kind=self.kind)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def elapsed(self) -> float:
        return max(0.0, self.clock() - self._started)

    def pulse(self) -> None:
        with self._lock:
            heartbeat_attempt(self.path, self.elapsed())

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.pulse()

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s + 1.0))
        try:
            self.pulse()
        except ValueError:
            # recover_interrupted may have cleared the record in tests.
            pass
