"""Runner-owned identity, exclusive ownership, and cumulative attempt timing.

No pickle and no experiment-layer dependencies. A hard kill leaves a running
attempt whose periodically saved duration is a lower bound, never a full cost.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.runner.chunk_checkpoint import file_digest


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"dtype": str(array.dtype), "shape": list(array.shape),
                "sha256": hashlib.sha256(array.data).hexdigest()}
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"checkpoint identity cannot represent {type(value).__name__}")


def runner_identity(config: Any, chunks: Any, objects: Any, contexts: Any, extra: str | None) -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted(root.rglob("*.py"))
    with ThreadPoolExecutor(max_workers=24) as pool:
        code = list(zip((str(path.relative_to(root)) for path in paths), pool.map(file_digest, paths)))
    payload = {"schema": 2, "config": config, "sources": chunks, "objects": objects,
               "contexts": contexts, "implementation": code, "injected_identity": extra}
    return hashlib.sha256(json.dumps(_plain(payload), sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(".pending")
    with temp.open("w") as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RecoverySession:
    """One owner at a time. Persist elapsed work even while a stage blocks."""

    def __init__(self, root: Path, identity: str, *, clock: Any = time.monotonic,
                 started_at: float | None = None) -> None:
        self.root, self.clock = root, clock
        root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / "owner.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = root / "identity.json"
            if path.exists():
                if json.loads(path.read_text()) != {"schema": 2, "identity": identity}:
                    raise ValueError("checkpoint identity mismatch; use a new checkpoint directory")
            elif any(path.name != "owner.lock" for path in root.iterdir()):
                raise ValueError("legacy checkpoint has no verified identity; use a new directory")
            else:
                atomic_json(path, {"schema": 2, "identity": identity})
            timing = root / "timing.json"
            self.attempts = json.loads(timing.read_text()) if timing.exists() else []
            for attempt in self.attempts:
                if attempt["status"] == "running":
                    attempt["status"] = "interrupted"
            self.current = {"status": "running", "seconds": 0.0, "max_checkpoint_gap": 0.0}
            self.attempts.append(self.current)
            self.start = self.clock() if started_at is None else started_at
            self.last_checkpoint = self.start
            self.stop = threading.Event()
            self.error: BaseException | None = None
            self.mutex = threading.Lock()
            self._write()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        except BaseException:
            self.lock.close()
            raise

    def _write(self) -> None:
        self.current["seconds"] = self.clock() - self.start
        self.current["max_checkpoint_gap"] = max(
            self.current["max_checkpoint_gap"], self.clock() - self.last_checkpoint,
        )
        atomic_json(self.root / "timing.json", self.attempts)

    def _loop(self) -> None:
        try:
            while not self.stop.wait(30.0):
                with self.mutex:
                    self._write()
        except BaseException as exc:
            self.error = exc

    def checkpoint(self) -> None:
        with self.mutex:
            self._write()
            self.last_checkpoint = self.clock()

    def finish(self, *, success: bool) -> dict[str, Any]:
        self.stop.set()
        self.thread.join()
        try:
            if self.error is not None:
                raise RuntimeError("could not persist recovery timing") from self.error
            self.current["status"] = "complete" if success else "failed"
            self._write()
            complete = all(item["status"] != "interrupted" for item in self.attempts)
            total = sum(item["seconds"] for item in self.attempts)
            gap = max(item["max_checkpoint_gap"] for item in self.attempts)
            return {"invocation_seconds": self.current["seconds"],
                    "run_seconds": total if complete else None,
                    "run_seconds_lower_bound": total, "timing_complete": complete,
                    "attempts": len(self.attempts), "max_checkpoint_gap_seconds": gap,
                    "hourly_checkpoint_budget_met": gap <= 3600.0 if complete else None}
        finally:
            self.lock.close()
