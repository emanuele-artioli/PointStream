"""Resumable per-point JSON checkpoints. A crash must not lose finished work.

Each point is one file. The aggregate report is rewritten after every point so
a later session can resume without re-encoding.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def fingerprint(value: Any) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def implementation_digest(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[2]
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
         "src", "experiments", "config", "pyproject.toml"], cwd=root, text=True,
    ).splitlines()
    def digest(name: str) -> tuple[str, str]:
        return name, hashlib.sha256((root / name).read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=24) as pool:
        return fingerprint(list(pool.map(digest, sorted(set(names)))))


def source_identity(clips: list[Any]) -> list[dict[str, Any]]:
    """Hash every decoded input frame, not just three manifest samples."""
    return [
        {"context_id": clip.context_id, "shape": list(clip.frames.shape),
         "sha256": hashlib.sha256(np.ascontiguousarray(clip.frames).data).hexdigest()}
        for clip in clips
    ]


def guard_checkpoints(directory: Path, identity: dict[str, Any]) -> None:
    """Fail closed rather than relabel old points with the current configuration."""
    path = directory / "identity.json"
    expected = fingerprint(identity)
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != expected:
            raise SystemExit("checkpoint identity changed; use a new output directory")
    elif directory.exists() and any(directory.glob("*.json")):
        raise SystemExit("unverified legacy checkpoints; use a new output directory")
    else:
        write_json(path, {"fingerprint": expected, "identity": identity})


def completion_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    succeeded = sum(bool((row.get("pointstream") or {}).get("usable")) for row in rows)
    return {"submitted": len(rows), "succeeded": succeeded, "failed": len(rows) - succeeded}


def save_checkpoint(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{name}.json"
    body = dict(payload)
    body.setdefault("checkpointed", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest


def load_checkpoint(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / f"{name}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a JSON object; delete it and rerun.")
    return payload


def write_json(dest: Path, payload: dict[str, Any]) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(dest)
    return dest


__all__ = ["load_checkpoint", "save_checkpoint", "write_json"]
