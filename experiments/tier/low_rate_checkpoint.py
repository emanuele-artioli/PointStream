"""Resumable per-point JSON checkpoints. A crash must not lose finished work.

Each point is one file. The aggregate report is rewritten after every point so
a later session can resume without re-encoding.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
