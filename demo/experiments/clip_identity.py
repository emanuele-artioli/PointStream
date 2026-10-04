"""Explicit, hashed input identities for rebuilding historical demo assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

CLIP_IDS = ("clip_01", "clip_02", "clip_03")


def file_sha256(path: Path, *, timeout: float = 60) -> str:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("identity timeout must be positive and finite")
    script = """import hashlib, pathlib, sys
h = hashlib.sha256()
with pathlib.Path(sys.argv[1]).open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"source identity read exceeded its allowance: {path}") from exc
    return result.stdout.strip()


@dataclass(frozen=True)
class ClipInput:
    clip_id: str
    path: Path
    sha256: str

    def verify(self) -> None:
        if not self.path.is_file() or file_sha256(self.path) != self.sha256:
            raise ValueError(f"source identity mismatch for {self.clip_id}: {self.path}")

    def receipt(self) -> dict[str, str]:
        return {"clip_id": self.clip_id, "path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class ClipManifest:
    path: Path
    sha256: str
    clips: tuple[ClipInput, ...]


def load_clip_manifest(path: Path) -> ClipManifest:
    """Reject positional legacy manifests; filenames are not input identity."""
    path = path.resolve(strict=True)
    payload = path.read_bytes()
    rows = json.loads(payload)
    if not isinstance(rows, list) or len(rows) != len(CLIP_IDS):
        raise ValueError("source manifest must contain exactly three explicitly identified clips")
    clips: dict[str, ClipInput] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each source manifest row must be an object")
        clip_id, source, digest = row.get("clip_id"), row.get("path"), row.get("sha256")
        if clip_id not in CLIP_IDS or clip_id in clips:
            raise ValueError("source manifest needs each unique clip_id: clip_01, clip_02, clip_03")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                f"explicit source SHA-256 required for {clip_id}; legacy manifests need review"
            )
        if not isinstance(source, str) or not source:
            raise ValueError(f"source path required for {clip_id}")
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        clip = ClipInput(clip_id, source_path.resolve(strict=True), digest)
        clip.verify()
        clips[clip_id] = clip
    if len({clip.path for clip in clips.values()}) != len(CLIP_IDS):
        raise ValueError("different clip IDs must identify different source files")
    return ClipManifest(
        path, hashlib.sha256(payload).hexdigest(), tuple(clips[key] for key in CLIP_IDS)
    )


def check_output_paths(paths: tuple[Path, ...], *, source_root: Path) -> None:
    """Keep new runs outside source and reject historical output reuse."""
    root = source_root.resolve()
    for path in paths:
        resolved = path.resolve()
        if resolved == root or resolved.is_relative_to(root):
            raise ValueError(f"demo output must be outside the code tree: {path}")
        if path.exists() or path.is_symlink():
            raise ValueError(f"demo output must be new; preserve existing artifacts: {path}")
