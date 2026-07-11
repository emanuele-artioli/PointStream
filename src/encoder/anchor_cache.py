"""Content-addressed cache for anchor/fallback-codec encodes (report 10
Phase 3 "Experiment-efficiency architecture").

Scene cuts are deterministic and shared across all tiers and runs (report
10's "Deterministic shared segmentation" design rule: cuts are cached once
per video and never recomputed per tier). That means a fallback-codec
encode of a given scene span at a given codec/CRF/preset never needs to be
redone across separate `encode_full_match` runs on the same video — only
when the video, span, or codec settings actually change. This module keys
the cache on the *original* video and scene span (stable across runs),
not on the extracted temporary clip file (which gets a fresh path every
run).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable


def video_fingerprint(video_path: Path) -> str:
    """Cheap identity proxy for a video file: path + size + mtime.

    Full-file hashing would be correct but expensive for multi-GB 4K
    matches; path+size+mtime is what most build/content caches use for the
    same reason and is sufficient here since we control how these files are
    produced (raw_4k sources are not edited in place).
    """
    stat = video_path.stat()
    return f"{video_path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"


def anchor_cache_key(
    video_path: Path,
    t_start: float,
    t_end: float,
    codec: str,
    crf: int | None,
    preset: str | None,
) -> str:
    raw = "|".join(
        [
            video_fingerprint(video_path),
            f"{t_start:.6f}",
            f"{t_end:.6f}",
            codec,
            str(crf),
            str(preset),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_anchor_path(cache_root: Path, key: str) -> Path:
    return cache_root / f"{key}.mp4"


def get_or_encode(
    cache_root: Path,
    video_path: Path,
    t_start: float,
    t_end: float,
    codec: str,
    crf: int | None,
    preset: str | None,
    output_path: Path,
    encode_fn: Callable[[Path], None],
) -> tuple[int, bool]:
    """Returns `(bytes, was_cache_hit)`; writes the result to `output_path`.

    On a miss, `encode_fn(output_path)` must perform the real encode and
    leave the result at `output_path`; it is then copied into the cache for
    future calls.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    key = anchor_cache_key(video_path, t_start, t_end, codec, crf, preset)
    cached_path = cached_anchor_path(cache_root, key)

    if cached_path.exists():
        shutil.copy2(cached_path, output_path)
        return int(output_path.stat().st_size), True

    encode_fn(output_path)
    shutil.copy2(output_path, cached_path)
    return int(output_path.stat().st_size), False
