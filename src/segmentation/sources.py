"""Clip input (video file or image directory) and run provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
REPO_ROOT = Path(__file__).resolve().parents[2]


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def iter_frames(source: Path | str, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """BGR uint8 frames, as OpenCV and Ultralytics expect them."""
    import cv2

    path = Path(source)
    if path.is_dir():
        for index, image in enumerate(image_files(path)):
            if max_frames is not None and index >= max_frames:
                return
            frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"cannot read {image}")
            yield frame
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    try:
        index = 0
        while max_frames is None or index < max_frames:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
            index += 1
    finally:
        capture.release()


def video_fps(source: Path | str, default: float = 30.0) -> float:
    path = Path(source)
    if path.is_dir():
        return default
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    return fps if fps > 1e-3 else default


def extract_jpegs(
    source: Path | str,
    dest: Path,
    max_frames: int | None = None,
    *,
    start: int = 0,
    fps: float | None = None,
    ffmpeg: str | None = None,
) -> int:
    """``dest/00000.jpg …`` from frame ``start`` on, the layout SAM 3.1's loader reads.

    Returns the number of frames written; past the end of the clip that is 0.
    """
    dest.mkdir(parents=True, exist_ok=True)
    path = Path(source)
    if path.is_dir():
        images = image_files(path)[start : None if max_frames is None else start + max_frames]
        from PIL import Image

        for index, image in enumerate(images):
            target = dest / f"{index:05d}.jpg"
            if image.suffix.lower() in {".jpg", ".jpeg"}:
                shutil.copyfile(image, target)
            else:
                Image.open(image).convert("RGB").save(target, quality=95)
        count = len(images)
    else:
        command = [ffmpeg or shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error"]
        if start:
            if not fps:
                raise ValueError("seeking into a video needs its fps")
            # Half a frame early, so the first decoded frame is exactly `start`.
            command += ["-ss", f"{(start - 0.5) / fps:.6f}"]
        command += ["-i", str(path)]
        if max_frames is not None:
            command += ["-frames:v", str(int(max_frames))]
        command += ["-q:v", "2", "-start_number", "0", str(dest / "%05d.jpg")]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"frame extraction failed for {path}: {result.stderr[-1500:]}")
        count = len(list(dest.glob("*.jpg")))
    if count == 0 and start == 0:
        raise RuntimeError(f"no frames extracted from {path}")
    return count


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(source: Path | str, max_frames: int | None = None) -> dict[str, Any]:
    """sha256 of the video file, or of the ordered frame files a run read."""
    path = Path(source).resolve()
    stat = path.stat()
    return dict(_source_identity(path, max_frames, stat.st_size, stat.st_mtime_ns))


@lru_cache(maxsize=64)
def _source_identity(
    path: Path, max_frames: int | None, _size: int, _mtime: int
) -> tuple[tuple[str, Any], ...]:
    """Cached per file version: every backend in a suite reads the same clip."""
    return tuple(_hash_source(path, max_frames).items())


def _hash_source(path: Path, max_frames: int | None) -> dict[str, Any]:
    if path.is_dir():
        digest = hashlib.sha256()
        images = image_files(path)[:max_frames]
        for image in images:
            digest.update(image.name.encode() + b"\0" + sha256_file(image).encode())
        return {
            "path": str(path),
            "kind": "frames",
            "frames": len(images),
            "sha256": digest.hexdigest(),
        }
    return {
        "path": str(path),
        "kind": "video",
        "sha256": sha256_file(path),
        "max_frames": max_frames,
    }


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def code_identity() -> dict[str, Any]:
    head = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    if head is None:
        return {"revision": os.environ.get("PS_SOURCE_REVISION"), "git": False}
    diff = _run(["git", "-C", str(REPO_ROOT), "diff", "HEAD"]) or ""
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]) or ""
    return {
        "revision": head,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def gpu_identity() -> dict[str, Any] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    command = ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"]
    if visible:
        command[1:1] = ["-i", visible]
    out = _run(command)
    if not out:
        return None
    uuid, name, driver = (part.strip() for part in out.splitlines()[0].split(","))
    return {"uuid": uuid, "name": name, "driver": driver}


def package_versions(*names: str) -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_identity() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": package_versions("torch", "ultralytics", "numpy", "opencv-python-headless"),
        "gpu": gpu_identity(),
        "ps_stage": os.environ.get("PS_STAGE"),
    }


def write_provenance(out_dir: Path, record: dict[str, Any]) -> Path:
    target = out_dir / "provenance.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    return target


def timing_summary(
    frame_ms: list[float], *, model_load_s: float, total_s: float, frames: int
) -> dict[str, Any]:
    """Throughput is the comparable number across backends; percentiles describe streaming."""
    values = np.asarray(frame_ms, dtype=float)
    return {
        "frames": frames,
        "model_load_s": round(model_load_s, 3),
        "total_s": round(total_s, 3),
        "ms_per_frame": round(1000.0 * total_s / frames, 3) if frames else None,
        "fps": round(frames / total_s, 3) if total_s > 0 else None,
        "step_ms_p50": round(float(np.percentile(values, 50)), 3) if values.size else None,
        "step_ms_p95": round(float(np.percentile(values, 95)), 3) if values.size else None,
    }
