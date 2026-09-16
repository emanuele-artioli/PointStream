"""Extract and curate the longest manipulation clips from worker 001."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("/home/itec/emanuele/Datasets/Egocentric-10K/raw")
DEFAULT_CURATED_DIR = Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated")


def get_video_info(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"valid": False, "frames": 0, "width": 0, "height": 0, "fps": 0.0, "duration": 0.0}

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "valid": True,
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
    }


def extract_tar(tar_path: Path, extract_to: Path) -> list[Path]:
    extract_to.mkdir(parents=True, exist_ok=True)
    extracted_mp4s: list[Path] = []
    logger.info(f"Inspecting and extracting {tar_path}...")
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mp4"):
                tar.extract(member, path=extract_to)
                extracted_path = extract_to / member.name
                extracted_mp4s.append(extracted_path)
            elif member.name.endswith(".json"):
                tar.extract(member, path=extract_to)
    logger.info(f"Extracted {len(extracted_mp4s)} mp4 clips.")
    return extracted_mp4s


def curate_longest_clips(
    extracted_clips: list[Path],
    curated_dir: Path,
    top_k: int = 3,
    min_duration_sec: float = 4.0,
) -> list[dict[str, Any]]:
    curated_dir.mkdir(parents=True, exist_ok=True)
    clip_records: list[dict[str, Any]] = []

    for clip in extracted_clips:
        info = get_video_info(clip)
        if not info["valid"] or info["duration"] < min_duration_sec:
            continue
        info["path"] = clip
        info["name"] = clip.stem
        clip_records.append(info)

    # Sort by duration descending to pick the longest clips
    clip_records.sort(key=lambda x: x["duration"], reverse=True)
    selected = clip_records[:top_k]

    logger.info(f"Selected top {len(selected)} longest clips for worker 001:")
    curated_manifest: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        dest_name = f"clip_{rank:02d}_{item['name']}.mp4"
        dest_path = curated_dir / dest_name
        shutil.copy2(item["path"], dest_path)
        logger.info(
            f"  Clip {rank}: {dest_name} | {item['frames']} frames, {item['fps']} fps, {item['duration']:.2f}s ({item['width']}x{item['height']})"
        )
        curated_manifest.append(
            {
                "rank": rank,
                "filename": dest_name,
                "path": str(dest_path),
                "frames": item["frames"],
                "fps": item["fps"],
                "width": item["width"],
                "height": item["height"],
                "duration_sec": item["duration"],
            }
        )

    manifest_path = curated_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(curated_manifest, f, indent=2)
    logger.info(f"Saved curation manifest to {manifest_path}")
    return curated_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and curate longest clips from Egocentric-10K")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with tar files")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR, help="Output directory for curated clips")
    parser.add_argument("--top-k", type=int, default=3, help="Number of longest clips to select")
    args = parser.parse_args()

    tars = list(args.raw_dir.glob("*.tar"))
    if not tars:
        # Check subdirectories
        tars = list(args.raw_dir.rglob("*.tar"))
    if not tars:
        raise FileNotFoundError(f"No tar files found in {args.raw_dir}")

    extracted: list[Path] = []
    extract_dir = args.raw_dir / "extracted"
    for tar_path in tars:
        extracted.extend(extract_tar(tar_path, extract_dir))
    curate_longest_clips(extracted, args.curated_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
