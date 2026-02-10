#!/usr/bin/env python3
"""
Orchestrate segmentation (SAM3) and DWpose extraction for tennis datasets.

Usage:
    python process_tennis_datasets.py --folder /path/to/dataset [--model_path <path>] [--skip-existing]

Input:
    A folder containing .mp4 video files, either at the root or in a videos/ subfolder.

Output (same structure as Datasets/djokovic_federer):
    <folder>/
    ├── videos/            # Source videos (moved here if originally at root)
    ├── crops/             # Segmented player crops per scene/id
    ├── crop_videos/       # Crops encoded as H.265 videos
    ├── dwpose_videos/     # DWpose keypoint visualisation videos
    └── meta.json          # Mapping crop_video paths → dwpose_video paths
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path
import json

BASE = Path("/home/itec/emanuele")
POINTSTREAM = BASE / "pointstream"
A1_SCRIPT = POINTSTREAM / "A1_segment_with_sam.py"
EXPERIMENTS_DIR = POINTSTREAM / "experiments"


def run_cmd(cmd, check=True):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    return res


def find_latest_experiment():
    dirs = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and d.name.endswith("_sam_seg")]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def copy_masked_crops(exp_dir, dst_crops_scene_dir):
    src = exp_dir / "masked_crops"
    if not src.exists():
        raise RuntimeError(f"Expected masked_crops in {exp_dir}, but not found")
    dst_crops_scene_dir.mkdir(parents=True, exist_ok=True)
    for id_dir in sorted(src.iterdir()):
        if not id_dir.is_dir():
            continue
        dst_id_dir = dst_crops_scene_dir / id_dir.name
        if dst_id_dir.exists():
            shutil.rmtree(dst_id_dir)
        shutil.copytree(id_dir, dst_id_dir)


def get_fps(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    s = res.stdout.strip()
    if not s:
        return 30
    if "/" in s:
        num, den = s.split("/")
        try:
            num = int(num); den = int(den)
            if den == 0:
                return 30
            return max(1, num // den)
        except Exception:
            return 30
    try:
        return int(round(float(s)))
    except Exception:
        return 30


def encode_crops_to_video(crops_scene_dir, src_video_path, out_scene_dir):
    fps = get_fps(src_video_path)
    out_scene_dir.mkdir(parents=True, exist_ok=True)
    for id_dir in sorted(crops_scene_dir.iterdir()):
        if not id_dir.is_dir():
            continue
        out_file = out_scene_dir / f"{id_dir.name}.mp4"
        if out_file.exists():
            print(f"Skipping existing crop video: {out_file}")
            continue
        # Check for missing frame indices. If frame numbering is non-contiguous, use glob pattern
        png_files = sorted(id_dir.glob("*.png"))
        if not png_files:
            print(f"No PNG frames found in {id_dir}; skipping")
            continue
        # Extract numeric stems where possible
        indices = []
        for p in png_files:
            try:
                indices.append(int(p.stem))
            except Exception:
                indices = None
                break
        use_glob = False
        if indices is None:
            use_glob = True
        else:
            if sorted(indices) != list(range(min(indices), max(indices) + 1)):
                use_glob = True

        if use_glob:
            print(f"Non-contiguous or non-numeric frame names in {id_dir}; using glob pattern for ffmpeg input")
            pattern = str(id_dir / "*.png")
            cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", pattern,
                   "-c:v", "libx265", "-crf", "15", "-pix_fmt", "yuv420p", str(out_file)]
        else:
            # Safe to use sequential pattern
            pattern = str(id_dir / "%05d.png")
            cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
                   "-c:v", "libx265", "-crf", "15", "-pix_fmt", "yuv420p", str(out_file)]
        run_cmd(cmd)


def find_videos(ds_dir):
    """Find .mp4 source videos. Prefers videos/ subfolder, falls back to root."""
    videos_dir = ds_dir / "videos"
    if videos_dir.is_dir():
        mp4s = sorted([p for p in videos_dir.iterdir()
                       if p.suffix.lower() == ".mp4" and not p.stem.endswith("_kps")])
        if mp4s:
            return mp4s, videos_dir

    # Fall back to mp4s at root level (exclude output subdirectories)
    mp4s = sorted([p for p in ds_dir.iterdir()
                   if p.suffix.lower() == ".mp4" and not p.stem.endswith("_kps")])
    if mp4s:
        # Move them into videos/ for consistent structure
        videos_dir.mkdir(parents=True, exist_ok=True)
        moved = []
        for mp4 in mp4s:
            dst = videos_dir / mp4.name
            shutil.move(str(mp4), str(dst))
            moved.append(dst)
        print(f"Moved {len(moved)} videos into {videos_dir}")
        return moved, videos_dir

    return [], videos_dir


def build_meta_json(crop_videos_root, dwpose_root, meta_path):
    """Build meta.json mapping each crop video to its dwpose keypoint video."""
    entries = []
    if not crop_videos_root.exists():
        return
    for scene_dir in sorted(crop_videos_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for vid in sorted(scene_dir.iterdir()):
            if vid.suffix.lower() != ".mp4":
                continue
            kps_name = vid.stem + "_kps.mp4"
            kps_path = dwpose_root / scene_dir.name / kps_name
            entries.append({
                "video_path": str(vid),
                "kps_path": str(kps_path),
            })
    with open(meta_path, "w") as f:
        json.dump(entries, f, indent=4)
    print(f"Wrote {len(entries)} entries to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment tennis players and extract DWpose from video scenes.")
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to dataset folder (contains .mp4 files or a videos/ subfolder)")
    parser.add_argument("--model_path", type=str,
                        default=str(BASE / "Models" / "SAM" / "sam3.pt"))
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    ds_dir = Path(args.folder).resolve()
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {ds_dir}")

    crop_root = ds_dir / "crops"
    crop_videos_root = ds_dir / "crop_videos"
    dwpose_root = ds_dir / "dwpose_videos"
    crop_root.mkdir(parents=True, exist_ok=True)
    crop_videos_root.mkdir(parents=True, exist_ok=True)
    dwpose_root.mkdir(parents=True, exist_ok=True)

    mp4_files, videos_dir = find_videos(ds_dir)
    if not mp4_files:
        raise FileNotFoundError(f"No .mp4 files found in {ds_dir} or {ds_dir / 'videos'}")
    print(f"Found {len(mp4_files)} video(s) in {videos_dir}")

    for mp4 in mp4_files:
        scene = mp4.stem
        print(f"\n{'='*60}\nProcessing scene: {scene}\n{'='*60}")
        dst_crops_scene_dir = crop_root / scene
        if args.skip_existing and dst_crops_scene_dir.exists() and any(dst_crops_scene_dir.iterdir()):
            print(f"Skipping {scene}, crops exist and --skip-existing is set")
            continue

        if args.dry_run:
            print(f"DRY-RUN: Would run segmentation on {mp4}")
            continue

        # Step 1: Run SAM3 segmentation
        cmd = ["python", str(A1_SCRIPT),
               "--video_path", str(mp4),
               "--model_path", args.model_path]
        run_cmd(cmd)

        # Step 2: Copy masked crops from latest experiment
        exp = find_latest_experiment()
        if exp is None:
            print("No experiment dir found after segmentation. Skipping scene.")
            continue
        print(f"Copying masked crops from {exp}")
        copy_masked_crops(exp, dst_crops_scene_dir)

        # Step 3: Encode crops to H.265 videos
        print(f"Encoding crops to videos for scene {scene}")
        encode_crops_to_video(dst_crops_scene_dir, mp4, crop_videos_root / scene)

    # Step 4: Run DWpose extraction on all crop_videos
    print(f"\n{'='*60}\nDWpose extraction on all crop_videos\n{'='*60}")
    if args.dry_run:
        print("DRY-RUN: Would run DWpose extraction via vid2pose")
    else:
        moore_root = BASE / "Moore-AnimateAnyone"
        cmd = ["bash", "-lc",
               f'cd "{moore_root}" && PYTHONPATH="$(pwd)" '
               f'python tools/vid2pose.py '
               f'--input_dir "{crop_videos_root}" --output_dir "{dwpose_root}"']
        run_cmd(cmd)

    # Step 5: Generate meta.json
    meta_path = ds_dir / "meta.json"
    build_meta_json(crop_videos_root, dwpose_root, meta_path)

    print("\nDone.")
