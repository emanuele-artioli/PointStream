"""Sampled hand/arm removal for a whole egocentric recording.

DiffuEraser needs more than 22 contiguous frames, and on the press a native
30 fps second stays closer to the accepted fill than 1, 6, or 12 fps.
This script keeps one such second every 30 seconds.

The 978s region of clip_01_factory001_worker001_00001 is the worker walking
the aisle. Training frames are the press. That walk is stored and tagged, and
must not be the hold-out for a workstation model. The workstation hold-out is
the other recording of the same press, clip_03_factory001_worker001_00000.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.segmentation import build, load_domain

CURATED = Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated")
DEST_ROOT = Path("/home/itec/emanuele/Datasets/pointstream-demo")
TRAIN_FILL = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/filled-hnerv-factory001/train")
FFMPEG = "/opt/local/bin/ffmpeg"
DIFF_PY = "/home/itec/emanuele/.conda/envs/pointstream-diffueraser/bin/python"
DIFF_DIR = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DiffuEraser")
FPS = 30
EVERY_S = 30.0
BATCH_S = 1.0
# Visually checked: 970s is the press, 990s is the aisle, 1100s is the press again.
WALK = (976.0, 1104.0)
PRESS_TRAIN = (916.0, 976.0)
CLIPS = {
    "clip_01_factory001_worker001_00001": "train",
    "clip_03_factory001_worker001_00000": "workstation_holdout",
    "factory002_worker001_00000": "train",
}
SOURCES = {
    "clip_01_factory001_worker001_00001": CURATED / "clip_01_factory001_worker001_00001.mp4",
    "clip_03_factory001_worker001_00000": CURATED / "clip_03_factory001_worker001_00000.mp4",
    "factory002_worker001_00000": Path("/home/itec/emanuele/Datasets/Egocentric-10K/raw/extracted_f2/factory002_worker001_00000.mp4"),
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def duration(path: Path) -> float:
    probe = subprocess.run(
        [FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(probe.stdout.strip())


def batches(seconds: float) -> list[float]:
    starts = []
    t = 0.0
    while t + BATCH_S <= seconds + 1e-3:
        starts.append(round(t, 3))
        t += EVERY_S
    return starts


def activity(stem: str, start: float) -> str:
    if stem.startswith("clip_01_factory001_worker001_00001") and WALK[0] <= start < WALK[1]:
        return "walk"
    # 210 s is the aisle. 240 s leaves the press for the parts bin.
    # 420 s is blown out, with a second person crossing the frame.
    if stem.startswith("clip_03_factory001_worker001_00000") and int(round(start)) in (210, 240, 420):
        return "drop"
    return "press"


def extract_batch(src: Path, start: float, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if len(list(dest.glob("*.jpg"))) >= FPS:
        return
    run([
        FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", str(src), "-frames:v", str(FPS),
        "-q:v", "2", "-start_number", "0", str(dest / "%05d.jpg"),
    ])


def sam_union(frames: Path, work: Path) -> list[np.ndarray]:
    work.mkdir(parents=True, exist_ok=True)
    cached = work / "union.npy"
    if cached.exists():
        return list(np.load(cached))
    paths = sorted(frames.glob("*.jpg"))
    if len(paths) < 23:
        raise SystemExit(f"{frames} has {len(paths)} frames; DiffuEraser needs more than 22")
    # SAM 3.1 with the egocentric foreground classes (arm, hand); keep the lossless run.
    masks = build("sam31").segment(frames, load_domain("egocentric"))
    masks.save(work)
    union = np.stack([masks.foreground(index) for index in range(len(masks))])
    np.save(cached, union)
    return list(union)


def fill_batch(frames: Path, masks: list[np.ndarray], work: Path) -> list[Path]:
    kept = work / "kept"
    if len(list(kept.glob("*.jpg"))) >= len(masks):
        return sorted(kept.glob("*.jpg"))
    clip = work / "clip.mp4"
    mclip = work / "mask.mp4"
    if not clip.exists():
        run([FFMPEG, "-y", "-framerate", str(FPS), "-i", str(frames / "%05d.jpg"), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)])
    if not mclip.exists():
        mask_dir = work / "mask_png"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(masks):
            cv2.imwrite(str(mask_dir / f"{i:05d}.png"), mask.astype(np.uint8) * 255)
        run([FFMPEG, "-y", "-framerate", str(FPS), "-i", str(mask_dir / "%05d.png"), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mclip)])
    save = work / "diffueraser"
    result = save / "diffueraser_result.mp4"
    if not result.exists() or result.stat().st_size == 0:
        save.mkdir(parents=True, exist_ok=True)
        run([
            DIFF_PY, "run_diffueraser.py",
            "--input_video", str(clip), "--input_mask", str(mclip),
            "--video_length", "3", "--max_img_size", "1920", "--save_path", str(save),
        ], cwd=DIFF_DIR)
    kept.mkdir(parents=True, exist_ok=True)
    run([FFMPEG, "-y", "-i", str(result), "-q:v", "2", str(kept / "%05d.jpg")])
    return sorted(kept.glob("*.jpg"))


def reuse_train(start: float) -> tuple[list[Path], list[Path], list[Path]] | None:
    lo, hi = PRESS_TRAIN
    if not (lo <= start and start + BATCH_S <= hi):
        return None
    offset = int(round((start - lo) * FPS))
    frames = sorted((TRAIN_FILL / "frames").glob("*.jpg"))
    masks = sorted((TRAIN_FILL / "mask_frames").glob("*.png"))
    filled = sorted((TRAIN_FILL / "diffueraser" / "kept").glob("*.jpg"))
    if offset < 0 or offset + FPS > min(len(frames), len(masks), len(filled)):
        return None
    return frames[offset:offset + FPS], masks[offset:offset + FPS], filled[offset:offset + FPS]


def write_preview(folder: Path, name: str, pattern: str) -> None:
    dest = folder / name
    if dest.exists() and dest.stat().st_size > 0:
        return
    run([
        FFMPEG, "-y", "-framerate", str(FPS), "-i", str(folder / pattern),
        "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(dest),
    ])


def press_starts(stem: str, max_batches: int | None) -> list[float]:
    src = SOURCES[stem]
    starts = batches(duration(src))
    if max_batches is not None:
        starts = starts[:max_batches]
    return [start for start in starts if activity(stem, start) == "press"]


def _run_stage(src: Path, stem: str, starts: list[float], stage: str) -> None:
    pending = list(starts)
    deadline = time.monotonic() + 7 * 3600
    while pending:
        still: list[float] = []
        for start in pending:
            origin = int(round(start * FPS))
            work = DEST_ROOT / stem / "_work" / f"t{origin:06d}"
            work.mkdir(parents=True, exist_ok=True)
            cached = (work / "sam" / "union.npy").exists()
            if stage == "sam":
                if not cached:
                    extract_batch(src, start, work / "frames")
                    sam_union(work / "frames", work / "sam")
                print("sam", stem, start, flush=True)
                continue
            if not cached:
                still.append(start)
                continue
            extract_batch(src, start, work / "frames")
            fill_batch(work / "frames", sam_union(work / "frames", work / "sam"), work)
            print("inpaint", stem, start, flush=True)
        pending = still
        if not pending:
            return
        if time.monotonic() > deadline:
            raise SystemExit(f"masks still missing for {stem}: {pending}")
        print("waiting", stem, len(pending), flush=True)
        time.sleep(60)


def process(stem: str, role: str, max_batches: int | None, stage: str, shard: int, shards: int) -> None:
    src = SOURCES[stem]
    starts = press_starts(stem, max_batches)
    starts = [start for index, start in enumerate(starts) if index % shards == shard]
    if not starts:
        print("no batches", stem, stage, shard, flush=True)
        return
    if stage in {"sam", "inpaint"}:
        _run_stage(src, stem, starts, stage)
        return
    first = int(round(starts[0] * FPS))
    last = int(round((starts[-1] + BATCH_S) * FPS)) - 1
    root = DEST_ROOT / stem / f"f{first:06d}-f{last:06d}"
    original = root / "original"
    masks_dir = root / "masks"
    inpainted = root / "inpainted"
    overlay_seq = root / "_overlay_seq"
    filled_seq = root / "_inpainted_seq"
    for folder in (original, masks_dir, inpainted, overlay_seq, filled_seq):
        folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    seq = 0
    for start in starts:
        kind = activity(stem, start)
        origin = int(round(start * FPS))
        work = DEST_ROOT / stem / "_work" / f"t{origin:06d}"
        work.mkdir(parents=True, exist_ok=True)
        reused = reuse_train(start) if stem.startswith("clip_01") else None
        if reused is None:
            extract_batch(src, start, work / "frames")
            mask_list = sam_union(work / "frames", work / "sam")
            filled_paths = fill_batch(work / "frames", mask_list, work)
            frame_paths = sorted((work / "frames").glob("*.jpg"))
            mask_paths = []
            for i, mask in enumerate(mask_list):
                dest = work / "masks" / f"{i:05d}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), mask.astype(np.uint8) * 255)
                mask_paths.append(dest)
        else:
            frame_paths, mask_paths, filled_paths = reused
        for i, (frame, mask, filled) in enumerate(zip(frame_paths, mask_paths, filled_paths)):
            name = f"{origin + i:06d}"
            shutil.copyfile(frame, original / f"{name}.jpg")
            shutil.copyfile(mask, masks_dir / f"{name}.png")
            shutil.copyfile(filled, inpainted / f"{name}.jpg")
            image = cv2.imread(str(frame))
            alpha = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if image is not None and alpha is not None:
                if alpha.shape[:2] != image.shape[:2]:
                    alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                vis = image.copy()
                sel = alpha > 8
                vis[sel] = (0.45 * vis[sel] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
                cv2.imwrite(str(overlay_seq / f"{seq:05d}.jpg"), vis)
            shutil.copyfile(filled, filled_seq / f"{seq:05d}.jpg")
            seq += 1
        manifest.append({
            "start_s": start,
            "frame": origin,
            "frames": FPS,
            "activity": kind,
            "role": "exclude_from_workstation" if kind == "walk" else role,
            "reused_press_fill": reused is not None,
        })
        print("batch", stem, start, kind, "reused" if reused else "filled", flush=True)
    (root / "batches.json").write_text(json.dumps({
        "source": str(src),
        "role": role,
        "recipe": "1s at 30fps every 30s",
        "mask": "src.segmentation sam31, egocentric foreground (arm + hand)",
        "filler": "diffueraser",
        "workstation_holdout": "clip_03_factory001_worker001_00000",
        "rejected_holdout": {
            "source": "clip_01_factory001_worker001_00001",
            "start_s": 978.0,
            "end_s": 988.0,
            "reason": "The worker leaves the press and walks the aisle. Training frames are the press, so this window is not a workstation hold-out.",
        },
        "walk_span_s": list(WALK),
        "batches": manifest,
    }, indent=2))
    write_preview(root, "mask_overlay.mp4", "_overlay_seq/%05d.jpg")
    write_preview(root, "inpainted.mp4", "_inpainted_seq/%05d.jpg")
    shutil.rmtree(overlay_seq, ignore_errors=True)
    shutil.rmtree(filled_seq, ignore_errors=True)
    print("WROTE", root, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", choices=sorted(CLIPS))
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--stage", choices=("all", "sam", "inpaint", "assemble"), default="all")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must be in range(--shards)")
    stems = [args.stem] if args.stem else list(CLIPS)
    for stem in stems:
        process(stem, CLIPS[stem], args.max_batches, args.stage, args.shard, args.shards)


if __name__ == "__main__":
    main()
