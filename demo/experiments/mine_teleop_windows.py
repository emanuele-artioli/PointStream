"""Rank 10s windows by large hands and slow/simple background.

PointStream saves bitrate when the background is cheap (slow/simple) and hands
cover a large fraction of the frame (those pixels come from the 47-byte wire
instead of AV1).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FFMPEG = "/opt/local/bin/ffmpeg"


def _hand_mask(frame: np.ndarray, poses_hands: list, scale: float) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for hand in poses_hands:
        x1, y1, x2, y2 = hand.bbox
        x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    if mask.any():
        k = max(5, int(0.04 * min(h, w)) | 1)
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return mask


def score_video(path: Path, sample_stride: int, max_frames: int | None, width: int) -> list[dict[str, Any]]:
    from demo.pipeline.hand_keypoints import HandPoseEstimator

    cap = cv2.VideoCapture(str(path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    height = max(1, int(src_h * (width / float(src_w))))
    estimator = HandPoseEstimator(static_image_mode=False, model_complexity=0, min_detection_confidence=0.4)

    records: list[dict[str, Any]] = []
    prev_gray = None
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_stride != 0:
            idx += 1
            continue
        small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        full = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA) if (src_w, src_h) != (1920, 1080) else frame
        pose = estimator.process_frame(full, frame_idx=idx)
        mask = _hand_mask(small, pose.hands, width / 1920.0)
        fg_frac = float(mask.mean() / 255.0)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        bg_flow = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            bg = mag[mask == 0] if (mask == 0).any() else mag.reshape(-1)
            bg_flow = float(np.median(bg)) if bg.size else 0.0
        prev_gray = gray
        records.append(
            {
                "frame": idx,
                "fg_frac": fg_frac,
                "bg_flow": bg_flow,
                "n_hands": len(pose.hands),
            }
        )
        if len(records) % 40 == 0:
            logger.info("%s sampled %s frames (src frame %s/%s)", path.name, len(records), idx, n_total)
        idx += 1
    cap.release()
    estimator.close()
    return records


def window_scores(records: list[dict[str, Any]], window_frames: int, sample_stride: int) -> list[dict[str, Any]]:
    if not records:
        return []
    span = max(1, window_frames // sample_stride)
    out: list[dict[str, Any]] = []
    for i in range(0, max(1, len(records) - span + 1), max(1, span // 3)):
        chunk = records[i : i + span]
        if len(chunk) < max(4, span // 2):
            continue
        fg = float(np.mean([r["fg_frac"] for r in chunk]))
        flow = float(np.mean([r["bg_flow"] for r in chunk]))
        presence = float(np.mean([1.0 if r["n_hands"] else 0.0 for r in chunk]))
        score = (fg * (0.25 + presence)) / (0.35 + flow)
        out.append(
            {
                "start_frame": int(chunk[0]["frame"]),
                "end_frame": int(chunk[-1]["frame"]),
                "fg_frac": round(fg, 4),
                "bg_flow": round(flow, 4),
                "presence": round(presence, 4),
                "score": round(score, 5),
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def cut_window(src: Path, dest: Path, start_frame: int, n_frames: int, fps: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ss = start_frame / max(fps, 1.0)
    cmd = [
        FFMPEG, "-y", "-ss", f"{ss:.3f}", "-i", str(src),
        "-frames:v", str(n_frames),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--curated-dir", type=Path, default=Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated_v2"))
    parser.add_argument("--window-frames", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=9000)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--top-per-video", type=int, default=2)
    parser.add_argument("--cut-top", type=int, default=8)
    args = parser.parse_args()

    ranked: list[dict[str, Any]] = []
    for video in args.videos:
        logger.info("scoring %s", video)
        recs = score_video(video, args.sample_stride, args.max_frames, args.width)
        wins = window_scores(recs, args.window_frames, args.sample_stride)[: args.top_per_video]
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()
        for w in wins:
            ranked.append({"video": str(video), "fps": fps, **w})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"windows": ranked}, indent=2))
    logger.info("wrote %s (%s windows)", args.out, len(ranked))

    cuts = []
    for i, item in enumerate(ranked[: args.cut_top], start=1):
        dest = args.curated_dir / f"cand_{i:02d}_{Path(item['video']).stem}_f{item['start_frame']}.mp4"
        logger.info("cut %s score=%.4f fg=%.3f flow=%.3f presence=%.3f", dest.name, item["score"], item["fg_frac"], item["bg_flow"], item["presence"])
        cut_window(Path(item["video"]), dest, item["start_frame"], args.window_frames, item["fps"])
        cuts.append({**item, "path": str(dest), "rank": i, "filename": dest.name})
    manifest = args.curated_dir / "candidates.json"
    args.curated_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(cuts, indent=2))
    logger.info("candidates manifest %s", manifest)


if __name__ == "__main__":
    main()
