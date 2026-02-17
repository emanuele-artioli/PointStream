#!/usr/bin/env python3
"""
Background Server – build a panorama model for an experiment.

Takes an existing experiment folder and input video, estimates frame-to-frame
homographies, stitches a panorama, and saves reconstruction intrinsics/metadata.

Outputs written into <experiment_dir>/:
  - background_panorama.png
  - background_intrinsics.json
  - evaluation_background_server.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import subprocess
import shutil
import os


def _read_video_frames(video_path: Path, skip_frames: int = 1) -> Tuple[List[np.ndarray], List[int], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[np.ndarray] = []
    frame_indices: List[int] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % skip_frames == 0:
            frames.append(frame)
            frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    return frames, frame_indices, fps, total_frames


def _estimate_homography_orb(
    detector: cv2.ORB,
    matcher: cv2.BFMatcher,
    src_img: np.ndarray,
    dst_img: np.ndarray,
) -> np.ndarray:
    kp_src, des_src = detector.detectAndCompute(src_img, None)
    kp_dst, des_dst = detector.detectAndCompute(dst_img, None)

    if des_src is None or des_dst is None:
        return np.eye(3, dtype=np.float64)

    matches = matcher.match(des_src, des_dst)
    if len(matches) < 4:
        return np.eye(3, dtype=np.float64)

    matches = sorted(matches, key=lambda m: m.distance)
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return np.eye(3, dtype=np.float64)
    return H.astype(np.float64)


def _compute_global_homographies(frames: List[np.ndarray], nfeatures: int) -> List[np.ndarray]:
    detector = cv2.ORB_create(nfeatures=nfeatures)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    global_h = [np.eye(3, dtype=np.float64)]
    current_h = np.eye(3, dtype=np.float64)

    for idx in range(len(frames) - 1):
        h_rel = _estimate_homography_orb(detector, matcher, frames[idx + 1], frames[idx])
        current_h = current_h @ h_rel
        global_h.append(current_h.copy())

    return global_h


def _compute_canvas(frame_shape: Tuple[int, int, int], global_h: List[np.ndarray]) -> Tuple[np.ndarray, int, int, Tuple[float, float]]:
    h, w = frame_shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

    all_warped = []
    for hmat in global_h:
        warped = cv2.perspectiveTransform(corners, hmat)
        all_warped.append(warped)

    all_warped = np.concatenate(all_warped, axis=0)
    xmin, ymin = all_warped.min(axis=0).ravel()
    xmax, ymax = all_warped.max(axis=0).ravel()

    tx = -float(xmin)
    ty = -float(ymin)
    h_translation = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)

    canvas_w = int(round(float(xmax - xmin)))
    canvas_h = int(round(float(ymax - ymin)))
    return h_translation, canvas_w, canvas_h, (tx, ty)


def _stitch_panorama(
    frames: List[np.ndarray],
    global_h: List[np.ndarray],
    h_translation: np.ndarray,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        h_final = h_translation @ global_h[idx]
        warped = cv2.warpPerspective(frame, h_final, (canvas_w, canvas_h))

        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        panorama = cv2.bitwise_and(panorama, panorama, mask=cv2.bitwise_not(mask))
        panorama = cv2.add(panorama, warped)

    return panorama


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create background panorama + intrinsics in an experiment folder")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Existing experiment directory")
    parser.add_argument("--video_path", type=str, required=True, help="Input source video")
    parser.add_argument("--skip_frames", type=int, default=1, help="Use every N-th frame for panorama (default: 1)")
    parser.add_argument("--nfeatures", type=int, default=1000, help="ORB feature count (default: 1000)")
    parser.add_argument("--panorama_name", type=str, default="background_panorama.png", help="Panorama filename")
    parser.add_argument("--intrinsics_name", type=str, default="background_intrinsics.json", help="Intrinsics/metadata filename")
    args = parser.parse_args()

    if args.skip_frames <= 0:
        raise ValueError("--skip_frames must be >= 1")

    exp_dir = Path(args.experiment_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("\n" + "#" * 80)
    print("#  Background Server")
    print("#" * 80)
    print(f"Experiment : {exp_dir}")
    print(f"Video      : {video_path}")

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    frames, frame_indices, source_fps, source_frame_count = _read_video_frames(video_path, skip_frames=args.skip_frames)
    t_read = time.perf_counter() - t0

    if not frames:
        raise RuntimeError("No frames extracted from input video")

    print(f"Extracted {len(frames)} frames (skip_frames={args.skip_frames})")

    t0 = time.perf_counter()
    global_h = _compute_global_homographies(frames, nfeatures=args.nfeatures)
    t_align = time.perf_counter() - t0

    t0 = time.perf_counter()
    h_translation, canvas_w, canvas_h, (tx, ty) = _compute_canvas(frames[0].shape, global_h)
    panorama = _stitch_panorama(frames, global_h, h_translation, canvas_w, canvas_h)
    t_stitch = time.perf_counter() - t0

    # store background artifacts in `background/` subfolder
    bg_dir = exp_dir / "background"
    bg_dir.mkdir(parents=True, exist_ok=True)
    pano_path = bg_dir / args.panorama_name
    intrinsics_path = bg_dir / args.intrinsics_name

    # Heuristics to detect a failed panorama (e.g. camera moving too fast / poor alignment)
    frame_h, frame_w = frames[0].shape[:2]
    pano_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    coverage = float(np.count_nonzero(pano_gray)) / float(max(1, canvas_w * canvas_h))

    # identity-ratio: how many relative homographies are exactly identity (returned when matching failed)
    identity_count = 0
    for i in range(max(0, len(global_h) - 1)):
        rel = np.linalg.inv(global_h[i]) @ global_h[i + 1]
        if np.allclose(rel, np.eye(3), atol=1e-8):
            identity_count += 1
    identity_ratio = float(identity_count) / float(max(1, len(global_h) - 1)) if len(global_h) > 1 else 0.0

    too_large_canvas = (canvas_w > frame_w * 6) or (canvas_h > frame_h * 6)
    low_coverage = coverage < 0.02
    bad_alignment = identity_ratio > 0.4

    panorama_failed = too_large_canvas or low_coverage or bad_alignment

    if panorama_failed:
        print("Panorama creation flagged as FAILED — falling back to traditional frame encoding (AV1).")
        fallback_name = "background_fallback_av1.mp4"
        fallback_path = bg_dir / fallback_name

        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe is not None:
            cmd = [
                ffmpeg_exe,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{frame_w}x{frame_h}",
                "-r",
                f"{float(source_fps) if source_fps else 24.0}",
                "-i",
                "-",
                "-c:v",
                "libsvtav1",
                "-crf",
                "30",
                "-b:v",
                "0",
                "-threads",
                "0",
                "-row-mt",
                "1",
                str(fallback_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for frame in frames:
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
                returncode = proc.wait()
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
                if returncode != 0:
                    raise RuntimeError(f"ffmpeg fallback encoding failed: {stderr}")
            finally:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
        else:
            # fallback to OpenCV writer
            writer = cv2.VideoWriter(
                str(fallback_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(source_fps) if source_fps else 24.0,
                (frame_w, frame_h),
            )
            if not writer.isOpened():
                raise RuntimeError("Failed to open fallback video writer")
            for frame in frames:
                writer.write(frame)
            writer.release()

        intrinsics_payload: Dict[str, Any] = {
            "script": "background_server.py",
            "timestamp": datetime.now().isoformat(),
            "video_path": str(video_path),
            "source_video_name": video_path.name,
            "source_fps": float(source_fps) if source_fps else None,
            "source_frame_count": int(source_frame_count),
            "processed_frame_count": int(len(frames)),
            "processed_frame_indices": [int(i) for i in frame_indices],
            "skip_frames": int(args.skip_frames),
            "frame_size": [int(frame_w), int(frame_h)],
            "canvas_size": None,
            "translation": [0.0, 0.0],
            "homographies": [],
            "panorama_failed": True,
            "fallback_video": fallback_name,
        }
        _write_json(intrinsics_path, intrinsics_payload)

        total_sec = time.perf_counter() - t_total
        eval_payload = {
            "script": "background_server.py",
            "timestamp": datetime.now().isoformat(),
            "experiment_dir": str(exp_dir),
            "config": {
                "video_path": str(video_path),
                "skip_frames": int(args.skip_frames),
                "nfeatures": int(args.nfeatures),
                "panorama_name": args.panorama_name,
                "intrinsics_name": args.intrinsics_name,
            },
            "timings": {
                "read_frames_sec": round(t_read, 3),
                "alignment_sec": round(t_align, 3),
                "stitch_sec": round(t_stitch, 3),
                "server_total_sec": round(total_sec, 3),
            },
            "fallback": {
                "panorama_failed": True,
                "fallback_video": fallback_name,
            },
        }
        _write_json(bg_dir / "evaluation_background_server.json", eval_payload)

        print(f"Panorama failed — wrote fallback video: {fallback_path}")
        print(f"Intrinsics saved : {intrinsics_path}")
        print(f"Total time (sec) : {total_sec:.3f}")
        print("#" * 80 + "\n")
        return

    # normal (successful) path
    ok = cv2.imwrite(str(pano_path), panorama)
    if not ok:
        raise RuntimeError(f"Failed to write panorama: {pano_path}")

    intrinsics_payload: Dict[str, Any] = {
        "script": "background_server.py",
        "timestamp": datetime.now().isoformat(),
        "video_path": str(video_path),
        "source_video_name": video_path.name,
        "source_fps": float(source_fps) if source_fps else None,
        "source_frame_count": int(source_frame_count),
        "processed_frame_count": int(len(frames)),
        "processed_frame_indices": [int(i) for i in frame_indices],
        "skip_frames": int(args.skip_frames),
        "frame_size": [int(frame_w), int(frame_h)],
        "canvas_size": [int(canvas_w), int(canvas_h)],
        "translation": [float(tx), float(ty)],
        "homographies": [h.tolist() for h in global_h],
    }
    _write_json(intrinsics_path, intrinsics_payload)

    total_sec = time.perf_counter() - t_total
    eval_payload = {
        "script": "background_server.py",
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": str(exp_dir),
        "config": {
            "video_path": str(video_path),
            "skip_frames": int(args.skip_frames),
            "nfeatures": int(args.nfeatures),
            "panorama_name": args.panorama_name,
            "intrinsics_name": args.intrinsics_name,
        },
        "timings": {
            "read_frames_sec": round(t_read, 3),
            "alignment_sec": round(t_align, 3),
            "stitch_sec": round(t_stitch, 3),
            "server_total_sec": round(total_sec, 3),
        },
    }
    _write_json(bg_dir / "evaluation_background_server.json", eval_payload)

    print(f"Panorama saved   : {pano_path}")
    print(f"Intrinsics saved : {intrinsics_path}")
    print(f"Canvas size      : {canvas_w}x{canvas_h}")
    print(f"Total time (sec) : {total_sec:.3f}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
