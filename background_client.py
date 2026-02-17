#!/usr/bin/env python3
"""
Background Client – reconstruct background video from panorama + intrinsics.

Reads background artifacts from an experiment folder and reconstructs frames via
inverse warping from panorama space back to frame space.

Expected inputs in <experiment_dir>/:
  - background_panorama.png
  - background_intrinsics.json

Outputs in <experiment_dir>/:
  - background_reconstructed.mp4
  - evaluation_background_client.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import subprocess
import shutil


def _load_intrinsics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def reconstruct_background_video(
    panorama_path: Path,
    intrinsics_path: Path,
    output_video_path: Path,
    fps_override: float | None = None,
) -> Dict[str, Any]:
    panorama = cv2.imread(str(panorama_path))
    if panorama is None:
        raise RuntimeError(f"Failed to read panorama image: {panorama_path}")

    data = _load_intrinsics(intrinsics_path)

    frame_w, frame_h = data["frame_size"]
    tx, ty = data["translation"]
    homographies = np.array(data["homographies"], dtype=np.float64)
    source_fps = data.get("source_fps")
    skip_frames = int(data.get("skip_frames", 1))

    out_fps = fps_override
    if out_fps is None:
        if source_fps and source_fps > 0:
            out_fps = float(source_fps) / float(max(skip_frames, 1))
        else:
            out_fps = 24.0

    out_fps = max(float(out_fps), 1.0)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    h_translation = np.array(
        [[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    num_written = 0

    # Prefer ffmpeg + libaom-av1 for output encoding; fall back to OpenCV writer if ffmpeg is not available.
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is not None:
        print("Writing reconstructed background with ffmpeg (AV1 / libsvtav1)...")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(frame_w)}x{int(frame_h)}",
            "-r",
            f"{out_fps}",
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
            str(output_video_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for h_global in homographies:
                h_composite = h_translation @ h_global
                h_inv = np.linalg.inv(h_composite)
                frame = cv2.warpPerspective(panorama, h_inv, (int(frame_w), int(frame_h)))
                # ensure contiguous BGR bytes for ffmpeg stdin
                proc.stdin.write(frame.tobytes())
                num_written += 1
            proc.stdin.close()
            # wait for ffmpeg to finish and capture stderr (avoid calling communicate() after closing stdin)
            returncode = proc.wait()
            err = ""
            if proc.stderr is not None:
                err = proc.stderr.read().decode('utf-8', errors='replace')
            if returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed with returncode={returncode}: {err}"
                )
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
    else:
        # fallback to OpenCV writer (mp4v)
        print("ffmpeg not found in PATH — falling back to OpenCV VideoWriter (mp4v).")
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (int(frame_w), int(frame_h)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_video_path}")

        try:
            for h_global in homographies:
                h_composite = h_translation @ h_global
                h_inv = np.linalg.inv(h_composite)
                frame = cv2.warpPerspective(panorama, h_inv, (int(frame_w), int(frame_h)))
                writer.write(frame)
                num_written += 1
        finally:
            writer.release()

    return {
        "frames_written": int(num_written),
        "fps": float(out_fps),
        "frame_size": [int(frame_w), int(frame_h)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct background video from panorama + intrinsics")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--panorama_name", type=str, default="background_panorama.png", help="Panorama filename")
    parser.add_argument("--intrinsics_name", type=str, default="background_intrinsics.json", help="Intrinsics filename")
    parser.add_argument("--output_name", type=str, default="background_reconstructed.mp4", help="Output video filename")
    parser.add_argument("--fps", type=float, default=None, help="Optional FPS override")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    pano_path = exp_dir / args.panorama_name
    intrinsics_path = exp_dir / args.intrinsics_name
    out_video = exp_dir / args.output_name

    if not pano_path.exists():
        raise FileNotFoundError(f"Panorama not found: {pano_path}")
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")

    print("\n" + "#" * 80)
    print("#  Background Client")
    print("#" * 80)
    print(f"Experiment : {exp_dir}")
    print(f"Panorama   : {pano_path}")
    print(f"Intrinsics : {intrinsics_path}")

    t0 = time.perf_counter()
    recon = reconstruct_background_video(
        panorama_path=pano_path,
        intrinsics_path=intrinsics_path,
        output_video_path=out_video,
        fps_override=args.fps,
    )
    total_sec = time.perf_counter() - t0

    eval_payload = {
        "script": "background_client.py",
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": str(exp_dir),
        "config": {
            "panorama_name": args.panorama_name,
            "intrinsics_name": args.intrinsics_name,
            "output_name": args.output_name,
            "fps_override": args.fps,
        },
        "timings": {
            "reconstruction_sec": round(total_sec, 3),
            "client_total_sec": round(total_sec, 3),
        },
        "output": recon,
    }
    _write_json(exp_dir / "evaluation_background_client.json", eval_payload)

    print(f"Reconstructed video: {out_video}")
    print(f"Frames written     : {recon['frames_written']}")
    print(f"Output FPS         : {recon['fps']:.3f}")
    print(f"Total time (sec)   : {total_sec:.3f}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
