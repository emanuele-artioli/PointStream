#!/usr/bin/env python3
"""
Aggregate experiment evaluation metrics for PointStream.

For each experiment directory, this script collects:
- configuration parameters (server/client metadata + inferred constants)
- time taken by each sub-task (SAM, DWPose, skeleton reconstruction, inference)
- output quality measured in PSNR

Usage:
    python evaluate_experiments.py
    python evaluate_experiments.py --experiments_root /path/to/experiments
    python evaluate_experiments.py --max_frames 120
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

POINTSTREAM_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENTS_ROOT = POINTSTREAM_DIR / "experiments"


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_constants_from_merged_filename(experiment_dir: Path) -> Dict[str, Any]:
    # prefer merged metadata inside `foreground/` if available
    fg_dir = experiment_dir / "foreground"
    merged_files = []
    if fg_dir.exists():
        merged_files = sorted(fg_dir.glob("merged_metadata*.csv"))
    if not merged_files:
        merged_files = sorted(experiment_dir.glob("merged_metadata*.csv"))
    if not merged_files:
        return {"detect_width": None, "detect_height": None, "source_fps": None, "merged_metadata": None}

    merged = merged_files[0]
    stem = merged.stem
    out = {
        "detect_width": None,
        "detect_height": None,
        "source_fps": None,
        "merged_metadata": str(merged),
    }

    match_wh = re.search(r"w(\d+)_h(\d+)", stem)
    if match_wh:
        out["detect_width"] = int(match_wh.group(1))
        out["detect_height"] = int(match_wh.group(2))

    match_fps = re.search(r"fps_([0-9]+\.?[0-9]*)", stem)
    if match_fps:
        try:
            out["source_fps"] = float(match_fps.group(1))
        except Exception:
            out["source_fps"] = None

    return out


def _extract_player_id_from_name(path: Path) -> Optional[int]:
    match = re.search(r"output_player_(\d+)\.mp4$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _extract_generated_panel(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h % 3 == 0:
        row_h = h // 3
        if row_h >= 128 and row_h <= max(2048, 2 * w):
            return frame[2 * row_h : 3 * row_h, :, :]
    return frame


def _compute_psnr_for_player(experiment_dir: Path, player_id: int, max_frames: Optional[int] = None) -> Dict[str, Any]:
    output_video = experiment_dir / f"output_player_{player_id}.mp4"
    # masked crops / generated outputs live under foreground/ in the new layout
    fg_dir = experiment_dir / "foreground"
    output_video = (fg_dir / f"output_player_{player_id}.mp4") if fg_dir.exists() else (experiment_dir / f"output_player_{player_id}.mp4")
    masked_dir = (fg_dir / "masked_crops" / f"id{player_id}") if fg_dir.exists() else (experiment_dir / "masked_crops" / f"id{player_id}")

    if not output_video.exists() or not masked_dir.exists():
        return {
            "player_id": player_id,
            "psnr_mean": None,
            "psnr_std": None,
            "num_frames": 0,
            "note": "missing output video or masked crops",
        }

    gt_frames = sorted(masked_dir.glob("*.png"))
    if not gt_frames:
        return {
            "player_id": player_id,
            "psnr_mean": None,
            "psnr_std": None,
            "num_frames": 0,
            "note": "no masked crop frames",
        }

    cap = cv2.VideoCapture(str(output_video))
    if not cap.isOpened():
        return {
            "player_id": player_id,
            "psnr_mean": None,
            "psnr_std": None,
            "num_frames": 0,
            "note": "failed to open output video",
        }

    psnr_values: List[float] = []
    frame_idx = 0

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, pred_frame = cap.read()
            if not ret:
                break
            if frame_idx >= len(gt_frames):
                break

            pred_frame = _extract_generated_panel(pred_frame)
            gt_frame = cv2.imread(str(gt_frames[frame_idx]))
            if gt_frame is None:
                frame_idx += 1
                continue

            if pred_frame.shape[:2] != gt_frame.shape[:2]:
                gt_frame = cv2.resize(gt_frame, (pred_frame.shape[1], pred_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            psnr = cv2.PSNR(pred_frame, gt_frame)
            if np.isfinite(psnr):
                psnr_values.append(float(psnr))

            frame_idx += 1
    finally:
        cap.release()

    if not psnr_values:
        return {
            "player_id": player_id,
            "psnr_mean": None,
            "psnr_std": None,
            "num_frames": 0,
            "note": "no valid frame pairs",
        }

    return {
        "player_id": player_id,
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "num_frames": int(len(psnr_values)),
        "note": None,
    }


def _compute_psnr_for_experiment(experiment_dir: Path, max_frames: Optional[int] = None) -> Dict[str, Any]:
    # look for generated outputs in foreground/ first, then fallback to root
    fg_dir = experiment_dir / "foreground"
    outputs = sorted((fg_dir).glob("output_player_*.mp4")) if fg_dir.exists() else sorted(experiment_dir.glob("output_player_*.mp4"))
    player_ids = [pid for pid in (_extract_player_id_from_name(p) for p in outputs) if pid is not None]

    if not player_ids:
        return {
            "psnr_mean": None,
            "psnr_std": None,
            "psnr_num_frames": 0,
            "psnr_by_player": {},
        }

    per_player = {}
    weighted_values = []
    weighted_counts = []

    for player_id in sorted(set(player_ids)):
        metrics = _compute_psnr_for_player(experiment_dir, player_id, max_frames=max_frames)
        per_player[str(player_id)] = metrics
        if metrics["psnr_mean"] is not None and metrics["num_frames"] > 0:
            weighted_values.append(metrics["psnr_mean"] * metrics["num_frames"])
            weighted_counts.append(metrics["num_frames"])

    if not weighted_counts:
        return {
            "psnr_mean": None,
            "psnr_std": None,
            "psnr_num_frames": 0,
            "psnr_by_player": per_player,
        }

    total_frames = int(np.sum(weighted_counts))
    weighted_mean = float(np.sum(weighted_values) / total_frames)

    player_means = [m["psnr_mean"] for m in per_player.values() if m["psnr_mean"] is not None]
    std_across_players = float(np.std(player_means)) if player_means else None

    return {
        "psnr_mean": weighted_mean,
        "psnr_std": std_across_players,
        "psnr_num_frames": total_frames,
        "psnr_by_player": per_player,
    }


def _discover_experiments(root: Path) -> List[Path]:
    if not root.exists():
        return []
    experiments = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        # accept experiments that either use the new `foreground/` layout or the legacy root layout
        fg_child = child / "foreground"
        if (fg_child.exists() and ((fg_child / "dwpose_keypoints.csv").exists() or (fg_child / "tracking_metadata.csv").exists())):
            experiments.append(child)
            continue
        if (child / "dwpose_keypoints.csv").exists() or (child / "tracking_metadata.csv").exists():
            experiments.append(child)
    return experiments


def _flatten_record(experiment_dir: Path, max_frames: Optional[int] = None) -> Dict[str, Any]:
    # prefer evaluation JSONs inside foreground/ when present
    fg_dir = experiment_dir / "foreground"
    server_eval = _safe_load_json((fg_dir / "evaluation_server.json") if fg_dir.exists() else (experiment_dir / "evaluation_server.json"))
    client_eval = _safe_load_json((fg_dir / "evaluation_client.json") if fg_dir.exists() else (experiment_dir / "evaluation_client.json"))
    inferred = _infer_constants_from_merged_filename(experiment_dir)
    psnr = _compute_psnr_for_experiment(experiment_dir, max_frames=max_frames)

    server_timings = (server_eval or {}).get("timings", {})
    client_timings = (client_eval or {}).get("timings", {})

    row = {
        "experiment": experiment_dir.name,
        "experiment_dir": str(experiment_dir),
        "detect_width": inferred["detect_width"],
        "detect_height": inferred["detect_height"],
        "source_fps": inferred["source_fps"],
        "merged_metadata": inferred["merged_metadata"],
        "server_sam_segmentation_sec": server_timings.get("sam_segmentation_sec"),
        "server_dwpose_extraction_sec": server_timings.get("dwpose_extraction_sec"),
        "server_total_sec": server_timings.get("server_total_sec"),
        "client_skeleton_reconstruction_sec": client_timings.get("skeleton_reconstruction_sec"),
        "client_inference_sec": client_timings.get("inference_sec"),
        "client_total_sec": client_timings.get("client_total_sec"),
        "pipeline_total_sec": None,
        "psnr_mean": psnr["psnr_mean"],
        "psnr_std": psnr["psnr_std"],
        "psnr_num_frames": psnr["psnr_num_frames"],
        "server_config_json": json.dumps((server_eval or {}).get("config", {}), ensure_ascii=False),
        "client_config_json": json.dumps((client_eval or {}).get("config", {}), ensure_ascii=False),
        "per_player_timings_json": json.dumps(client_timings.get("per_player", {}), ensure_ascii=False),
        "psnr_by_player_json": json.dumps(psnr["psnr_by_player"], ensure_ascii=False),
    }

    numeric_total = [
        row["server_total_sec"],
        row["client_total_sec"],
    ]
    if all(v is not None for v in numeric_total):
        row["pipeline_total_sec"] = round(float(row["server_total_sec"]) + float(row["client_total_sec"]), 3)

    return row


def evaluate_all(experiments_root: Path, max_frames: Optional[int] = None) -> pd.DataFrame:
    experiments = _discover_experiments(experiments_root)
    records = [_flatten_record(exp, max_frames=max_frames) for exp in experiments]
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PointStream experiments with timing + PSNR")
    parser.add_argument("--experiments_root", type=str, default=str(DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to output CSV (default: <experiments_root>/evaluation_summary.csv)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to write records as JSON")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Optional cap of frames per player for PSNR computation")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else (experiments_root / "evaluation_summary.csv")

    df = evaluate_all(experiments_root, max_frames=args.max_frames)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)

    print(f"Evaluated {len(df)} experiments")
    print(f"CSV summary: {output_csv}")
    if args.output_json:
        print(f"JSON summary: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
