#!/usr/bin/env python3
"""
PointStream Server – SAM3 Segmentation + DWPose Keypoint Extraction.

Processes a tennis video into compact pose data that the client can animate:
  1. Segment players from each frame using SAM3 (delegates to A1_segment_with_sam.py).
  2. Extract whole-body DWPose keypoints (body + hands + face) from each crop.
  3. Save a keypoints CSV and a reference image per player.

Usage:
    conda activate pointstream
    python pointstream_server.py --video_path /path/to/video.mp4

Output structure:
    <experiment_dir>/
        masked_crops/id{N}/XXXXX.png   512x512 segmented crops
        reference/id{N}.png            first frame per player (for client)
        dwpose_keypoints.csv           per-frame per-player 134-keypoint data
        tracking_metadata.csv          SAM tracking bounding boxes
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

POINTSTREAM_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = POINTSTREAM_DIR / "experiments"

# Ensure the pointstream directory is importable (for dwpose module)
if str(POINTSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(POINTSTREAM_DIR))


# ---------------------------------------------------------------------------
# Step 1 – SAM3 segmentation (thin wrapper around the existing A1 script)
# ---------------------------------------------------------------------------

def run_sam_segmentation(video_path, model_path):
    """
    Run A1_segment_with_sam.py and return the path of the created experiment directory.
    """
    a1_script = str(POINTSTREAM_DIR / "A1_segment_with_sam.py")
    cmd = [sys.executable, a1_script,
           "--video_path", str(video_path),
           "--model_path", str(model_path)]

    print(f"\n{'='*80}")
    print(f"Step 1: SAM3 segmentation")
    print(f"  Video : {video_path}")
    print(f"  Model : {model_path}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError("SAM3 segmentation failed (see output above)")

    # Find the latest experiment directory
    experiment_dirs = sorted(
        [d for d in EXPERIMENTS_DIR.iterdir()
         if d.is_dir() and d.name.endswith("_sam_seg")],
        key=lambda p: p.stat().st_mtime,
    )
    if not experiment_dirs:
        raise FileNotFoundError("No experiment directory found after SAM3 segmentation")

    exp_dir = experiment_dirs[-1]
    print(f"  Experiment directory: {exp_dir}")
    return exp_dir


# ---------------------------------------------------------------------------
# Step 2 – DWPose keypoint extraction
# ---------------------------------------------------------------------------

def extract_dwpose_keypoints(experiment_dir, det_model=None, pose_model=None):
    """
    Run DWPose (ONNX) on every masked crop produced by SAM3 and save:
      - dwpose_keypoints.csv   (frame_index, player_id, 134 keypoints + scores)
      - reference/id{N}.png    (first crop per player for the client)

    Args:
        experiment_dir: path containing masked_crops/ and tracking_metadata.csv.
        det_model: path to YOLOX ONNX model (default: Moore-AnimateAnyone weights).
        pose_model: path to DW-LL pose ONNX model (default: Moore-AnimateAnyone weights).
    """
    from dwpose import Wholebody, extract_best_person

    experiment_dir = Path(experiment_dir)
    csv_path = experiment_dir / "tracking_metadata.csv"
    masked_crops_dir = experiment_dir / "masked_crops"
    reference_dir = experiment_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"tracking_metadata.csv not found in {experiment_dir}")

    df = pd.read_csv(csv_path)
    people_df = df[df["class_id"] == 0].copy()
    print(f"\n{'='*80}")
    print(f"Step 2: DWPose keypoint extraction")
    print(f"  Crops     : {masked_crops_dir}")
    print(f"  Detections: {len(people_df)} person frames out of {len(df)} total")
    print(f"{'='*80}\n")

    wb = Wholebody(device="cuda:0", det_model_path=det_model, pose_model_path=pose_model)

    pose_results = []
    reference_saved = set()

    for _, row in tqdm(people_df.iterrows(), total=len(people_df), desc="DWPose extraction"):
        frame_idx = int(row["frame_index"])
        det_id = int(row["id"])

        crop_path = masked_crops_dir / f"id{det_id}" / f"{frame_idx:05d}.png"
        if not crop_path.exists():
            continue

        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        H, W = crop.shape[:2]

        # Run whole-body pose estimation
        keypoints, scores = wb(crop)  # (N, 134, 2), (N, 134)
        best_kpts, best_scores = extract_best_person(keypoints, scores)

        if best_kpts is None:
            continue

        pose_results.append({
            "frame_index": frame_idx,
            "player_id": det_id,
            "keypoints": json.dumps(best_kpts.tolist()),
            "scores": json.dumps(best_scores.tolist()),
            "detect_width": W,
            "detect_height": H,
        })

        # Save the very first crop as the reference image for this player
        if det_id not in reference_saved:
            ref_path = reference_dir / f"id{det_id}.png"
            cv2.imwrite(str(ref_path), crop)
            reference_saved.add(det_id)
            print(f"  Saved reference image: {ref_path}")

    if not pose_results:
        print("  No keypoints extracted – check crops / DWPose models.")
        return None

    out_csv = experiment_dir / "dwpose_keypoints.csv"
    pd.DataFrame(pose_results).to_csv(out_csv, index=False)
    print(f"\n  Saved {len(pose_results)} keypoint rows to {out_csv}")
    print(f"  Reference images in {reference_dir}")
    return out_csv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PointStream Server – segment players + extract DWPose keypoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video_path", type=str, default="",
                        help="Input tennis video (required unless --experiment_dir is given)")
    parser.add_argument("--sam_model", type=str,
                        default="/home/itec/emanuele/models/sam3.pt",
                        help="Path to SAM3 model")
    parser.add_argument("--dwpose_det_model", type=str, default=None,
                        help="Path to YOLOX ONNX detection model "
                             "(default: Moore-AnimateAnyone/pretrained_weights/DWPose/yolox_l.onnx)")
    parser.add_argument("--dwpose_pose_model", type=str, default=None,
                        help="Path to DW-LL pose ONNX model "
                             "(default: Moore-AnimateAnyone/pretrained_weights/DWPose/dw-ll_ucoco_384.onnx)")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Skip SAM3 and run DWPose on an existing experiment directory")
    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"#  PointStream Server")
    print(f"{'#'*80}")

    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        if not args.video_path:
            parser.error("--video_path is required unless --experiment_dir is given")
        exp_dir = run_sam_segmentation(args.video_path, args.sam_model)

    out_csv = extract_dwpose_keypoints(
        exp_dir,
        det_model=args.dwpose_det_model,
        pose_model=args.dwpose_pose_model,
    )

    print(f"\n{'='*80}")
    print(f"Server finished.")
    if out_csv:
        print(f"  Experiment : {exp_dir}")
        print(f"  Keypoints  : {out_csv}")
        print(f"  Reference  : {exp_dir / 'reference'}")
        print(f"\nNext step – run the client:")
        print(f"  conda activate animate-anyone")
        print(f"  cd /home/itec/emanuele/Moore-AnimateAnyone")
        print(f"  python {POINTSTREAM_DIR}/pointstream_client.py \\")
        print(f"      --experiment_dir {exp_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
