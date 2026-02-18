#!/usr/bin/env python3
"""
PointStream Server – SAM3 Segmentation + DWPose Keypoint Extraction.

Processes a tennis video into compact pose data that the client can animate:
    1. Segment players from each frame using SAM3 (native server implementation).
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
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Runtime compatibility shim: make CLIP/OpenCLIP SimpleTokenizer callable
# as expected by Ultralytics SAM3 semantic predictor.
try:
    from clip import simple_tokenizer as _simple_tokenizer
except Exception:
    _simple_tokenizer = None

try:
    from clip import tokenizer as _clip_tokenizer
except Exception:
    _clip_tokenizer = None

try:
    from open_clip import tokenizer as _open_tokenizer
except Exception:
    _open_tokenizer = None

from ultralytics.models.sam import SAM3SemanticPredictor, SAM3VideoPredictor

POINTSTREAM_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = POINTSTREAM_DIR / "experiments"

# Ensure the pointstream directory is importable (for dwpose module)
if str(POINTSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(POINTSTREAM_DIR))


# ---------------------------------------------------------------------------
# Step 1 – SAM3 segmentation (fully inlined, independent from A1 script)
# ---------------------------------------------------------------------------


def _make_callable_module(module):
    if module is None or not hasattr(module, "SimpleTokenizer"):
        return

    cls = module.SimpleTokenizer
    try:
        tokenizer_instance = cls()
    except Exception:
        tokenizer_instance = None

    if tokenizer_instance is not None and callable(tokenizer_instance):
        return

    def _simple_tokenizer_call(self, text, context_length=77):
        texts = text if isinstance(text, (list, tuple)) else [text]
        token_ids = [self.encode(t) if hasattr(self, "encode") else [] for t in texts]
        import torch

        out = torch.zeros((len(token_ids), context_length), dtype=torch.long)
        for i, tks in enumerate(token_ids):
            tks = tks[:context_length]
            out[i, : len(tks)] = torch.tensor(tks, dtype=torch.long)
            if len(tks) < context_length:
                out[i, len(tks)] = 0
        return out

    cls.__call__ = _simple_tokenizer_call


_make_callable_module(_simple_tokenizer)
_make_callable_module(_clip_tokenizer)
_make_callable_module(_open_tokenizer)


def _resize_and_pad(img, target_size=512):
    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if len(img_resized.shape) == 3:
        padded = np.zeros((target_size, target_size, img_resized.shape[2]), dtype=img_resized.dtype)
    else:
        padded = np.zeros((target_size, target_size), dtype=img_resized.dtype)

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top : top + new_h, left : left + new_w] = img_resized

    transform_info = {
        "orig_h": h,
        "orig_w": w,
        "scale": scale,
        "pad_top": top,
        "pad_left": left,
        "resized_h": new_h,
        "resized_w": new_w,
    }
    return padded, transform_info


def _get_initial_bboxes_from_frame0(frame0_path, model_path):
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        imgsz=1288,
        model=model_path,
        half=True,
        save=False,
        compile=None,
    )
    semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
    semantic_predictor.set_image(frame0_path)

    names = ["tennis player"]
    print(f"Running SAM3 semantic segmentation on frame 0 with text prompts: {names}")
    try:
        frame0_results = semantic_predictor(text=names)
    except Exception as exc:
        raise RuntimeError(
            "SAM3 semantic prediction failed. "
            f"Error: {exc}. "
            "Tokenizer shim has been applied but semantic prompting still failed. "
            "Please verify clip/open-clip package compatibility in the pointstream environment."
        ) from exc

    class_0_detections = []
    if frame0_results and len(frame0_results) > 0:
        for det in frame0_results[0].boxes:
            cls_id = int(det.cls[0].cpu().item())
            if cls_id == 0:
                bbox = det.xyxy[0].cpu().numpy().tolist()
                conf = float(det.conf[0].cpu().item())
                class_0_detections.append({"bbox": bbox, "conf": conf})

    if len(class_0_detections) < 2:
        raise RuntimeError(
            "SAM3 semantic predictor found fewer than 2 tennis players on frame 0. "
            f"Detections found: {len(class_0_detections)}"
        )

    class_0_detections.sort(key=lambda x: x["conf"], reverse=True)
    return [det["bbox"] for det in class_0_detections[:2]]


def run_sam_segmentation(video_path, model_path, experiment_dir=None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if experiment_dir:
        output_dir = Path(experiment_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = EXPERIMENTS_DIR / f"{timestamp}_sam_seg"

    output_dir.mkdir(parents=True, exist_ok=True)
    # store all foreground-related artifacts in a `foreground/` subfolder
    fg_dir = output_dir / "foreground"
    fg_dir.mkdir(parents=True, exist_ok=True)

    masked_crops_dir = fg_dir / "masked_crops"
    masked_crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = fg_dir / "segmentation_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Step 1: SAM3 segmentation")
    print(f"  Video : {video_path}")
    print(f"  Model : {model_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}\n")

    cap = cv2.VideoCapture(str(video_path))
    # Extract FPS (fall back to framecount/duration if unavailable)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if video_fps == 0.0 and frame_count > 0:
        # attempt to compute fps via duration when possible
        video_duration = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 or 0.0)
        if video_duration > 0:
            video_fps = frame_count / video_duration
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to read frame 0 from input video")

    # store FPS for downstream tools
    try:
        video_fps = float(video_fps)
    except Exception:
        video_fps = 0.0

    frame0_path = output_dir / "frame0_temp.png"
    cv2.imwrite(str(frame0_path), frame0)

    try:
        initial_bboxes = _get_initial_bboxes_from_frame0(str(frame0_path), str(model_path))
    finally:
        if frame0_path.exists():
            frame0_path.unlink()

    print("Using top-2 frame-0 SAM3 detections for tracking:")
    for i, bbox in enumerate(initial_bboxes):
        print(f"  Player {i + 1}: {bbox}")

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        imgsz=644,
        model=str(model_path),
        half=True,
        save=False,
        compile=None,
    )
    predictor = SAM3VideoPredictor(overrides=overrides)
    results = predictor(source=str(video_path), bboxes=initial_bboxes, stream=True)

    metadata = []
    for frame_idx, result in enumerate(results):
        frame = result.orig_img
        frame_h, frame_w = frame.shape[:2]
        frame_data = {"frame_index": frame_idx, "detections": []}

        # Store a full-resolution foreground mask (union of all tracked players) for background suppression.
        combined_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

        if result.masks is None or len(result.boxes) == 0:
            mask_path = masks_dir / f"{frame_idx:05d}.png"
            cv2.imwrite(str(mask_path), combined_mask)
            metadata.append(frame_data)
            continue

        for idx, (det, mask) in enumerate(zip(result.boxes, result.masks)):
            cls_id = 0
            det_id = idx
            bbox = det.xyxy[0].cpu().numpy().tolist()
            mask_data = mask.data.cpu().numpy()[0]
            mask_uint8 = (mask_data > 0.5).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, mask_uint8)

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            mask_3ch = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
            masked_frame = cv2.bitwise_and(frame, mask_3ch)
            masked_crop = masked_frame[y1:y2, x1:x2]

            if masked_crop is not None and masked_crop.size > 0:
                masked_crop_padded, transform_info = _resize_and_pad(masked_crop, target_size=512)
                id_subfolder = masked_crops_dir / f"id{det_id}"
                id_subfolder.mkdir(parents=True, exist_ok=True)
                crop_path = id_subfolder / f"{frame_idx:05d}.png"
                cv2.imwrite(str(crop_path), masked_crop_padded)
            else:
                transform_info = None

            frame_data["detections"].append(
                {
                    "id": det_id,
                    "class_id": cls_id,
                    "bbox": bbox,
                    "transform_info": transform_info,
                }
            )

        mask_path = masks_dir / f"{frame_idx:05d}.png"
        cv2.imwrite(str(mask_path), combined_mask)
        metadata.append(frame_data)

    metadata_frame = pd.DataFrame(metadata)
    metadata_frame = metadata_frame.explode("detections").reset_index(drop=True)
    detections_expanded = metadata_frame["detections"].apply(pd.Series)
    metadata_frame = pd.concat([metadata_frame.drop("detections", axis=1), detections_expanded], axis=1)

    if "transform_info" in metadata_frame.columns:
        transform_expanded = metadata_frame["transform_info"].apply(
            lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series()
        )
        transform_expanded.columns = [f"transform_{col}" for col in transform_expanded.columns]
        metadata_frame = pd.concat([metadata_frame.drop("transform_info", axis=1), transform_expanded], axis=1)

    # add video FPS to tracking metadata so downstream tools (client) can reuse it
    metadata_frame["video_fps"] = video_fps

    metadata_csv_path = fg_dir / "tracking_metadata.csv"
    metadata_frame.to_csv(metadata_csv_path, index=False)
    print(f"Tracking metadata saved to: {metadata_csv_path} (video_fps={video_fps})")
    print(f"Full-resolution segmentation masks saved to: {masks_dir}")
    # return experiment root (output_dir) — foreground artifacts are under `foreground/`
    return output_dir


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
        det_model: path to YOLOX ONNX model (default: /home/itec/emanuele/Models/DWPose/yolox_l.onnx).
        pose_model: path to DW-LL pose ONNX model (default: /home/itec/emanuele/Models/DWPose/dw-ll_ucoco_384.onnx).
    """
    from dwpose import Wholebody, extract_best_person

    experiment_dir = Path(experiment_dir)
    csv_path = experiment_dir / "tracking_metadata.csv"
    masked_crops_dir = experiment_dir / "masked_crops"
    # prefer foreground/ layout, otherwise fall back to root experiment dir
    fg_dir = experiment_dir / "foreground"
    if fg_dir.exists():
        csv_path = fg_dir / "tracking_metadata.csv"
        masked_crops_dir = fg_dir / "masked_crops"
        reference_dir = fg_dir / "reference"
    else:
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

    # write DWpose CSV into foreground/ when present, otherwise root
    fg_dir = experiment_dir / "foreground"
    if fg_dir.exists():
        out_csv = fg_dir / "dwpose_keypoints.csv"
    else:
        out_csv = experiment_dir / "dwpose_keypoints.csv"
    pose_df = pd.DataFrame(pose_results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pose_df.to_csv(out_csv, index=False)
    print(f"\n  Saved {len(pose_results)} keypoint rows to {out_csv}")
    print(f"  Reference images in {reference_dir}")

    # --- Merge tracking metadata and pose results into a single metadata file ---
    tracking_csv = csv_path
    if tracking_csv.exists():
        track_df = pd.read_csv(tracking_csv)

        # Round keypoints to nearest integer and round scores to 3 decimals
        # for the merged file (keep original CSV unchanged)
        def _round_kpts_json(jstr):
            try:
                arr = json.loads(jstr)
                arr_rounded = [[int(round(x)), int(round(y))] for x, y in arr]
                return json.dumps(arr_rounded)
            except Exception:
                return jstr

        def _round_scores_json(jstr):
            try:
                arr = json.loads(jstr)
                arr_rounded = [round(float(x), 3) for x in arr]
                return json.dumps(arr_rounded)
            except Exception:
                return jstr

        pose_df_merged = pose_df.copy()
        pose_df_merged["keypoints"] = pose_df_merged["keypoints"].apply(_round_kpts_json)
        # Round per-keypoint confidence scores to 3 decimal places for readability/storage
        if "scores" in pose_df_merged.columns:
            pose_df_merged["scores"] = pose_df_merged["scores"].apply(_round_scores_json)

        # Merge pose + tracking metadata on frame_index & id/player_id
        merged = pd.merge(
            pose_df_merged,
            track_df,
            left_on=["frame_index", "player_id"],
            right_on=["frame_index", "id"],
            how="left",
            suffixes=("_pose", "_track"),
        )

        # Detect constants (detect_width/height from pose_df and video_fps from tracking)
        detect_w = None
        detect_h = None
        if "detect_width" in pose_df.columns:
            detect_vals_w = pose_df["detect_width"].dropna().unique()
            detect_w = int(detect_vals_w[0]) if len(detect_vals_w) > 0 else None
        if "detect_height" in pose_df.columns:
            detect_vals_h = pose_df["detect_height"].dropna().unique()
            detect_h = int(detect_vals_h[0]) if len(detect_vals_h) > 0 else None

        fps_val = None
        if "video_fps" in track_df.columns:
            fps_vals = track_df["video_fps"].dropna().unique()
            fps_val = float(fps_vals[0]) if len(fps_vals) > 0 else None

        # Cleanup merged dataframe for readability:
        # - Remove duplicate join columns from tracking side (id)
        # - Remove detect_width/detect_height/video_fps columns (constants represented in filename)
        for col in ["id"]:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True)
        for col in ["detect_width", "detect_height", "video_fps"]:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True)

        # Keep 'scores' in merged because the client uses them to draw skeletons.

        # Build descriptive filename including detect size + fps when available
        name_parts = [f"merged_metadata"]
        if detect_w and detect_h:
            name_parts.append(f"w{detect_w}_h{detect_h}")
        if fps_val:
            name_parts.append(f"fps_{fps_val:.3f}")
        merged_name = "_".join(name_parts) + ".csv"

        # save merged metadata inside foreground/ when available
        merged_parent = (experiment_dir / "foreground") if (experiment_dir / "foreground").exists() else experiment_dir
        merged_path = merged_parent / merged_name
        merged_parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_path, index=False)
        print(f"  Wrote merged metadata to: {merged_path}")
    else:
        print("  Warning: tracking_metadata.csv not found, skipping merge step")

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
                        default="/home/itec/emanuele/Models/SAM/sam3.pt",
                        help="Path to SAM3 model")
    parser.add_argument("--dwpose_det_model", type=str, default=None,
                        help="Path to YOLOX ONNX detection model "
                            "(default: /home/itec/emanuele/Models/DWPose/yolox_l.onnx)")
    parser.add_argument("--dwpose_pose_model", type=str, default=None,
                        help="Path to DW-LL pose ONNX model "
                            "(default: /home/itec/emanuele/Models/DWPose/dw-ll_ucoco_384.onnx)")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Skip SAM3 and run DWPose on an existing experiment directory")
    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"#  PointStream Server")
    print(f"{'#'*80}")

    start_total = time.perf_counter()
    timings = {}

    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        if not args.video_path:
            parser.error("--video_path is required unless --experiment_dir is given")
        step_start = time.perf_counter()
        exp_dir = run_sam_segmentation(args.video_path, args.sam_model)
        timings["sam_segmentation_sec"] = round(time.perf_counter() - step_start, 3)

    step_start = time.perf_counter()
    out_csv = extract_dwpose_keypoints(
        exp_dir,
        det_model=args.dwpose_det_model,
        pose_model=args.dwpose_pose_model,
    )
    timings["dwpose_extraction_sec"] = round(time.perf_counter() - step_start, 3)
    timings["server_total_sec"] = round(time.perf_counter() - start_total, 3)

    try:
        eval_payload = {
            "script": "pointstream_server.py",
            "timestamp": datetime.datetime.now().isoformat(),
            "experiment_dir": str(exp_dir),
            "config": {
                "video_path": args.video_path,
                "sam_model": args.sam_model,
                "dwpose_det_model": args.dwpose_det_model,
                "dwpose_pose_model": args.dwpose_pose_model,
                "experiment_dir": args.experiment_dir,
            },
            "timings": timings,
        }
        # write evaluation metadata next to foreground artifacts when present
        eval_parent = (exp_dir / "foreground") if (exp_dir / "foreground").exists() else exp_dir
        eval_path = eval_parent / "evaluation_server.json"
        eval_parent.mkdir(parents=True, exist_ok=True)
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2)
        print(f"  Wrote evaluation metadata: {eval_path}")
    except Exception as exc:
        print(f"  Warning: failed to write evaluation_server.json: {exc}")

    print(f"\n{'='*80}")
    print(f"Server finished.")
    if out_csv:
        ref_display = (exp_dir / 'foreground' / 'reference') if (exp_dir / 'foreground').exists() else (exp_dir / 'reference')
        print(f"  Experiment : {exp_dir}")
        print(f"  Keypoints  : {out_csv}")
        print(f"  Reference  : {ref_display}")
        print(f"\nNext step – run the client:")
        print(f"  conda activate animate-anyone")
        print(f"  cd /home/itec/emanuele/Moore-AnimateAnyone")
        print(f"  python {POINTSTREAM_DIR}/pointstream_client.py \\")
        print(f"      --experiment_dir {exp_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
