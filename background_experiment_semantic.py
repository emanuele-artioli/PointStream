#!/usr/bin/env python3
"""
Experimental background pipeline with multi-class SAM3 semantic + video tracking.

Goals:
1) Segment and track tennis players + other people across the video.
2) Save rich debugging artifacts:
   - per-object masked crops
   - per-frame masks for tennis players, people, and all tracked objects
   - per-frame background candidates with all tracked objects removed
3) Build panorama from object-removed frames.
4) Add back static people using frame-0 people mask (warped to panorama coords).
5) Reconstruct background video from panorama + intrinsics (AV1).
6) Evaluate reconstruction quality with PSNR (full-frame + object-masked).

This is intentionally experimental to compare methods and tune segmentation/tracking choices.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Runtime compatibility shim for CLIP/OpenCLIP tokenizer expected by SAM3 semantic predictor.
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


def _make_callable_module(module: Any) -> None:
    if module is None or not hasattr(module, "SimpleTokenizer"):
        return

    cls = module.SimpleTokenizer
    try:
        tokenizer_instance = cls()
    except Exception:
        tokenizer_instance = None

    if tokenizer_instance is not None and callable(tokenizer_instance):
        return

    def _simple_tokenizer_call(self, text: Any, context_length: int = 77):
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


@dataclass
class TrackedObject:
    object_id: int
    role: str  # "tennis_player" or "person"
    source_label: str  # semantic label from predictor
    confidence: float
    bbox_xyxy: List[float]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_semantic_first_frame(
    frame0_path: Path,
    sam_model: Path,
    semantic_conf: float,
) -> List[Dict[str, Any]]:
    overrides = dict(
        conf=float(semantic_conf),
        task="segment",
        mode="predict",
        imgsz=1288,
        model=str(sam_model),
        half=True,
        save=False,
        compile=None,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(str(frame0_path))

    prompts = ["tennis player", "person"]
    try:
        results = predictor(text=prompts)
    except Exception as exc:
        raise RuntimeError(
            "SAM3 semantic prediction failed for prompts ['tennis player', 'person']. "
            f"Error: {exc}"
        ) from exc

    detections: List[Dict[str, Any]] = []
    if results and len(results) > 0:
        for det in results[0].boxes:
            cls_id = int(det.cls[0].cpu().item())
            conf = float(det.conf[0].cpu().item())
            bbox = det.xyxy[0].cpu().numpy().tolist()
            label = "tennis_player" if cls_id == 0 else "person"
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": bbox,
            })

    return detections


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = _bbox_area(box_a)
    area_b = _bbox_area(box_b)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0.0 else 0.0


def _suppress_duplicates(detections: Sequence[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: float(d["confidence"]), reverse=True)
    kept: List[Dict[str, Any]] = []
    for d in dets:
        keep = True
        for k in kept:
            if _bbox_iou(d["bbox"], k["bbox"]) >= float(iou_thr):
                keep = False
                break
        if keep:
            kept.append(d)
    return kept


def _select_tracked_objects(
    detections: Sequence[Dict[str, Any]],
    required_tennis_players: int,
    frame_shape: Tuple[int, int],
    min_box_area_ratio: float,
    dedup_iou_thr: float,
    max_people: int,
) -> List[TrackedObject]:
    if not detections:
        raise RuntimeError("No detections on frame 0 from SAM3 semantic predictor")

    frame_h, frame_w = frame_shape
    min_area = float(max(0.0, min_box_area_ratio)) * float(frame_h * frame_w)

    # Remove tiny detections that are unlikely to be meaningful tracked people.
    filtered = [d for d in detections if _bbox_area(d["bbox"]) >= min_area]
    if not filtered:
        filtered = list(detections)

    # Suppress duplicate boxes from semantic predictor overlap.
    filtered = _suppress_duplicates(filtered, iou_thr=dedup_iou_thr)

    tennis_candidates = [
        {"idx": i, **d}
        for i, d in enumerate(filtered)
        if d["label"] == "tennis_player"
    ]
    people_candidates = [
        {"idx": i, **d}
        for i, d in enumerate(filtered)
        if d["label"] == "person"
    ]

    tennis_candidates.sort(key=lambda d: d["confidence"], reverse=True)
    people_candidates.sort(key=lambda d: d["confidence"], reverse=True)

    selected_tennis = tennis_candidates[:required_tennis_players]

    # If semantic "tennis player" detections are insufficient, promote high-confidence person detections.
    if len(selected_tennis) < required_tennis_players:
        need = required_tennis_players - len(selected_tennis)
        promoted = people_candidates[:need]
        selected_tennis.extend(promoted)

    if len(selected_tennis) < required_tennis_players:
        raise RuntimeError(
            f"Could not find {required_tennis_players} player detections on frame 0; "
            f"found only {len(selected_tennis)} after promotion"
        )

    selected_indices = {d["idx"] for d in selected_tennis}

    # Limit additional tracked people to avoid over-fragmentation / moving clutter noise.
    if max_people >= 0:
        people_candidates = people_candidates[: int(max_people)]
    allowed_people_indices = {p["idx"] for p in people_candidates}

    objects: List[TrackedObject] = []
    object_id = 0

    # Keep tennis players first for stable ordering.
    for d in selected_tennis:
        objects.append(
            TrackedObject(
                object_id=object_id,
                role="tennis_player",
                source_label=d["label"],
                confidence=float(d["confidence"]),
                bbox_xyxy=[float(x) for x in d["bbox"]],
            )
        )
        object_id += 1

    # Everything else becomes "person".
    for idx, d in enumerate(filtered):
        if idx in selected_indices:
            continue
        if idx not in allowed_people_indices:
            continue
        objects.append(
            TrackedObject(
                object_id=object_id,
                role="person",
                source_label=d["label"],
                confidence=float(d["confidence"]),
                bbox_xyxy=[float(x) for x in d["bbox"]],
            )
        )
        object_id += 1

    return objects


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

    h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if h is None:
        return np.eye(3, dtype=np.float64)
    return h.astype(np.float64)


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


def _compute_canvas(
    frame_shape: Tuple[int, int, int],
    global_h: List[np.ndarray],
) -> Tuple[np.ndarray, int, int, Tuple[float, float]]:
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


def _encode_video_ffmpeg_av1(
    frames: Sequence[np.ndarray],
    output_path: Path,
    fps: float,
    crf: int,
) -> None:
    if not frames:
        raise RuntimeError("No frames to encode")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        # fallback to OpenCV if ffmpeg is unavailable
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(max(fps, 1.0)),
            (int(w), int(h)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open fallback writer: {output_path}")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{int(w)}x{int(h)}",
        "-r",
        f"{float(max(fps, 1.0))}",
        "-i",
        "-",
        "-c:v",
        "libsvtav1",
        "-crf",
        str(int(crf)),
        "-b:v",
        "0",
        "-threads",
        "0",
        "-row-mt",
        "1",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        returncode = proc.wait()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        if returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed (code={returncode}): {stderr}")
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()


def _masked_psnr(pred: np.ndarray, gt: np.ndarray, keep_mask: np.ndarray) -> Optional[float]:
    if keep_mask.ndim == 3:
        keep_mask = cv2.cvtColor(keep_mask, cv2.COLOR_BGR2GRAY)
    keep = (keep_mask > 0)
    if not np.any(keep):
        return None

    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)

    diff = pred_f - gt_f
    diff2 = np.sum(diff * diff, axis=2)
    mse = float(np.mean(diff2[keep]))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def _evaluate_reconstruction(
    recon_frames: Sequence[np.ndarray],
    gt_frames: Sequence[np.ndarray],
    all_object_masks: Sequence[np.ndarray],
) -> Dict[str, Any]:
    full_psnr: List[float] = []
    background_only_psnr: List[float] = []

    n = min(len(recon_frames), len(gt_frames), len(all_object_masks))
    for i in range(n):
        pred = recon_frames[i]
        gt = gt_frames[i]
        obj_mask = all_object_masks[i]

        if pred.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(obj_mask, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        psnr_full = cv2.PSNR(pred, gt)
        if np.isfinite(psnr_full):
            full_psnr.append(float(psnr_full))

        keep_mask = cv2.bitwise_not(obj_mask)
        psnr_bg = _masked_psnr(pred, gt, keep_mask)
        if psnr_bg is not None and np.isfinite(psnr_bg):
            background_only_psnr.append(float(psnr_bg))

    return {
        "num_eval_frames": int(n),
        "psnr_full_mean": float(np.mean(full_psnr)) if full_psnr else None,
        "psnr_full_std": float(np.std(full_psnr)) if full_psnr else None,
        "psnr_background_only_mean": float(np.mean(background_only_psnr)) if background_only_psnr else None,
        "psnr_background_only_std": float(np.std(background_only_psnr)) if background_only_psnr else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental SAM3-based background pipeline with static people")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--sam_model", type=str, default="/home/itec/emanuele/Models/SAM/sam3.pt", help="Path to SAM3 model")
    parser.add_argument("--semantic_conf", type=float, default=0.10, help="Semantic predictor confidence threshold")
    parser.add_argument("--required_tennis_players", type=int, default=2, help="How many tennis players to force-select")
    parser.add_argument("--max_people", type=int, default=8, help="Max extra people to track (-1 = unlimited)")
    parser.add_argument("--min_box_area_ratio", type=float, default=0.0008, help="Min bbox area ratio (relative to frame area)")
    parser.add_argument("--dedup_iou_thr", type=float, default=0.65, help="IoU threshold for duplicate suppression")
    parser.add_argument("--skip_frames", type=int, default=1, help="Use every N-th frame for panorama/reconstruction")
    parser.add_argument("--nfeatures", type=int, default=1200, help="ORB feature count")
    parser.add_argument("--av1_crf", type=int, default=30, help="AV1 quality (lower is better quality)")
    parser.add_argument("--output_prefix", type=str, default="semantic", help="Prefix for generated files")
    args = parser.parse_args()

    if args.skip_frames <= 0:
        raise ValueError("--skip_frames must be >= 1")

    exp_dir = Path(args.experiment_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video_path).resolve()
    sam_model = Path(args.sam_model).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not sam_model.exists():
        raise FileNotFoundError(f"SAM model not found: {sam_model}")

    out_dir = exp_dir / "background_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops_root = out_dir / "masked_crops"
    masks_tennis_dir = out_dir / "masks_tennis_players"
    masks_people_dir = out_dir / "masks_people"
    masks_all_dir = out_dir / "masks_all_objects"
    frames_removed_dir = out_dir / "frames_all_objects_removed"
    for p in [crops_root, masks_tennis_dir, masks_people_dir, masks_all_dir, frames_removed_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print("#  Background Experiment (semantic players + people)")
    print("#" * 80)
    print(f"Experiment : {exp_dir}")
    print(f"Video      : {video_path}")
    print(f"Output     : {out_dir}")

    t_total = time.perf_counter()

    # Frame 0 for semantic detection setup.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read frame 0")

    frame0_path = out_dir / "frame0_temp.png"
    cv2.imwrite(str(frame0_path), frame0)

    t0 = time.perf_counter()
    detections = _run_semantic_first_frame(frame0_path, sam_model, args.semantic_conf)
    if frame0_path.exists():
        frame0_path.unlink()

    tracked_objects = _select_tracked_objects(
        detections,
        args.required_tennis_players,
        frame_shape=frame0.shape[:2],
        min_box_area_ratio=args.min_box_area_ratio,
        dedup_iou_thr=args.dedup_iou_thr,
        max_people=args.max_people,
    )
    t_semantic = time.perf_counter() - t0

    print(f"Semantic detections on frame 0: {len(detections)}")
    print(f"Tracked objects selected: {len(tracked_objects)}")
    for obj in tracked_objects:
        print(
            f"  id={obj.object_id:02d} role={obj.role:13s} source={obj.source_label:13s} "
            f"conf={obj.confidence:.3f} bbox={obj.bbox_xyxy}"
        )

    # Track all selected objects through the whole video.
    t0 = time.perf_counter()
    video_overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        imgsz=644,
        model=str(sam_model),
        half=True,
        save=False,
        compile=None,
    )
    video_predictor = SAM3VideoPredictor(overrides=video_overrides)
    bboxes = [obj.bbox_xyxy for obj in tracked_objects]
    stream = video_predictor(source=str(video_path), bboxes=bboxes, stream=True)

    all_frames: List[np.ndarray] = []
    all_masks_all: List[np.ndarray] = []
    all_frame_indices: List[int] = []
    people_layer_frame0: Optional[np.ndarray] = None

    for frame_idx, result in enumerate(stream):
        frame = result.orig_img
        h, w = frame.shape[:2]

        mask_tennis = np.zeros((h, w), dtype=np.uint8)
        mask_people = np.zeros((h, w), dtype=np.uint8)
        mask_all = np.zeros((h, w), dtype=np.uint8)

        if result.masks is not None and len(result.boxes) > 0:
            for obj_idx, (det, m) in enumerate(zip(result.boxes, result.masks)):
                if obj_idx >= len(tracked_objects):
                    continue
                obj = tracked_objects[obj_idx]

                bbox = det.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                mask_data = m.data.cpu().numpy()[0]
                mask_uint8 = (mask_data > 0.5).astype(np.uint8) * 255
                mask_all = np.maximum(mask_all, mask_uint8)
                if obj.role == "tennis_player":
                    mask_tennis = np.maximum(mask_tennis, mask_uint8)
                else:
                    mask_people = np.maximum(mask_people, mask_uint8)

                # Save masked crop for each object.
                crop = frame[y1:y2, x1:x2]
                crop_mask = mask_uint8[y1:y2, x1:x2]
                if crop.size > 0 and crop_mask.size > 0:
                    crop_mask_3ch = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
                    masked_crop = cv2.bitwise_and(crop, crop_mask_3ch)
                    obj_dir = crops_root / obj.role / f"id{obj.object_id:02d}"
                    obj_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(obj_dir / f"{frame_idx:05d}.png"), masked_crop)

        removed = frame.copy()
        removed[mask_all > 0] = 0

        cv2.imwrite(str(masks_tennis_dir / f"{frame_idx:05d}.png"), mask_tennis)
        cv2.imwrite(str(masks_people_dir / f"{frame_idx:05d}.png"), mask_people)
        cv2.imwrite(str(masks_all_dir / f"{frame_idx:05d}.png"), mask_all)
        cv2.imwrite(str(frames_removed_dir / f"{frame_idx:05d}.png"), removed)

        if frame_idx == 0:
            people_layer_frame0 = cv2.bitwise_and(frame, cv2.cvtColor(mask_people, cv2.COLOR_GRAY2BGR))

        all_frames.append(frame)
        all_masks_all.append(mask_all)
        all_frame_indices.append(frame_idx)

    t_tracking = time.perf_counter() - t0

    if not all_frames:
        raise RuntimeError("No frames were processed by SAM3 video predictor")

    if people_layer_frame0 is None:
        people_layer_frame0 = np.zeros_like(all_frames[0])

    # Subsample for panorama/reconstruction according to skip_frames.
    selected_idx = list(range(0, len(all_frames), args.skip_frames))
    sel_frames_removed = [cv2.imread(str(frames_removed_dir / f"{i:05d}.png")) for i in selected_idx]
    sel_frames_removed = [f for f in sel_frames_removed if f is not None]
    sel_gt_frames = [all_frames[i] for i in selected_idx[: len(sel_frames_removed)]]
    sel_masks_all = [all_masks_all[i] for i in selected_idx[: len(sel_frames_removed)]]
    sel_frame_indices = selected_idx[: len(sel_frames_removed)]

    if not sel_frames_removed:
        raise RuntimeError("No frames available after skip-frame selection")

    t0 = time.perf_counter()
    global_h = _compute_global_homographies(sel_frames_removed, nfeatures=args.nfeatures)
    t_align = time.perf_counter() - t0

    t0 = time.perf_counter()
    h_translation, canvas_w, canvas_h, (tx, ty) = _compute_canvas(sel_frames_removed[0].shape, global_h)
    panorama = _stitch_panorama(sel_frames_removed, global_h, h_translation, canvas_w, canvas_h)

    # Add frame-0 people back as static content in panorama space (not tennis players).
    people_warp = cv2.warpPerspective(people_layer_frame0, h_translation @ global_h[0], (canvas_w, canvas_h))
    people_mask = cv2.cvtColor(people_warp, cv2.COLOR_BGR2GRAY)
    _, people_mask = cv2.threshold(people_mask, 1, 255, cv2.THRESH_BINARY)
    panorama = cv2.bitwise_and(panorama, panorama, mask=cv2.bitwise_not(people_mask))
    panorama = cv2.add(panorama, people_warp)

    t_stitch = time.perf_counter() - t0

    pano_path = out_dir / f"background_panorama_{args.output_prefix}.png"
    intrinsics_path = out_dir / f"background_intrinsics_{args.output_prefix}.json"
    recon_video_path = out_dir / f"background_reconstructed_{args.output_prefix}.mp4"
    eval_json_path = out_dir / f"evaluation_background_experiment_{args.output_prefix}.json"

    ok = cv2.imwrite(str(pano_path), panorama)
    if not ok:
        raise RuntimeError(f"Failed to write panorama: {pano_path}")

    # Reconstruct from panorama + intrinsics (same approach as background_client).
    frame_h, frame_w = sel_frames_removed[0].shape[:2]
    recon_fps = float(source_fps) / float(max(args.skip_frames, 1)) if source_fps > 0 else 24.0

    t0 = time.perf_counter()
    recon_frames: List[np.ndarray] = []
    for h_global in global_h:
        h_comp = h_translation @ h_global
        h_inv = np.linalg.inv(h_comp)
        rec = cv2.warpPerspective(panorama, h_inv, (int(frame_w), int(frame_h)))
        recon_frames.append(rec)

    _encode_video_ffmpeg_av1(recon_frames, recon_video_path, fps=recon_fps, crf=args.av1_crf)
    t_reconstruct = time.perf_counter() - t0

    # Evaluate reconstruction quality against original frames.
    metrics = _evaluate_reconstruction(recon_frames, sel_gt_frames, sel_masks_all)

    intrinsics_payload: Dict[str, Any] = {
        "script": "background_experiment_semantic.py",
        "timestamp": datetime.now().isoformat(),
        "video_path": str(video_path),
        "source_fps": float(source_fps) if source_fps else None,
        "source_frame_count": int(total_frame_count),
        "processed_frame_count": int(len(sel_frames_removed)),
        "processed_frame_indices": [int(i) for i in sel_frame_indices],
        "skip_frames": int(args.skip_frames),
        "frame_size": [int(frame_w), int(frame_h)],
        "canvas_size": [int(canvas_w), int(canvas_h)],
        "translation": [float(tx), float(ty)],
        "homographies": [h.tolist() for h in global_h],
        "method": {
            "semantic_prompts": ["tennis player", "person"],
            "required_tennis_players": int(args.required_tennis_players),
            "static_people_from_frame0": True,
        },
    }
    _write_json(intrinsics_path, intrinsics_payload)

    total_sec = time.perf_counter() - t_total
    eval_payload: Dict[str, Any] = {
        "script": "background_experiment_semantic.py",
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": str(exp_dir),
        "output_dir": str(out_dir),
        "config": {
            "video_path": str(video_path),
            "sam_model": str(sam_model),
            "semantic_conf": float(args.semantic_conf),
            "required_tennis_players": int(args.required_tennis_players),
            "max_people": int(args.max_people),
            "min_box_area_ratio": float(args.min_box_area_ratio),
            "dedup_iou_thr": float(args.dedup_iou_thr),
            "skip_frames": int(args.skip_frames),
            "nfeatures": int(args.nfeatures),
            "av1_crf": int(args.av1_crf),
            "output_prefix": args.output_prefix,
        },
        "objects": {
            "num_semantic_detections_frame0": int(len(detections)),
            "num_tracked_objects": int(len(tracked_objects)),
            "num_tennis_players": int(sum(1 for o in tracked_objects if o.role == "tennis_player")),
            "num_people": int(sum(1 for o in tracked_objects if o.role == "person")),
            "tracked_objects": [
                {
                    "object_id": int(o.object_id),
                    "role": o.role,
                    "source_label": o.source_label,
                    "confidence": float(o.confidence),
                    "bbox_xyxy": [float(x) for x in o.bbox_xyxy],
                }
                for o in tracked_objects
            ],
        },
        "timings": {
            "semantic_frame0_sec": round(t_semantic, 3),
            "tracking_and_masks_sec": round(t_tracking, 3),
            "alignment_sec": round(t_align, 3),
            "stitch_and_static_people_sec": round(t_stitch, 3),
            "reconstruction_sec": round(t_reconstruct, 3),
            "total_sec": round(total_sec, 3),
        },
        "quality": metrics,
        "outputs": {
            "panorama": str(pano_path),
            "intrinsics": str(intrinsics_path),
            "reconstructed_video": str(recon_video_path),
            "masks_tennis_dir": str(masks_tennis_dir),
            "masks_people_dir": str(masks_people_dir),
            "masks_all_dir": str(masks_all_dir),
            "frames_all_objects_removed_dir": str(frames_removed_dir),
            "masked_crops_root": str(crops_root),
        },
    }
    _write_json(eval_json_path, eval_payload)

    print(f"Panorama saved          : {pano_path}")
    print(f"Intrinsics saved        : {intrinsics_path}")
    print(f"Reconstructed video     : {recon_video_path}")
    print(f"Evaluation JSON         : {eval_json_path}")
    print(
        "Quality (PSNR)         : "
        f"full_mean={metrics['psnr_full_mean']} | "
        f"background_only_mean={metrics['psnr_background_only_mean']}"
    )
    print(f"Total time (sec)        : {total_sec:.3f}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
