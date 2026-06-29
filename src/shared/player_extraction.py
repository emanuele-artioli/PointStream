import math
from typing import Any
import numpy as np

from src.shared.geometry import get_iou

def coco17_to_dwpose18(coco17: np.ndarray, confidence_threshold: float = 0.2) -> np.ndarray:
    """Convert a COCO-17 pose [17,3] into a DWPose/OpenPose-18 pose [18,3]."""
    if coco17.shape != (17, 3):
        raise ValueError(f"Expected COCO pose shape [17,3], got: {coco17.shape}")

    out = np.zeros((18, 3), dtype=np.float32)
    # openpose18 index -> coco17 index, neck is synthesized
    op18_from_coco17 = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    for op_idx, coco_idx in enumerate(op18_from_coco17):
        if coco_idx is None:
            continue
        out[op_idx] = coco17[coco_idx]

    # Synthesize neck if both shoulders are confident.
    lsho = coco17[5]
    rsho = coco17[6]
    if float(lsho[2]) >= confidence_threshold and float(rsho[2]) >= confidence_threshold:
        neck_xy = 0.5 * (lsho[:2] + rsho[:2])
        neck_conf = min(float(lsho[2]), float(rsho[2]))
        out[1] = np.array([neck_xy[0], neck_xy[1], neck_conf], dtype=np.float32)

    return out

def build_crop_with_padding(frame_bgr: np.ndarray, bbox: list[float] | tuple, pad_ratio: float = 0.15) -> tuple[np.ndarray, tuple[int, int]]:
    """Crops a region from the image with padding, returning the crop and the top-left offset."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    cx1 = int(max(0, np.floor(x1 - pad_x)))
    cy1 = int(max(0, np.floor(y1 - pad_y)))
    cx2 = int(min(w, np.ceil(x2 + pad_x)))
    cy2 = int(min(h, np.ceil(y2 + pad_y)))
    if cx2 <= cx1 or cy2 <= cy1:
        return np.empty((0, 0, 3), dtype=np.uint8), (0, 0)

    return frame_bgr[cy1:cy2, cx1:cx2].copy(), (cx1, cy1)

def track_persons_iou(
    detections: list[dict],
    active_tracks: dict,
    frame_id: int,
    max_gap: int = 30,
    fallback_distance: float = 150.0,
) -> tuple[list[dict], dict]:
    """
    IoU+proximity tracking with configurable gap stitching.
    detections: list of dicts with 'bbox': (x1, y1, x2, y2)
    active_tracks: dict {tid: {'bbox': (x1,y1,x2,y2), 'last_seen': int}}
    Returns: (updated_detections with 'tid', updated_active_tracks)
    """
    next_tid = max(active_tracks.keys()) + 1 if active_tracks else 1
    current_frame_tids = set()
    
    updated_dets = []
    
    for det in detections:
        bbox = det['bbox']
        tid = -1
        
        best_iou = 0.1
        best_dist = fallback_distance
        
        for t, t_info in active_tracks.items():
            if t in current_frame_tids:
                continue
                
            t_bbox = t_info['bbox']
            gap = frame_id - t_info['last_seen']
            
            if gap <= max_gap:
                iou = get_iou(bbox, t_bbox)
                if iou > best_iou:
                    best_iou = iou
                    tid = t
                    best_dist = 0.0 # Priority over pure distance
                    
                if best_iou == 0.1: # Fallback to proximity
                    cx1, cy1 = (bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0
                    cx2, cy2 = (t_bbox[0]+t_bbox[2])/2.0, (t_bbox[1]+t_bbox[3])/2.0
                    dist = math.hypot(cx1-cx2, cy1-cy2)
                    if dist < best_dist:
                        best_dist = dist
                        tid = t
                        
        if tid == -1:
            tid = next_tid
            next_tid += 1
            
        active_tracks[tid] = {'bbox': bbox, 'last_seen': frame_id}
        current_frame_tids.add(tid)
        
        det_out = dict(det)
        det_out['tid'] = tid
        updated_dets.append(det_out)
        
    return updated_dets, active_tracks

def match_rackets_to_players(
    person_dets: list[dict],
    racket_dets: list[dict],
    strategy: str = "proximity",
    racket_scores: dict | None = None
) -> dict[int, dict]:
    """
    Matches rackets to players.
    strategy: "proximity" (fast runtime) or "accumulate" (quality dataset)
    If "accumulate", updates the racket_scores dict in-place and relies on it
    across frames (not done here directly, this function just returns best matches for *this* frame).
    Returns {person_tid: {'racket': racket_det}} for the current frame.
    """
    matches = {}
    if not person_dets or not racket_dets:
        return matches
        
    for r in racket_dets:
        rx = (r['bbox'][0] + r['bbox'][2]) / 2.0
        ry = (r['bbox'][1] + r['bbox'][3]) / 2.0
        
        best_tid = None
        best_dist = float('inf')
        
        for p in person_dets:
            if 'tid' not in p:
                continue
                
            px = (p['bbox'][0] + p['bbox'][2]) / 2.0
            py = (p['bbox'][1] + p['bbox'][3]) / 2.0
            
            dist = (px - rx)**2 + (py - ry)**2
            if dist < best_dist:
                best_dist = dist
                best_tid = p['tid']
                
        if best_tid is not None:
            if strategy == "accumulate" and racket_scores is not None:
                racket_scores[best_tid] = racket_scores.get(best_tid, 0) + 1
            matches[best_tid] = {'racket': r}
            
    return matches

def extract_pose_dwpose18(
    frame_bgr: np.ndarray,
    bbox: list[float] | tuple,
    model: Any, # Ultralytics YOLO pose model or DwposeDetector
    model_type: str = "yolo", # "yolo" or "dwpose"
    pad_ratio: float = 0.10,
) -> np.ndarray | None:
    """Extracts pose using the provided model and converts it to DWPose-18 format."""
    crop, (ox, oy) = build_crop_with_padding(frame_bgr, bbox, pad_ratio=pad_ratio)
    if crop.size == 0:
        return None
        
    if model_type == "yolo":
        results = model.predict(source=crop, verbose=False, conf=0.2)
        if not results:
            return None
            
        keypoints = getattr(results[0], "keypoints", None)
        if keypoints is None or getattr(keypoints, "xy", None) is None or len(keypoints.xy) == 0:
            return None
            
        xy = keypoints.xy[0]
        conf = keypoints.conf[0] if getattr(keypoints, "conf", None) is not None else None
        
        xy_np = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
        
        if conf is None:
            conf_np = np.ones((xy_np.shape[0],), dtype=np.float32)
        else:
            conf_np = conf.cpu().numpy().astype(np.float32) if hasattr(conf, "cpu") else np.asarray(conf, dtype=np.float32)
            
        if xy_np.shape[0] != 17:
            return None # Expected 17 keypoints for YOLO pose
            
        coco17 = np.concatenate([xy_np, conf_np[:, None]], axis=-1).astype(np.float32)
        coco17[:, 0] += float(ox)
        coco17[:, 1] += float(oy)
        
        return coco17_to_dwpose18(coco17)
        
    elif model_type == "dwpose":
        _canvas, pose_json, _ = model(
            crop,
            output_type="np",
            image_and_json=True,
            include_body=True,
            include_hand=False,
            include_face=False,
        )
        
        people = pose_json.get("people", []) if isinstance(pose_json, dict) else []
        if not people:
            return None
            
        raw = people[0].get("pose_keypoints_2d")
        if not raw:
            return None
            
        pts = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 17:
            dw = coco17_to_dwpose18(pts)
        elif pts.shape[0] >= 18:
            dw = pts[:18].copy()
        else:
            return None
            
        dw[:, 0] += float(ox)
        dw[:, 1] += float(oy)
        return dw
        
    return None
