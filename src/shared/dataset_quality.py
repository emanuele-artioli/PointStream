"""Conservative, auditable gates for SAM3.1 tennis training labels.

The filters intentionally prefer precision over recall. They only decide
whether an observation is suitable for a training view; raw observations stay
available in the audit manifest with the reason they were withheld.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class QualityDecision:
    eligible: bool
    score: float
    reasons: tuple[str, ...]
    features: Mapping[str, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "score": round(float(self.score), 6),
            "reasons": list(self.reasons),
            "features": {key: round(float(value), 6) for key, value in self.features.items()},
        }


def mask_geometry(mask: np.ndarray) -> dict[str, float] | None:
    """Return size and compactness features for a non-empty 2-D mask."""
    binary = np.asarray(mask) != 0
    if binary.ndim != 2 or not np.any(binary):
        return None
    ys, xs = np.nonzero(binary)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    area = int(xs.size)
    return {
        "area_px": float(area),
        "bbox_width_px": float(width),
        "bbox_height_px": float(height),
        "bbox_diagonal_px": float(math.hypot(width, height)),
        "aspect_ratio": float(max(width, height) / max(1, min(width, height))),
        "fill_ratio": float(area / max(1, width * height)),
        "center_x_px": float(xs.mean()),
        "center_y_px": float(ys.mean()),
    }


def _pose_quality(pose: Any | None) -> float:
    if pose is None:
        return 0.0
    visibility = np.asarray(getattr(pose, "visibility", ()), dtype=np.uint8).reshape(-1)
    values = np.asarray(getattr(pose, "values", ()), dtype=np.float32)
    if visibility.size == 0:
        return 0.0
    # COCO WholeBody-133 begins with its 17 body joints. Requiring visible
    # body evidence avoids counting detailed face/hand keypoints alone.
    body = visibility[:17]
    visible_fraction = float(np.count_nonzero(body == 2) / max(1, body.size))
    confidence = 0.0
    if values.ndim == 2 and values.shape[1] >= 3:
        body_scores = np.clip(values[:17, 2], 0.0, 1.0)
        confidence = float(body_scores.mean()) if body_scores.size else 0.0
    return float(np.clip(0.65 * visible_fraction + 0.35 * confidence, 0.0, 1.0))


def estimate_camera_translations(frames: np.ndarray) -> np.ndarray:
    """Estimate frame-to-frame global translation from the dominant scene.

    Phase correlation runs on small grayscale frames; the court/background
    dominates the player pixels, so its motion is removed before scoring player
    motion. A zero translation is used when a frame pair cannot be estimated.
    """
    image = np.asarray(frames)
    if image.ndim != 4 or len(image) < 2:
        return np.zeros((max(0, len(image) - 1), 2), dtype=np.float32)
    try:
        import cv2
    except ImportError:
        return np.zeros((len(image) - 1, 2), dtype=np.float32)
    height, width = image.shape[1:3]
    scale = min(1.0, 384.0 / max(height, width))
    small_size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
    shifts: list[tuple[float, float]] = []
    window = None
    for previous, current in zip(image[:-1], image[1:], strict=True):
        prev_small = cv2.resize(previous[..., :3].astype(np.uint8), small_size, interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(current[..., :3].astype(np.uint8), small_size, interpolation=cv2.INTER_AREA)
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY)
        if window is None:
            window = cv2.createHanningWindow(small_size, cv2.CV_32F)
        try:
            shift, response = cv2.phaseCorrelate(
                prev_gray.astype(np.float32), curr_gray.astype(np.float32), window
            )
            if not np.isfinite((*shift, response)).all() or response < 0.02:
                shift = (0.0, 0.0)
            shifts.append((float(shift[0] / scale), float(shift[1] / scale)))
        except cv2.error:
            shifts.append((0.0, 0.0))
    return np.asarray(shifts, dtype=np.float32)


def _motion_by_track(
    candidates: Mapping[tuple[int, str], Any],
    translations: np.ndarray,
    frame_shape: tuple[int, int],
) -> dict[str, float]:
    height, width = frame_shape
    frame_diagonal = math.hypot(width, height)
    centroids: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for (frame_index, object_id), item in candidates.items():
        geometry = mask_geometry(item.mask) if getattr(item, "mask", None) is not None else None
        if geometry is not None:
            centroids[object_id][frame_index] = (
                geometry["center_x_px"], geometry["center_y_px"]
            )
    result: dict[str, float] = {}
    for object_id, points in centroids.items():
        residuals: list[float] = []
        indices = sorted(points)
        for previous_index, current_index in zip(indices, indices[1:]):
            gap = current_index - previous_index
            if gap <= 0:
                continue
            camera = translations[previous_index:current_index].sum(axis=0) if len(translations) else (0.0, 0.0)
            dx = points[current_index][0] - points[previous_index][0] - float(camera[0])
            dy = points[current_index][1] - points[previous_index][1] - float(camera[1])
            residuals.append(math.hypot(dx, dy) / gap)
        mean_motion = float(np.mean(residuals)) if residuals else 0.0
        # Saturate at 0.3% of the image diagonal per frame. This is a ranking
        # signal, not a claim that lower-motion players are absent.
        result[object_id] = float(np.clip(mean_motion / max(1.0, frame_diagonal * 0.003), 0.0, 1.0))
    return result


def filter_player_candidates(
    candidates: Mapping[tuple[int, str], Any],
    poses_by_frame: Mapping[int, list[tuple[str, Any, Any]]],
    frames: np.ndarray,
    *,
    max_players_per_frame: int = 2,
    minimum_track_score: float = 0.58,
    minimum_sam_confidence: float = 0.40,
    minimum_pose_support: float = 0.08,
) -> tuple[dict[tuple[int, str], QualityDecision], dict[str, dict[str, float]]]:
    """Keep up to two temporally supported, person-shaped, posed SAM tracks."""
    if max_players_per_frame <= 0:
        raise ValueError("max_players_per_frame must be positive")
    image = np.asarray(frames)
    frame_count, height, width = len(image), int(image.shape[1]), int(image.shape[2])
    pose_map = {
        (frame_index, object_id): pose
        for frame_index, rows in poses_by_frame.items()
        for object_id, pose, _transform in rows
    }
    translations = estimate_camera_translations(image)
    motion = _motion_by_track(candidates, translations, (height, width))
    by_object: dict[str, list[tuple[int, Any, dict[str, float], float, float, float]]] = defaultdict(list)
    per_frame_reasons: dict[tuple[int, str], list[str]] = {}

    for key, item in candidates.items():
        frame_index, object_id = key
        raw_mask = getattr(item, "mask", None)
        geometry = mask_geometry(raw_mask) if raw_mask is not None else None
        reasons: list[str] = []
        if geometry is None:
            reasons.append("mask_missing_or_empty")
            per_frame_reasons[key] = reasons
            continue
        area_fraction = geometry["area_px"] / max(1.0, width * height)
        height_fraction = geometry["bbox_height_px"] / max(1.0, height)
        aspect = geometry["aspect_ratio"]
        fill = geometry["fill_ratio"]
        shape_score = float(
            np.clip(1.0 - max(0.0, area_fraction - 0.05) / 0.15, 0.0, 1.0)
            * np.clip((height_fraction - 0.012) / 0.05, 0.0, 1.0)
            * np.clip(1.0 - max(0.0, aspect - 2.5) / 2.5, 0.0, 1.0)
            * np.clip(fill / 0.25, 0.0, 1.0)
        )
        if area_fraction < 0.000015 or area_fraction > 0.20:
            reasons.append("person_mask_area_outside_range")
        if height_fraction < 0.012 or aspect > 5.0 or fill < 0.035:
            reasons.append("person_mask_shape_implausible")
        raw_score = getattr(item, "score", None)
        model_score = float(raw_score) if raw_score is not None else 0.0
        if raw_score is None:
            reasons.append("sam_confidence_missing")
        elif model_score < minimum_sam_confidence:
            reasons.append(f"sam_confidence_below_{minimum_sam_confidence:.2f}")
        pose_score = _pose_quality(pose_map.get(key))
        if pose_score < minimum_pose_support:
            reasons.append("body_pose_support_too_low")
        per_frame_reasons[key] = reasons
        by_object[object_id].append((frame_index, item, geometry, model_score, pose_score, shape_score))

    track_quality: dict[str, dict[str, float]] = {}
    for object_id, records in by_object.items():
        valid = [record for record in records if not per_frame_reasons[(record[0], object_id)]]
        persistence = len(valid) / max(1, frame_count)
        model_score = float(np.mean([row[3] for row in valid])) if valid else 0.0
        pose_score = float(np.mean([row[4] for row in valid])) if valid else 0.0
        shape_score = float(np.mean([row[5] for row in valid])) if valid else 0.0
        movement = motion.get(object_id, 0.0)
        score = 0.32 * model_score + 0.25 * pose_score + 0.20 * movement + 0.13 * min(1.0, persistence * 2.0) + 0.10 * shape_score
        track_quality[object_id] = {
            "score": score,
            "sam_confidence": model_score,
            "pose_support": pose_score,
            "camera_compensated_motion": movement,
            "persistence": persistence,
            "shape": shape_score,
        }

    by_frame: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for (frame_index, object_id), reasons in per_frame_reasons.items():
        if reasons:
            continue
        track_score = track_quality.get(object_id, {}).get("score", 0.0)
        if track_score >= minimum_track_score:
            by_frame[frame_index].append((object_id, track_score))

    decisions: dict[tuple[int, str], QualityDecision] = {}
    for key, item in candidates.items():
        frame_index, object_id = key
        reasons = list(per_frame_reasons.get(key, []))
        track = track_quality.get(object_id, {})
        track_score = float(track.get("score", 0.0))
        if track_score < minimum_track_score:
            reasons.append("player_track_score_below_threshold")
        ranked = sorted(by_frame.get(frame_index, ()), key=lambda row: (-row[1], row[0]))
        retained_ids = {candidate_id for candidate_id, _score in ranked[:max_players_per_frame]}
        if not reasons and object_id not in retained_ids:
            reasons.append("exceeds_two_player_frame_limit")
        local_geometry = mask_geometry(item.mask) if getattr(item, "mask", None) is not None else None
        local_score = float(getattr(item, "score", 0.0) or 0.0)
        local_pose = _pose_quality(pose_map.get(key))
        score = 0.55 * track_score + 0.20 * local_score + 0.15 * local_pose + 0.10 * float(track.get("shape", 0.0))
        features = {**track, "frame_sam_confidence": local_score, "frame_pose_support": local_pose}
        if local_geometry is not None:
            features["area_fraction"] = local_geometry["area_px"] / max(1.0, width * height)
        decisions[key] = QualityDecision(not reasons, score, tuple(dict.fromkeys(reasons)), features)
    return decisions, track_quality


def filter_racket_candidates(
    candidates: Mapping[tuple[int, str], Any],
    players: Mapping[tuple[int, str], Any],
    player_decisions: Mapping[tuple[int, str], QualityDecision],
    associations: Mapping[tuple[int, str], Any],
    poses_by_frame: Mapping[int, list[tuple[str, Any, Any]]],
    frame_shape: tuple[int, int],
    *,
    minimum_score: float = 0.68,
    minimum_sam_confidence: float = 0.50,
    maximum_player_area_fraction: float = 0.85,
    maximum_player_extent_fraction: float = 1.0,
    maximum_player_mask_overlap_fraction: float = 0.90,
) -> tuple[dict[tuple[int, str], QualityDecision], dict[str, dict[str, float]]]:
    """Reject net-like or weak racket masks and retain only wrist-linked tracks."""
    height, width = frame_shape
    pose_map = {
        (frame_index, object_id): pose
        for frame_index, rows in poses_by_frame.items()
        for object_id, pose, _transform in rows
    }
    preliminary: dict[tuple[int, str], dict[str, float | str]] = {}
    good_by_track: CounterLike = defaultdict(int)
    for key, item in candidates.items():
        frame_index, object_id = key
        reasons: list[str] = []
        racket_mask_raw = getattr(item, "mask", None)
        geometry = mask_geometry(racket_mask_raw) if racket_mask_raw is not None else None
        if geometry is None:
            reasons.append("racket_mask_missing_or_empty")
        association = associations.get(key)
        player_id = getattr(association, "associated_player_id", None) if association is not None else None
        if player_id is None or not player_decisions.get((frame_index, player_id), QualityDecision(False, 0, (), {})).eligible:
            reasons.append("no_retained_player_wrist_association")
        player_item = players.get((frame_index, player_id)) if player_id is not None else None
        player_mask_raw = getattr(player_item, "mask", None) if player_item is not None else None
        player_geometry = mask_geometry(player_mask_raw) if player_mask_raw is not None else None
        raw_score = getattr(item, "score", None)
        confidence = float(raw_score) if raw_score is not None else 0.0
        if raw_score is None:
            reasons.append("racket_sam_confidence_missing")
        elif confidence < minimum_sam_confidence:
            reasons.append(f"racket_sam_confidence_below_{minimum_sam_confidence:.2f}")
        ratio = 1.0
        extent_ratio = 1.0
        player_overlap_ratio = 0.0
        aspect_score = 0.0
        fill_score = 0.0
        if (
            geometry is not None and player_geometry is not None
            and racket_mask_raw is not None and player_mask_raw is not None
        ):
            ratio = geometry["area_px"] / max(1.0, player_geometry["area_px"])
            extent_ratio = geometry["bbox_diagonal_px"] / max(1.0, player_geometry["bbox_diagonal_px"])
            racket_mask = np.asarray(racket_mask_raw) != 0
            player_mask = np.asarray(player_mask_raw) != 0
            if racket_mask.shape != player_mask.shape:
                reasons.append("racket_player_mask_shape_mismatch")
            else:
                player_overlap_ratio = float(
                    np.count_nonzero(racket_mask & player_mask) / max(1, np.count_nonzero(racket_mask))
                )
                if player_overlap_ratio > maximum_player_mask_overlap_fraction:
                    reasons.append("racket_mask_mostly_overlaps_player_mask")
            aspect = geometry["aspect_ratio"]
            aspect_score = float(np.clip(1.0 - max(0.0, aspect - 3.0) / 3.0, 0.0, 1.0))
            fill_score = float(np.clip(geometry["fill_ratio"] / 0.25, 0.0, 1.0))
            if ratio >= maximum_player_area_fraction:
                reasons.append("racket_area_not_smaller_than_player")
            if extent_ratio > maximum_player_extent_fraction:
                reasons.append("racket_extent_larger_than_player")
            if aspect < 1.0 or aspect > 6.0:
                reasons.append("racket_shape_aspect_implausible")
            if geometry["fill_ratio"] < 0.035 or geometry["area_px"] < 4:
                reasons.append("racket_mask_too_thin_or_tiny")
        else:
            reasons.append("player_size_reference_unavailable")
        pose = pose_map.get((frame_index, player_id)) if player_id is not None else None
        wrist_point = _associated_wrist_point(pose, getattr(association, "associated_wrist", None))
        proximity = 0.0
        if geometry is not None and wrist_point is not None:
            dx = geometry["center_x_px"] - wrist_point[0]
            dy = geometry["center_y_px"] - wrist_point[1]
            distance = math.hypot(dx, dy)
            limit = max(48.0, min(160.0, (player_geometry or {}).get("bbox_diagonal_px", 192.0) * 0.30))
            proximity = float(np.clip(1.0 - distance / limit, 0.0, 1.0))
            if distance > limit:
                reasons.append("racket_too_far_from_associated_wrist")
        elif player_id is not None:
            reasons.append("associated_wrist_pose_unavailable")
        if not reasons:
            good_by_track[object_id] += 1
        preliminary[key] = {
            "reasons": ";".join(dict.fromkeys(reasons)),
            "confidence": confidence,
            "area_ratio": ratio,
            "extent_ratio": extent_ratio,
            "player_mask_overlap_fraction": player_overlap_ratio,
            "aspect_score": aspect_score,
            "fill_score": fill_score,
            "wrist_proximity": proximity,
            "player_id": player_id or "",
        }

    track_quality: dict[str, dict[str, float]] = {}
    for key, row in preliminary.items():
        object_id = key[1]
        count = good_by_track[object_id]
        continuity = min(1.0, count / 3.0)
        score = (
            0.30 * float(row["confidence"])
            + 0.22 * float(row["wrist_proximity"])
            + 0.20 * max(0.0, 1.0 - float(row["area_ratio"]))
            + 0.16 * (0.55 * float(row["aspect_score"]) + 0.45 * float(row["fill_score"]))
            + 0.12 * continuity
        )
        current = track_quality.setdefault(object_id, {"score": score, "accepted_observations": float(count), "continuity": continuity})
        current["score"] = max(current["score"], score)

    decisions: dict[tuple[int, str], QualityDecision] = {}
    for key, row in preliminary.items():
        object_id = key[1]
        reasons = [reason for reason in str(row["reasons"]).split(";") if reason]
        accepted_count = int(good_by_track[object_id])
        if accepted_count < 2:
            reasons.append("racket_track_not_persistent_across_two_frames")
        track_score = track_quality.get(object_id, {}).get("score", 0.0)
        if track_score < minimum_score:
            reasons.append("racket_track_score_below_threshold")
        eligible = not reasons
        features = {
            "track_score": track_score,
            "sam_confidence": float(row["confidence"]),
            "area_ratio_to_player": float(row["area_ratio"]),
            "extent_ratio_to_player": float(row["extent_ratio"]),
            "player_mask_overlap_fraction": float(row["player_mask_overlap_fraction"]),
            "wrist_proximity": float(row["wrist_proximity"]),
            "continuity": float(track_quality.get(object_id, {}).get("continuity", 0.0)),
        }
        decisions[key] = QualityDecision(eligible, track_score, tuple(dict.fromkeys(reasons)), features)
    return decisions, track_quality


def _associated_wrist_point(pose: Any | None, wrist_name: str | None) -> tuple[float, float] | None:
    if pose is None or wrist_name is None:
        return None
    try:
        from src.contracts.keypoints import schema as resolve_schema

        index = resolve_schema(pose.schema.name).index_of.get(wrist_name)
    except (AttributeError, ValueError):
        return None
    values = np.asarray(getattr(pose, "values", ()), dtype=np.float64)
    if index is None or values.ndim != 2 or index >= len(values) or values.shape[1] < 2:
        return None
    if values.shape[1] >= 3 and values[index, 2] < 0.20:
        return None
    point = (float(values[index, 0]), float(values[index, 1]))
    return point if np.isfinite(point).all() else None


CounterLike = dict[str, int]


__all__ = [
    "QualityDecision",
    "estimate_camera_translations",
    "filter_player_candidates",
    "filter_racket_candidates",
    "mask_geometry",
]
