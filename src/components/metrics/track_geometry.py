"""Spatial and temporal errors for tracked court and ball landmarks.

This module scores already-associated landmarks. It deliberately does not
detect or associate landmarks in RGB frames; those inputs need their own frozen
annotation/detector protocol and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LandmarkTrackScore:
    """Unweighted position, velocity, and detection measurements for a track.

    Position and velocity errors are measured only where both reference and
    decoded coordinates exist. Recall and tolerance-hit rate use every visible
    reference point, so a missing decode cannot silently improve accuracy.
    """

    mean_position_error_px: float | None
    p95_position_error_px: float | None
    mean_velocity_error_px_per_s: float | None
    detection_precision: float
    detection_recall: float
    tolerance_hit_rate: float
    reference_points: int
    decoded_points: int
    matched_points: int
    missed_points: int
    false_positives: int
    velocity_pairs: int
    tolerance_px: float


def score_landmark_tracks(
    reference_xy: np.ndarray,
    decoded_xy: np.ndarray,
    *,
    timestamps_s: np.ndarray,
    tolerance_px: float,
    reference_visible: np.ndarray | None = None,
) -> LandmarkTrackScore:
    """Measure aligned court/ball tracks in display-pixel coordinates.

    Args:
        reference_xy: ``(T, K, 2)`` source coordinates. Each K slot has a fixed
            semantic identity (for example, a named court intersection or the
            ball center); ``(nan, nan)`` denotes an unlabelled point.
        decoded_xy: ``(T, K, 2)`` coordinates from the decoded presentation.
            A pair of NaNs denotes a missed detection. Coordinates must use
            the same display-pixel system and point ordering as the reference.
        timestamps_s: Strictly increasing presentation timestamps in seconds,
            one per frame. These determine velocity units and prevent frame-rate
            changes from silently changing the temporal score.
        tolerance_px: Positive display-pixel tolerance for the hit-rate measure.
        reference_visible: Optional ``(T, K)`` mask of points expected to be
            visible. If omitted, finite reference coordinates define visibility.

    Returns:
        A metric vector with position error on matched points, recall and hit
        rate over all visible reference points, precision over all decoded
        detections, and velocity error over adjacent matched samples. No
        weighted composite is formed.

    Raises:
        ValueError: If arrays, visibility, timestamps, or tolerance are invalid,
            or if no reference landmark is visible.

    Invariant:
        Missing predictions remain in the recall/hit-rate denominators, and
        velocity is never computed across a missing or invisible frame.
    """
    try:
        reference = np.asarray(reference_xy, dtype=np.float64)
        decoded = np.asarray(decoded_xy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("track coordinates must be numeric arrays") from exc

    if reference.shape != decoded.shape:
        raise ValueError("reference and decoded tracks must have the same shape")
    if reference.ndim != 3 or reference.shape[-1] != 2:
        raise ValueError("track coordinates must have shape (T, K, 2)")

    def point_states(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        missing = np.isnan(points).all(axis=-1)
        partial_nan = np.isnan(points).any(axis=-1) & ~missing
        if np.any(partial_nan):
            raise ValueError("both coordinates must be NaN for a missing point")
        finite = np.isfinite(points).all(axis=-1)
        if np.any(~finite & ~missing):
            raise ValueError("coordinates must be finite or both coordinates NaN")
        return finite, missing

    reference_finite, _ = point_states(reference)
    decoded_finite, _ = point_states(decoded)
    frame_count, point_count, _ = reference.shape

    try:
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be a finite one-dimensional array") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        frame_deltas = np.diff(timestamps)
    if (
        timestamps.shape != (frame_count,)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(frame_deltas).all()
        or np.any(frame_deltas <= 0.0)
    ):
        raise ValueError("timestamps must be finite and strictly increasing")

    try:
        tolerance_value = np.asarray(tolerance_px)
        if tolerance_value.ndim != 0:
            raise ValueError
        tolerance = float(tolerance_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be a finite positive scalar") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive scalar")

    if reference_visible is None:
        visible = reference_finite
    else:
        visible_input = np.asarray(reference_visible)
        if visible_input.shape != (frame_count, point_count):
            raise ValueError("reference_visible must have shape (T, K)")
        if visible_input.dtype.kind == "b":
            visible = visible_input
        elif visible_input.dtype.kind in "iuf":
            if not np.isfinite(visible_input).all() or np.any(
                (visible_input != 0) & (visible_input != 1)
            ):
                raise ValueError("reference_visible must be a boolean mask")
            visible = visible_input.astype(bool)
        else:
            raise ValueError("reference_visible must be a boolean mask")
        if np.any(visible & ~reference_finite):
            raise ValueError("visible reference points must have finite coordinates")

    reference_points = int(np.count_nonzero(visible))
    if reference_points == 0:
        raise ValueError("at least one reference point must be visible")

    matched = visible & decoded_finite
    decoded_points = int(np.count_nonzero(decoded_finite))
    matched_points = int(np.count_nonzero(matched))
    missed_points = reference_points - matched_points
    false_positives = int(np.count_nonzero(decoded_finite & ~visible))

    if matched_points:
        with np.errstate(over="ignore", invalid="ignore"):
            position_errors = np.linalg.norm(decoded[matched] - reference[matched], axis=-1)
        if not np.isfinite(position_errors).all():
            raise ValueError("nonfinite computed position error")
        mean_position_error = float(np.mean(position_errors))
        p95_position_error = float(np.percentile(position_errors, 95))
        tolerance_hits = int(np.count_nonzero(position_errors <= tolerance))
    else:
        mean_position_error = None
        p95_position_error = None
        tolerance_hits = 0

    if frame_count >= 2:
        adjacent = visible[:-1] & visible[1:] & decoded_finite[:-1] & decoded_finite[1:]
        if np.any(adjacent):
            delta_grid = frame_deltas[:, np.newaxis, np.newaxis]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                reference_velocity = np.diff(reference, axis=0) / delta_grid
                decoded_velocity = np.diff(decoded, axis=0) / delta_grid
                velocity_errors = np.linalg.norm(
                    (decoded_velocity - reference_velocity)[adjacent], axis=-1
                )
            if not np.isfinite(velocity_errors).all():
                raise ValueError("nonfinite computed velocity error")
            mean_velocity_error = float(np.mean(velocity_errors))
            velocity_pairs = int(velocity_errors.size)
        else:
            mean_velocity_error = None
            velocity_pairs = 0
    else:
        mean_velocity_error = None
        velocity_pairs = 0

    return LandmarkTrackScore(
        mean_position_error_px=mean_position_error,
        p95_position_error_px=p95_position_error,
        mean_velocity_error_px_per_s=mean_velocity_error,
        detection_precision=(matched_points / decoded_points if decoded_points else 0.0),
        detection_recall=matched_points / reference_points,
        tolerance_hit_rate=tolerance_hits / reference_points,
        reference_points=reference_points,
        decoded_points=decoded_points,
        matched_points=matched_points,
        missed_points=missed_points,
        false_positives=false_positives,
        velocity_pairs=velocity_pairs,
        tolerance_px=tolerance,
    )
