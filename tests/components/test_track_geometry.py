"""Hand-computed controls for court/ball landmark track measurements.

Behaviour
1. Exact tracks have zero spatial and temporal errors.
2. A known constant shift is measured in display pixels and does not vanish
   from the position metric merely because velocity is unchanged.
3. Missing predictions reduce recall and tolerance-hit rate.
4. Detections where the reference is invisible reduce precision.
5. Velocity error uses actual, potentially irregular timestamps.
6. A missing interval is not bridged when computing velocity.

Plausible misuse
7. Mismatched track dimensions, partial-NaN coordinate pairs, invalid PTS,
   nonpositive tolerance, and an empty visible-reference set are rejected.

Deliberately not tested: RGB landmark detection/association, tennis court
annotation quality, ball contact events, perceptual metrics, or composite
weights. They require separate instruments and calibration evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.components.metrics.track_geometry import score_landmark_tracks


def test_exact_tracks_have_zero_position_and_velocity_error() -> None:
    reference = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[3.0, 1.0]]], dtype=np.float64)
    result = score_landmark_tracks(
        reference,
        reference.copy(),
        timestamps_s=np.array([0.0, 1.0, 2.0]),
        tolerance_px=1.0,
    )

    assert result.mean_position_error_px == pytest.approx(0.0)
    assert result.p95_position_error_px == pytest.approx(0.0)
    assert result.mean_velocity_error_px_per_s == pytest.approx(0.0)
    assert result.detection_precision == pytest.approx(1.0)
    assert result.detection_recall == pytest.approx(1.0)
    assert result.tolerance_hit_rate == pytest.approx(1.0)
    assert result.velocity_pairs == 2


def test_constant_shift_reports_pixel_error_even_when_velocity_matches() -> None:
    reference = np.array([[[0.0, 0.0]], [[2.0, 0.0]]], dtype=np.float64)
    decoded = reference + np.array([3.0, 4.0])
    result = score_landmark_tracks(
        reference,
        decoded,
        timestamps_s=np.array([0.0, 2.0]),
        tolerance_px=4.9,
    )

    assert result.mean_position_error_px == pytest.approx(5.0)
    assert result.p95_position_error_px == pytest.approx(5.0)
    assert result.mean_velocity_error_px_per_s == pytest.approx(0.0)
    assert result.tolerance_hit_rate == 0.0


def test_missing_predictions_remain_in_recall_and_hit_rate_denominators() -> None:
    reference = np.zeros((2, 2, 2), dtype=np.float64)
    decoded = reference.copy()
    decoded[:, 1] = np.nan
    result = score_landmark_tracks(
        reference,
        decoded,
        timestamps_s=np.array([0.0, 1.0]),
        tolerance_px=1.0,
    )

    assert result.reference_points == 4
    assert result.decoded_points == 2
    assert result.matched_points == 2
    assert result.missed_points == 2
    assert result.detection_recall == pytest.approx(0.5)
    assert result.detection_precision == pytest.approx(1.0)
    assert result.tolerance_hit_rate == pytest.approx(0.5)


def test_detection_at_reference_invisible_slot_is_a_false_positive() -> None:
    reference = np.array([[[0.0, 0.0], [np.nan, np.nan]], [[1.0, 0.0], [np.nan, np.nan]]])
    decoded = np.array([[[0.0, 0.0], [8.0, 8.0]], [[1.0, 0.0], [np.nan, np.nan]]])
    result = score_landmark_tracks(
        reference,
        decoded,
        timestamps_s=np.array([0.0, 1.0]),
        tolerance_px=1.0,
    )

    assert result.reference_points == 2
    assert result.decoded_points == 3
    assert result.matched_points == 2
    assert result.false_positives == 1
    assert result.detection_precision == pytest.approx(2.0 / 3.0)
    assert result.detection_recall == pytest.approx(1.0)


def test_velocity_error_uses_irregular_timestamps() -> None:
    reference = np.array([[[0.0, 0.0]], [[2.0, 0.0]], [[5.0, 0.0]]])
    decoded = np.array([[[0.0, 0.0]], [[6.0, 0.0]], [[15.0, 0.0]]])
    result = score_landmark_tracks(
        reference,
        decoded,
        timestamps_s=np.array([0.0, 2.0, 5.0]),
        tolerance_px=20.0,
    )

    assert result.mean_velocity_error_px_per_s == pytest.approx(2.0)
    assert result.velocity_pairs == 2


def test_velocity_does_not_bridge_a_missing_frame() -> None:
    reference = np.array([[[0.0, 0.0]], [[np.nan, np.nan]], [[4.0, 0.0]]], dtype=np.float64)
    decoded = reference.copy()
    result = score_landmark_tracks(
        reference,
        decoded,
        timestamps_s=np.array([0.0, 1.0, 2.0]),
        tolerance_px=1.0,
    )

    assert result.mean_position_error_px == pytest.approx(0.0)
    assert result.mean_velocity_error_px_per_s is None
    assert result.velocity_pairs == 0


def test_mismatched_track_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        score_landmark_tracks(
            np.zeros((2, 1, 2)),
            np.zeros((2, 2, 2)),
            timestamps_s=np.array([0.0, 1.0]),
            tolerance_px=1.0,
        )


@pytest.mark.parametrize(
    ("timestamps", "tolerance", "visible"),
    [
        (np.array([0.0, 0.0]), 1.0, None),
        (np.array([0.0, 1.0]), 0.0, None),
        (np.array([0.0, 1.0]), 1.0, np.zeros((2, 1), dtype=bool)),
    ],
)
def test_invalid_timestamps_tolerance_or_visibility_are_rejected(
    timestamps: np.ndarray, tolerance: float, visible: np.ndarray | None
) -> None:
    reference = np.zeros((2, 1, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        score_landmark_tracks(
            reference,
            reference.copy(),
            timestamps_s=timestamps,
            tolerance_px=tolerance,
            reference_visible=visible,
        )


def test_partial_nan_coordinate_pair_is_rejected() -> None:
    reference = np.array([[[0.0, 0.0]]])
    decoded = np.array([[[np.nan, 0.0]]])
    with pytest.raises(ValueError, match="both coordinates"):
        score_landmark_tracks(
            reference,
            decoded,
            timestamps_s=np.array([0.0]),
            tolerance_px=1.0,
        )


def test_finite_coordinates_cannot_produce_infinite_score():
    reference = np.array([[[1e308, 0.0]]])
    decoded = np.array([[[-1e308, 0.0]]])
    with pytest.raises(ValueError, match="computed position"):
        score_landmark_tracks(reference, decoded, timestamps_s=np.array([0.0]), tolerance_px=1.0)


def test_tiny_valid_interval_cannot_produce_infinite_velocity():
    reference = np.zeros((2, 1, 2))
    decoded = np.array([[[0.0, 0.0]], [[1.0, 0.0]]])
    with pytest.raises(ValueError, match="computed velocity"):
        score_landmark_tracks(
            reference, decoded, timestamps_s=np.array([0.0, 1e-320]), tolerance_px=1.0
        )
