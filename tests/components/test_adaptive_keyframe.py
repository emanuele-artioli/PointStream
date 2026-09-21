"""Unit tests for foreground adaptive keyframing."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.appearance import (
    AdaptiveCropConfig,
    AdaptiveKeyframeResult,
    AdaptiveKeyframeSelector,
)


def _make_keypoints(offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    """Generate 17 COCO-style keypoints centered within a 100x200 actor box."""
    kpts = np.array(
        [
            [50.0 + offset_x, 20.0 + offset_y],   # 0: nose
            [45.0 + offset_x, 15.0 + offset_y],   # 1: left eye
            [55.0 + offset_x, 15.0 + offset_y],   # 2: right eye
            [40.0 + offset_x, 20.0 + offset_y],   # 3: left ear
            [60.0 + offset_x, 20.0 + offset_y],   # 4: right ear
            [35.0 + offset_x, 50.0 + offset_y],   # 5: left shoulder
            [65.0 + offset_x, 50.0 + offset_y],   # 6: right shoulder
            [30.0 + offset_x, 80.0 + offset_y],   # 7: left elbow
            [70.0 + offset_x, 80.0 + offset_y],   # 8: right elbow
            [25.0 + offset_x, 110.0 + offset_y],  # 9: left wrist
            [75.0 + offset_x, 110.0 + offset_y],  # 10: right wrist
            [40.0 + offset_x, 120.0 + offset_y],  # 11: left hip
            [60.0 + offset_x, 120.0 + offset_y],  # 12: right hip
            [38.0 + offset_x, 160.0 + offset_y],  # 13: left knee
            [62.0 + offset_x, 160.0 + offset_y],  # 14: right knee
            [36.0 + offset_x, 195.0 + offset_y],  # 15: left ankle
            [64.0 + offset_x, 195.0 + offset_y],  # 16: right ankle
        ],
        dtype=np.float64,
    )
    return kpts


def _make_synth_frame(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """Generate a smooth synthetic test frame (resembles natural image textures)."""
    y, x = np.mgrid[0:h, 0:w]
    shift = (seed * 3) % 256
    r = ((x + shift) * 2) % 256
    g = ((y + shift) * 1) % 256
    b = ((x + y + shift) * 2) % 256
    return np.stack([r, g, b], axis=2).astype(np.uint8)


def test_config_defaults() -> None:
    """Verify AdaptiveCropConfig default parameters adhere to spec."""
    cfg = AdaptiveCropConfig()
    assert cfg.max_crops == 3
    assert cfg.max_total_bytes == 12000
    assert cfg.min_frame_interval == 24
    assert cfg.oks_threshold == 0.80
    assert cfg.format == "webp"
    assert cfg.quality == 75
    assert cfg.downscale == 1.0


def test_single_keyframe_when_pose_does_not_drift() -> None:
    """When pose is stable across 60 frames, only frame 0 is selected as keyframe."""
    selector = AdaptiveKeyframeSelector()
    num_frames = 60
    base_kpts = _make_keypoints(0.0, 0.0)
    bbox = (10, 10, 110, 210)

    # Slight jitter that keeps OKS well above 0.80 threshold
    frames = [_make_synth_frame(seed=i) for i in range(num_frames)]
    keypoints = [base_kpts + (0.1 * (i % 3)) for i in range(num_frames)]
    bboxes = [bbox] * num_frames

    result = selector.select_and_encode(frames, keypoints, bboxes)

    assert isinstance(result, AdaptiveKeyframeResult)
    assert result.keyframe_indices == [0]
    assert len(result.crops) == 1
    assert len(result.payloads) == 1
    assert len(result.descriptors) == 1
    assert result.total_bytes == len(result.payloads[0])
    assert result.total_bytes <= selector.config.max_total_bytes
    assert result.frame_to_crop_idx == [0] * num_frames
    assert result.descriptors[0].format == "webp"
    assert result.descriptors[0].quality == 75


def test_adaptive_keyframe_triggered_when_pose_drifts() -> None:
    """Adaptive keyframe is triggered when OKS drops below threshold and cooldown passed."""
    cfg = AdaptiveCropConfig(min_frame_interval=24, oks_threshold=0.80)
    selector = AdaptiveKeyframeSelector(cfg)
    num_frames = 60
    bbox = (10, 10, 110, 210)

    pose_a = _make_keypoints(0.0, 0.0)
    pose_b = _make_keypoints(40.0, 40.0)  # Significant shift causing OKS < 0.01

    frames = [_make_synth_frame(seed=i) for i in range(num_frames)]
    keypoints = [pose_a if i < 25 else pose_b for i in range(num_frames)]
    bboxes = [bbox] * num_frames

    result = selector.select_and_encode(frames, keypoints, bboxes)

    assert result.keyframe_indices == [0, 25]
    assert len(result.crops) == 2
    assert len(result.payloads) == 2
    assert len(result.descriptors) == 2
    assert result.total_bytes == sum(len(p) for p in result.payloads)
    assert result.total_bytes <= cfg.max_total_bytes

    # Mapping checks
    assert result.frame_to_crop_idx[:25] == [0] * 25
    assert result.frame_to_crop_idx[25:] == [1] * 35


def test_minimum_interval_cooldown_enforcement() -> None:
    """Drift before cooldown elapsed does NOT trigger a keyframe; triggers only at >= cooldown."""
    min_interval = 24
    cfg = AdaptiveCropConfig(min_frame_interval=min_interval, oks_threshold=0.80)
    selector = AdaptiveKeyframeSelector(cfg)
    num_frames = 40
    bbox = (10, 10, 110, 210)

    pose_a = _make_keypoints(0.0, 0.0)
    pose_b = _make_keypoints(40.0, 40.0)

    # Pose drifts early at frame 10 (which is < 0 + 24), persists until end
    frames = [_make_synth_frame(seed=i) for i in range(num_frames)]
    keypoints = [pose_a if i < 10 else pose_b for i in range(num_frames)]
    bboxes = [bbox] * num_frames

    result = selector.select_and_encode(frames, keypoints, bboxes)

    # Frame 10 is rejected by cooldown; frame 24 is the earliest eligible frame
    assert result.keyframe_indices == [0, 24]
    assert 10 not in result.keyframe_indices
    assert len(result.crops) == 2
    assert result.frame_to_crop_idx[:24] == [0] * 24
    assert result.frame_to_crop_idx[24:] == [1] * (num_frames - 24)


def test_rate_limiter_strict_byte_budget_with_severe_pose_changes() -> None:
    """Strict wire budget is respected even under continuous severe pose drift."""
    tight_budget = 4000
    cfg = AdaptiveCropConfig(
        max_crops=5,
        max_total_bytes=tight_budget,
        min_frame_interval=10,
        oks_threshold=0.80,
    )
    selector = AdaptiveKeyframeSelector(cfg)
    num_frames = 60
    bbox = (0, 0, 80, 80)

    # Pose changes drastically every 10 frames
    frames = [_make_synth_frame(h=100, w=100, seed=i) for i in range(num_frames)]
    keypoints = [_make_keypoints(float(i * 15), float(i * 15)) for i in range(num_frames)]
    bboxes = [bbox] * num_frames

    result = selector.select_and_encode(frames, keypoints, bboxes)

    # Guarantee total bytes <= max_total_bytes
    assert result.total_bytes <= tight_budget
    assert result.total_bytes == sum(len(p) for p in result.payloads)
    # Check that frame_to_crop_idx remains valid
    for crop_idx in result.frame_to_crop_idx:
        assert 0 <= crop_idx < len(result.crops)


def test_quality_step_down_to_fit_remaining_budget() -> None:
    """When a new crop exceeds remaining budget at quality 75, quality is stepped down to fit."""
    frame_a = _make_synth_frame(h=100, w=100, seed=1)
    frame_b = _make_synth_frame(h=100, w=100, seed=2)
    bbox = (0, 0, 80, 80)
    crop_b = frame_b[0:80, 0:80]

    # Measure exact payload sizes of crop_b
    sel_probe = AdaptiveKeyframeSelector()
    _, p_q75 = sel_probe.encoder.encode(crop_b, quality=75)
    _, p_q20 = sel_probe.encoder.encode(crop_b, quality=20)
    size_q75 = len(p_q75)
    size_q20 = len(p_q20)

    # Frame 0 crop size
    crop_a = frame_a[0:80, 0:80]
    _, p_a = sel_probe.encoder.encode(crop_a, quality=75)
    size_a = len(p_a)

    # Set budget so crop_b cannot fit at q75, but easily fits at lower quality
    remaining_for_b = (size_q75 + size_q20) // 2
    allowed_budget = size_a + remaining_for_b
    assert allowed_budget < size_a + size_q75
    assert allowed_budget > size_a + size_q20

    cfg = AdaptiveCropConfig(
        max_crops=3,
        max_total_bytes=allowed_budget,
        min_frame_interval=24,
        quality=75,
    )
    selector = AdaptiveKeyframeSelector(cfg)

    num_frames = 50
    pose_a = _make_keypoints(0.0, 0.0)
    pose_b = _make_keypoints(50.0, 50.0)

    frames = [frame_a if i < 25 else frame_b for i in range(num_frames)]
    keypoints = [pose_a if i < 25 else pose_b for i in range(num_frames)]
    bboxes = [bbox] * num_frames

    result = selector.select_and_encode(frames, keypoints, bboxes)

    assert result.keyframe_indices == [0, 25]
    assert len(result.crops) == 2
    # Quality for second crop was stepped down to fit budget
    assert result.descriptors[1].quality < 75
    assert result.total_bytes <= allowed_budget
    assert result.total_bytes == sum(len(p) for p in result.payloads)


def test_max_crops_limit_stops_adding_crops() -> None:
    """When max_crops is reached, no further keyframes are added even if pose drifts."""
    cfg = AdaptiveCropConfig(max_crops=2, min_frame_interval=10, oks_threshold=0.80)
    selector = AdaptiveKeyframeSelector(cfg)
    num_frames = 50
    bbox = (0, 0, 60, 60)

    # Pose drifts at frame 12 and frame 28
    pose_0 = _make_keypoints(0.0, 0.0)
    pose_1 = _make_keypoints(30.0, 30.0)
    pose_2 = _make_keypoints(60.0, 60.0)

    frames = [_make_synth_frame(h=80, w=80, seed=i) for i in range(num_frames)]
    keypoints = []
    for i in range(num_frames):
        if i < 12:
            keypoints.append(pose_0)
        elif i < 28:
            keypoints.append(pose_1)
        else:
            keypoints.append(pose_2)
    bboxes = [bbox] * num_frames

    result = selector.select_and_encode(frames, keypoints, bboxes)

    # Only 2 crops allowed: initial frame 0 and first drift at frame 12
    assert result.keyframe_indices == [0, 12]
    assert len(result.crops) == 2
    # Frames after 28 still map to crop 1 (active crop)
    assert result.frame_to_crop_idx[28] == 1
    assert result.frame_to_crop_idx[-1] == 1


def test_mapping_from_frame_index_to_active_crop_index() -> None:
    """Every frame in [0, T-1] maps to the active keyframe crop index."""
    cfg = AdaptiveCropConfig(min_frame_interval=20, oks_threshold=0.80)
    selector = AdaptiveKeyframeSelector(cfg)
    num_frames = 70
    bbox = (10, 10, 70, 70)

    frames = [_make_synth_frame(seed=i) for i in range(num_frames)]
    keypoints = [
        _make_keypoints(float(0 if i < 25 else (40 if i < 50 else 80)), 0.0)
        for i in range(num_frames)
    ]
    bboxes = [bbox] * num_frames

    result = selector.select_and_encode(frames, keypoints, bboxes)

    assert len(result.frame_to_crop_idx) == num_frames
    assert result.keyframe_indices == [0, 25, 50]
    assert len(result.crops) == 3

    # Check mapping intervals
    for t in range(num_frames):
        expected_crop = 0 if t < 25 else (1 if t < 50 else 2)
        assert result.frame_to_crop_idx[t] == expected_crop


def test_crop_extraction_with_bbox_and_none_bbox() -> None:
    """Crops respect bounding boxes and fall back to full frames when bbox is None."""
    selector = AdaptiveKeyframeSelector()
    frame = np.zeros((100, 150, 3), dtype=np.uint8)

    # With bbox
    crop = selector._extract_crop(frame, (10, 20, 60, 80))
    assert crop.shape == (60, 50, 3)

    # With None bbox
    crop_none = selector._extract_crop(frame, None)
    assert crop_none.shape == (100, 150, 3)

    # Clamping out-of-bounds bbox
    crop_clamped = selector._extract_crop(frame, (-10, -20, 200, 300))
    assert crop_clamped.shape == (100, 150, 3)


def test_missing_keypoints_does_not_trigger_keyframe() -> None:
    """Frames with None keypoints do not cause spurious keyframe updates."""
    cfg = AdaptiveCropConfig(min_frame_interval=10)
    selector = AdaptiveKeyframeSelector(cfg)
    num_frames = 30
    bbox = (0, 0, 50, 50)

    pose = _make_keypoints(0.0, 0.0)
    frames = [_make_synth_frame(seed=i) for i in range(num_frames)]
    # Frames 10..25 have no detection (None keypoints)
    keypoints: list[np.ndarray | None] = [
        pose if (i < 10 or i >= 25) else None for i in range(num_frames)
    ]
    bboxes = [bbox] * num_frames

    result = selector.select_and_encode(frames, keypoints, bboxes)

    # No drift occurred between frame 0 and frame 25+, and missing frames didn't trigger
    assert result.keyframe_indices == [0]
    assert len(result.crops) == 1
    assert result.frame_to_crop_idx == [0] * num_frames


def test_empty_sequence() -> None:
    """Empty sequence returns empty result with total_bytes = 0."""
    selector = AdaptiveKeyframeSelector()
    result = selector.select_and_encode([], [], [])
    assert result.keyframe_indices == []
    assert result.crops == []
    assert result.payloads == []
    assert result.descriptors == []
    assert result.total_bytes == 0
    assert result.frame_to_crop_idx == []


def test_mismatched_sequence_lengths_raises() -> None:
    """Mismatched sequence lengths raise ValueError."""
    selector = AdaptiveKeyframeSelector()
    frames = [_make_synth_frame()]
    keypoints = [_make_keypoints()]
    with pytest.raises(ValueError, match="Mismatched input lengths"):
        selector.select_and_encode(frames, keypoints, [])
