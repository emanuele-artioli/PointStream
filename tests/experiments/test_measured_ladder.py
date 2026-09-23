"""Tests for experiments.modular.measured_ladder."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.modular.measured_ladder import (
    MeasuredInputError,
    beats,
    load_sequence,
    measure_rungs,
    score_regions,
    write_comparison_strip,
)


def _moving_block_clip(
    *,
    fg_color: tuple[int, int, int] = (20, 180, 20),
) -> tuple[np.ndarray, np.ndarray]:
    frames = np.full((2, 32, 32, 3), (180, 20, 20), dtype=np.uint8)
    mask = np.zeros((2, 32, 32), dtype=bool)
    frames[0, 4:12, 4:12] = fg_color
    mask[0, 4:12, 4:12] = True
    frames[1, 16:24, 16:24] = fg_color
    mask[1, 16:24, 16:24] = True
    return frames, mask


def test_score_regions_one_pixel_difference() -> None:
    reference = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    reconstruction = reference.copy()
    reconstruction[0, 0, 0] = 255
    mask = np.zeros((1, 8, 8), dtype=bool)
    mask[0, 0, 0] = True

    overall, psnr_fg, psnr_bg, weighted = score_regions(
        reference,
        reconstruction,
        mask,
        fg_weight=0.7,
        bg_weight=0.3,
    )

    assert overall == pytest.approx(10.0 * math.log10(64), abs=1e-6)
    assert psnr_fg == 0.0
    assert psnr_bg == math.inf
    assert weighted is None


def test_beats_comparison() -> None:
    assert beats(100, None) is False
    assert beats(100, 100) is False
    assert beats(99, 100) is True
    assert beats(100, 99) is False


def test_measure_rungs_four_rungs_and_byte_sum() -> None:
    frames, mask = _moving_block_clip()
    rungs = measure_rungs(frames, mask)
    expected_ids = [
        "C0_compact_baseline",
        "C1_adaptive_keyframes",
        "C2_unified_video_residual",
        "C3_foreground_video_residual",
        "C3_background_video_residual",
    ]
    assert [rung.rung_id for rung in rungs] == expected_ids

    for rung in rungs:
        assert rung.total_bytes == (
            rung.bytes_background
            + rung.bytes_appearance
            + rung.bytes_metadata
            + rung.bytes_residual
            + rung.bytes_container
        )
        assert rung.bytes_container == 0
        assert rung.pose_oks is None
        assert rung.reconstruction.shape == frames.shape

        overall, psnr_fg, psnr_bg, _weighted = score_regions(
            frames,
            rung.reconstruction,
            mask,
            fg_weight=0.7,
            bg_weight=0.3,
        )
        assert rung.psnr_overall == overall
        assert rung.psnr_fg == psnr_fg
        assert rung.psnr_bg == psnr_bg

    c0 = rungs[0]
    # VVC intra QP 32 on a 32px plate lands just under 30 dB. The bound only
    # checks that the background is still the source, not a blank fill.
    assert c0.psnr_bg > 28.0


def test_different_fg_colour_changes_c0_appearance_pair() -> None:
    frames_a, mask_a = _moving_block_clip(fg_color=(20, 180, 20))
    frames_b, mask_b = _moving_block_clip(fg_color=(20, 20, 180))
    c0_a = measure_rungs(frames_a, mask_a)[0]
    c0_b = measure_rungs(frames_b, mask_b)[0]
    pair_a = (c0_a.bytes_appearance, round(c0_a.psnr_fg, 3))
    pair_b = (c0_b.bytes_appearance, round(c0_b.psnr_fg, 3))
    assert pair_a != pair_b


def test_write_comparison_strip_png_not_constant() -> None:
    frames, mask = _moving_block_clip()
    rungs = measure_rungs(frames, mask)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "strip.png"
        path = write_comparison_strip(
            frames,
            rungs[0].reconstruction,
            out,
            summary="test",
        )
        assert path.is_file()
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Ground-truth panel is the left panel under the banner.
        banner_h = 30
        panel_w = frames.shape[2]
        gt_panel = rgb[banner_h : banner_h + frames.shape[1], :panel_w]
        assert float(np.std(gt_panel)) > 0.0


def test_load_sequence_reads_pngs_and_mask() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames_dir = root / "frames"
        frames_dir.mkdir()
        frame0 = np.full((16, 16, 3), 10, dtype=np.uint8)
        frame1 = np.full((16, 16, 3), 20, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "a.png"), cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frames_dir / "b.png"), cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))
        mask = np.zeros((2, 16, 16), dtype=bool)
        mask[0, 1, 1] = True
        mask_path = root / "mask.npy"
        np.save(mask_path, mask)

        loaded_frames, loaded_mask = load_sequence(frames_dir, mask_path, n_frames=2)
        assert loaded_frames.shape == (2, 16, 16, 3)
        np.testing.assert_array_equal(loaded_frames[0], frame0)
        np.testing.assert_array_equal(loaded_frames[1], frame1)
        np.testing.assert_array_equal(loaded_mask, mask)


def test_load_sequence_too_few_images() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames_dir = root / "frames"
        frames_dir.mkdir()
        frame0 = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "only.png"), frame0)
        mask_path = root / "mask.npy"
        np.save(mask_path, np.zeros((2, 8, 8), dtype=bool))
        with pytest.raises(MeasuredInputError):
            load_sequence(frames_dir, mask_path, n_frames=2)


def test_mask_shape_mismatch_raises() -> None:
    frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    bad_mask = np.zeros((7, 7), dtype=bool)
    with pytest.raises(MeasuredInputError):
        score_regions(frames, frames, bad_mask, fg_weight=0.7, bg_weight=0.3)
    with pytest.raises(MeasuredInputError):
        measure_rungs(frames, bad_mask)


def test_beats_negative_raises() -> None:
    with pytest.raises(MeasuredInputError):
        beats(-1, 10)


def test_score_regions_weight_validation() -> None:
    frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    mask = np.zeros((1, 4, 4), dtype=bool)
    with pytest.raises(MeasuredInputError):
        score_regions(frames, frames, mask, fg_weight=0.2, bg_weight=0.2)
    overall, _fg, _bg, weighted = score_regions(
        frames, frames, mask, fg_weight=0.7, bg_weight=0.3
    )
    assert overall == math.inf
    assert weighted is None


def test_flat_red_survives_yuv420_roundtrip() -> None:
    """A constant red frame must come back red. Luma-only grey does not."""
    from experiments.modular.measured_ladder import _rgb_to_yuv420, _yuv420_to_rgb

    red = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255
    luma, chroma = _rgb_to_yuv420(red)
    restored = _yuv420_to_rgb(luma, chroma)
    mse_red = float(np.mean((restored.astype(np.float64) - red.astype(np.float64)) ** 2))
    grey = np.repeat(luma[..., None], 3, axis=-1)
    mse_grey = float(np.mean((grey.astype(np.float64) - red.astype(np.float64)) ** 2))
    assert mse_red < 4.0
    assert mse_red < mse_grey
