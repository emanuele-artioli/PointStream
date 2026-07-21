"""Guards that evaluation runs the *decoder's* inference path, not its own copy.

`scripts/eval_checkpoint.py` used to reimplement inference. It diverged: the
eval path built a `StableDiffusionControlNetPipeline` (text-to-image from pure
noise, reference frame explicitly unused) while the decoder ran
`StableDiffusionControlNetImg2ImgPipeline` seeded from the reference crop. The
G2 campaign then scored ControlNet at PSNR 9.76 / VMAF 0.11 -- the arithmetic
expectation when an unconditional sample is compared against a specific target,
and not a statement about the model at all.

These tests pin the three things that made that failure possible:
  1. eval and the decoder construct the *same* strategy for a given backend;
  2. the crop-local keypoint join is positional (the three per-track artefacts
     use different frame-id conventions);
  3. the BGR/RGB conversions at the eval boundary round-trip, since the
     `reference_crop_tensor` convention is BGR and getting it wrong silently
     swaps red and blue in the appearance cue.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.eval_checkpoint import (
    ARCH_TO_BACKEND,
    bgr_uint8_to_rgb01,
    build_eval_strategy,
    clip_keypoints,
    clip_track_index,
    rgb01_to_bgr_uint8_tensor,
)
from src.decoder.genai_compositor import build_genai_strategy
from src.shared.config import PointstreamConfig


# ---------------------------------------------------------------------------
# 1. One code path: eval and the decoder build the same strategy
# ---------------------------------------------------------------------------


def test_eval_and_decoder_build_the_same_strategy_class(tmp_path: Path) -> None:
    """Every eval arch must resolve to the strategy the decoder would construct.

    Uses the mock backend for the ControlNet family so no weights are needed;
    the point under test is the construction path, not the sampler.
    """
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")

    for arch, backend in ARCH_TO_BACKEND.items():
        eval_strategy = build_eval_strategy(arch, checkpoint)

        decoder_config = PointstreamConfig()
        decoder_config.genai_backend = backend
        decoder_config.genai_checkpoint_override = str(checkpoint)
        decoder_strategy = build_genai_strategy(backend, decoder_config)

        assert type(eval_strategy) is type(decoder_strategy), (
            f"arch {arch!r} builds {type(eval_strategy).__name__} in eval but "
            f"{type(decoder_strategy).__name__} in the decoder"
        )


def test_build_eval_strategy_rejects_unknown_config_override(tmp_path: Path) -> None:
    """A typo'd override must fail loudly rather than being silently ignored.

    Silently-dropped knobs are how eval drifts from the decoder in the first
    place.
    """
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"x")
    try:
        build_eval_strategy("pix2pix", checkpoint, {"num_inference_steps": 5})
    except ValueError as exc:
        assert "num_inference_steps" in str(exc)
    else:  # pragma: no cover - the assertion below always fires if we get here
        raise AssertionError("expected ValueError for an unknown config field")


def test_checkpoint_override_must_exist() -> None:
    """A missing checkpoint path must raise, not fall back to assets/weights.

    Falling back would score a *different* checkpoint than the campaign asked
    for and report it under the campaign's variant name.
    """
    from src.decoder.genai_compositor import _resolve_strategy_weight

    config = PointstreamConfig()
    config.genai_checkpoint_override = "/nonexistent/checkpoint.pt"
    try:
        _resolve_strategy_weight(config, "pix2pix_generator.pt")
    except FileNotFoundError as exc:
        assert "nonexistent" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError for a missing override")


# ---------------------------------------------------------------------------
# 2. The keypoint join is positional, not by frame_id
# ---------------------------------------------------------------------------


def _write_track_with_gap(root: Path) -> dict:
    """Track whose absolute frame ids have a gap, mirroring the real dataset.

    Real tracks look like 493, 498, 499, ... while `_keypoints.json` indexes
    0, 1, 2, ... -- joining by frame_id instead of position silently reads the
    wrong pose.
    """
    clip = {"video": "v", "scene": "scene_000", "track": "track_0001", "frame_ids": [498, 499]}
    seg = root / "v" / "segmentations" / "scene_000"
    color = seg / "track_0001"
    color.mkdir(parents=True, exist_ok=True)
    absolute_ids = [493, 498, 499]
    for fid in absolute_ids:
        (color / f"frame_{fid:06d}.png").write_bytes(b"fake")

    # keypoints.json is 0-indexed by position; make each pose identifiable.
    keypoints = [
        {"frame_id": idx, "keypoints": [[float(idx), 0.0, 1.0]] * 18}
        for idx, _ in enumerate(absolute_ids)
    ]
    (seg / "track_0001_keypoints.json").write_text(json.dumps(keypoints))
    return clip


def test_clip_track_index_maps_absolute_ids_to_positions(tmp_path: Path) -> None:
    clip = _write_track_with_gap(tmp_path)
    assert clip_track_index(tmp_path, clip) == {493: 0, 498: 1, 499: 2}


def test_clip_keypoints_joins_positionally_across_a_frame_id_gap(tmp_path: Path) -> None:
    clip = _write_track_with_gap(tmp_path)
    poses = clip_keypoints(tmp_path, clip, [498, 499])

    assert len(poses) == 2
    assert poses[0].shape == (18, 3)  # Shape: [Keypoints, (x, y, confidence)]
    # Frame 498 is position 1 in the track, so it must carry pose 1 -- not
    # pose 498, and not pose 0.
    assert poses[0][0][0] == 1.0
    assert poses[1][0][0] == 2.0


# ---------------------------------------------------------------------------
# 3. BGR/RGB conversions at the eval boundary
# ---------------------------------------------------------------------------


def test_rgb01_to_bgr_uint8_tensor_swaps_channels() -> None:
    """Strategies expect BGR uint8 (it comes from cv2.imdecode in the encoder)."""
    rgb = torch.zeros(3, 2, 2)  # Shape: [3, H, W]
    rgb[0] = 1.0  # pure red in RGB

    bgr = rgb01_to_bgr_uint8_tensor(rgb)  # Shape: [3, H, W] BGR uint8

    assert bgr.dtype == torch.uint8
    assert torch.all(bgr[2] == 255), "red must land in the BGR tensor's last channel"
    assert torch.all(bgr[0] == 0)


def test_bgr_uint8_to_rgb01_round_trips_colour() -> None:
    """A square BGR crop must come back as the same colour in RGB [0,1]."""
    bgr = torch.zeros(3, 4, 4, dtype=torch.uint8)  # Shape: [3, H, W]
    bgr[2] = 255  # red, in BGR channel order

    rgb01 = bgr_uint8_to_rgb01(bgr, size=4)  # Shape: [3, 4, 4] RGB float

    assert rgb01.shape == (3, 4, 4)
    assert torch.allclose(rgb01[0], torch.ones(4, 4), atol=1e-6), "red must return in RGB channel 0"
    assert torch.allclose(rgb01[1], torch.zeros(4, 4), atol=1e-6)


def test_bgr_uint8_to_rgb01_pads_non_square_crops_without_stretching() -> None:
    """Predictions must get the same pad-to-square geometry as the ground truth
    (`load_image_rgb01`), or every metric compares misaligned images."""
    bgr = torch.full((3, 8, 4), 255, dtype=torch.uint8)  # Shape: [3, H=8, W=4]

    rgb01 = bgr_uint8_to_rgb01(bgr, size=8)  # Shape: [3, 8, 8]

    assert rgb01.shape == (3, 8, 8)
    # Padding is black and sits on the left/right of a tall crop.
    assert float(rgb01[:, :, 0].max()) == 0.0
    assert float(rgb01[:, :, -1].max()) == 0.0
    assert float(rgb01[:, :, 4].max()) == 1.0


def test_reference_round_trip_preserves_colour_end_to_end() -> None:
    """rgb01 -> BGR (into the strategy) -> rgb01 (out) must be identity on colour.

    This is the invariant whose violation put a red/blue-swapped reference into
    pix2pix and SPADE in the decoder.
    """
    rgb = torch.zeros(3, 6, 6)  # Shape: [3, H, W]
    rgb[0] = 0.8  # distinctive red
    rgb[2] = 0.2

    bgr = rgb01_to_bgr_uint8_tensor(rgb)
    back = bgr_uint8_to_rgb01(bgr, size=6)

    assert torch.allclose(back, rgb, atol=2 / 255), "colour must survive the eval boundary"
    assert np.isclose(float(back[0].mean()), 0.8, atol=2 / 255)
