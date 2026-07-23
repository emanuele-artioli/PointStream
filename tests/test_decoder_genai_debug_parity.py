"""Report 10 Phase 5.1(c): regression coverage for the decoder-side GenAI
debug-artifact wiring fix.

Diagnosis (see 67a9ea6275d3d9785ce57026/RESEARCH_LOG.md, Bug registry #5; Phase
5.1(c) finding): a real full-match profile found `decode/genai_baseline`
running 2.16x slower than the encoder's `encode_chunk/residual/genai_baseline`
for the *same* 60 frames and the same canny-controlnet engine. The cause was
not extra GenAI compute on the decoder side -- both sides run the same
per-frame `compositor.process()` loop -- it was that
`DecoderRenderer._render_genai_baseline` unconditionally wired
`self.config.debug_artifact_dir` (always set by `main.py` for every real run)
into the GenAI compositor's `debug_dir` argument regardless of
`disable_debug_artifacts`, while the encoder-side call in
`ResidualCalculator._process_residuals` never passes a debug dir at all in
the real DAG path. So the decoder was unconditionally writing five PNG debug
artifacts per actor per frame (including two full-resolution frame dumps) to
disk while the encoder wrote none -- an I/O asymmetry, not a compute one.

These tests are the two-part tripwire the report calls for:
1. A correctness check: whether debug artifacts are written or not must
   never change the actual generated pixels (a Residual Guarantee concern --
   if disabling debug I/O silently changed frames, encoder/decoder could
   diverge whenever their debug settings differ).
2. A regression check on the fix itself: the decoder must stop wiring a
   debug dir into the compositor when `disable_debug_artifacts` is set,
   matching the `BackgroundModeler._debug_artifacts_enabled()` convention
   used elsewhere in the encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.decoder.decoder_renderer import DecoderRenderer, _ClientActorState
from src.decoder.genai_compositor import DiffusersCompositor
from src.shared.config import PointstreamConfig


class _SpyCompositor:
    """Records the `debug_dir` each `process()` call actually received."""

    def __init__(self) -> None:
        self.debug_dirs: list[Any] = []

    def process(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        warped_background_frame: torch.Tensor,
        actor_identity: str | None = None,
        metadata_mask: Any = None,
        metadata_bbox: Any = None,
        debug_dir: str | Path | None = None,
        frame_idx: int | None = None,
    ) -> torch.Tensor:
        self.debug_dirs.append(debug_dir)
        return warped_background_frame


def _build_single_actor_state(num_frames: int) -> dict[int, _ClientActorState]:
    reference_crop = torch.zeros((3, 32, 16), dtype=torch.uint8)
    dense_pose = torch.zeros((num_frames, 18, 3), dtype=torch.float32)
    return {
        1: _ClientActorState(
            track_id=1,
            object_id="player_1",
            reference_crop_tensor=reference_crop,
            dense_pose_tensor=dense_pose,
        )
    }


def test_decoder_only_wires_debug_dir_into_compositor_when_enabled(test_run_artifacts_dir: Path) -> None:
    debug_root = test_run_artifacts_dir / "genai_debug_parity"

    frame_tensor = torch.zeros((2, 3, 16, 16), dtype=torch.uint8)

    # PointstreamConfig.__post_init__ derives disable_debug_artifacts from
    # log_level (only "debug" keeps it False) -- log_level="info" is how a
    # real run actually ends up with disable_debug_artifacts=True.
    renderer_disabled = DecoderRenderer(
        output_root=test_run_artifacts_dir,
        config=PointstreamConfig(debug_artifact_dir=str(debug_root), log_level="info"),
    )
    assert renderer_disabled.config.disable_debug_artifacts is True
    spy_disabled = _SpyCompositor()
    renderer_disabled._genai_compositor = spy_disabled  # type: ignore[assignment]
    renderer_disabled._actor_state = _build_single_actor_state(num_frames=2)
    renderer_disabled._render_genai_baseline(frame_tensor)

    assert spy_disabled.debug_dirs
    assert all(debug_dir is None for debug_dir in spy_disabled.debug_dirs)

    renderer_enabled = DecoderRenderer(
        output_root=test_run_artifacts_dir,
        config=PointstreamConfig(debug_artifact_dir=str(debug_root), log_level="debug"),
    )
    assert renderer_enabled.config.disable_debug_artifacts is False
    spy_enabled = _SpyCompositor()
    renderer_enabled._genai_compositor = spy_enabled  # type: ignore[assignment]
    renderer_enabled._actor_state = _build_single_actor_state(num_frames=2)
    renderer_enabled._render_genai_baseline(frame_tensor)

    assert spy_enabled.debug_dirs
    assert all(debug_dir == str(debug_root) for debug_dir in spy_enabled.debug_dirs)


def test_disabling_debug_artifacts_does_not_change_generated_pixels(tmp_path: Path) -> None:
    """Residual Guarantee tripwire: writing (or skipping) debug PNGs is a
    pure side effect of `DiffusersCompositor.process()` -- it must never
    influence the returned, transmitted/reconstructed pixels."""
    reference_crop = torch.zeros((3, 64, 32), dtype=torch.uint8)
    dense_dwpose = torch.zeros((18, 3), dtype=torch.float32)
    dense_dwpose[0] = torch.tensor([20.0, 20.0, 0.9])
    dense_dwpose[1] = torch.tensor([40.0, 20.0, 0.9])
    dense_dwpose[2] = torch.tensor([30.0, 50.0, 0.9])
    warped_background = torch.zeros((3, 128, 96), dtype=torch.uint8)

    config = PointstreamConfig(
        genai_backend="mock-caption-controlnet",
        compositing_mask_mode="pose-heuristic-mask",
        postgen_segmenter_backend="heuristic",
        allow_auto_model_download=False,
    )

    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()

    def _run(debug_dir_arg: Path | None) -> torch.Tensor:
        compositor = DiffusersCompositor(
            backend="mock-caption-controlnet",
            seed=42,
            device="cpu",
            config=config,
        )
        compositor.set_debug_stage("decoder")
        return compositor.process(
            reference_crop_tensor=reference_crop,
            dense_dwpose_tensor=dense_dwpose,
            warped_background_frame=warped_background,
            actor_identity="actor_0",
            debug_dir=debug_dir_arg,
            frame_idx=0,
        )

    output_without_debug = _run(None)
    assert list(debug_dir.iterdir()) == []  # no artifacts written when debug_dir is None

    output_with_debug = _run(debug_dir)
    written_files = list(debug_dir.rglob("*.png"))
    assert len(written_files) > 0  # artifacts were written this time

    assert torch.equal(output_without_debug, output_with_debug)
