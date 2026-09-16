"""E06 serialized transport, honest predictors, ledger, and decode gates."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pytest

from experiments.tier.e03b_persist import DecodeCountError
from experiments.tier.e06_probe import refuse_overwrite
from experiments.tier.e06_transport import (
    PREDICTOR_BBOX_RESIZE,
    PREDICTOR_PER_FRAME,
    background_view,
    client_placements,
    ordinary_composite,
    reconcile_ledger,
    reconstruct_standalone,
    require_exact_count,
    require_predictor,
    serialize_setting,
)
from scripts.background_probe import pack_panorama_side_data
from src.components.background.sidecar import JpegSidecar
from src.components.codec import tools
from src.contracts.codecs import EncodeRequest, RateControl
from src.pipeline.residual.codec import TransmittedResidual, encode_residual_to_bitstream
from src.runner.client import reconstruct_serialized_client


def _toy() -> tuple[np.ndarray, np.ndarray]:
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    masks = np.zeros((2, 32, 32), dtype=bool)
    frames[0, 4:12, 4:12] = (30, 80, 200)
    frames[1, 8:16, 10:18] = (90, 10, 40)
    masks[0, 4:12, 4:12] = True
    masks[1, 8:16, 10:18] = True
    return frames, masks


def _jpeg_background(frames: np.ndarray):
    plate = np.full((32, 32, 3), 18, dtype=np.uint8)
    bitstream = JpegSidecar(quality=60).encode(plate)
    plate = JpegSidecar(quality=60).decode(bitstream)
    homographies = np.stack([np.eye(3, dtype=np.float32) for _ in range(frames.shape[0])])
    side = pack_panorama_side_data(
        homographies, plate_shape=(32, 32), frame_shape=(32, 32), fps=12.0
    )
    view = background_view(
        bitstream=bitstream,
        side=side,
        plate=plate,
        homographies=homographies,
        width=32,
        height=32,
        sidecar_codec="jpeg",
        expected_side_bytes=len(side),
    )
    panorama_b = len(bitstream) + len(side)
    return view, panorama_b


def test_old_predictor_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="not an E06 predictor"):
        require_predictor("paste")
    with pytest.raises(ValueError, match="not an E06 predictor"):
        require_predictor("warped_reference")
    assert require_predictor(PREDICTOR_PER_FRAME) == PREDICTOR_PER_FRAME
    assert require_predictor(PREDICTOR_BBOX_RESIZE) == PREDICTOR_BBOX_RESIZE


def test_bbox_resize_stores_one_reference_payload() -> None:
    frames, masks = _toy()
    _per_pl, per_refs, per_f = client_placements(frames, masks, PREDICTOR_PER_FRAME)
    box_pl, box_refs, box_f = client_placements(frames, masks, PREDICTOR_BBOX_RESIZE)
    assert per_refs == {}
    assert len(box_refs) == 1
    assert box_f < per_f
    assert all(item.encoded_crop is not None for item in _per_pl)
    assert all(item.encoded_crop is None for item in box_pl)


def test_ledger_reconciles_serialized_length() -> None:
    frames, masks = _toy()
    view, panorama_b = _jpeg_background(frames)
    _pl, _refs, actor_f = client_placements(frames, masks, PREDICTOR_BBOX_RESIZE)
    payload = serialize_setting(
        background=view,
        frames=frames,
        masks=masks,
        predictor=PREDICTOR_BBOX_RESIZE,
        residual=None,
    )
    ledger = reconcile_ledger(payload, panorama_b=panorama_b, actor_reference_f=actor_f, residual_r=0)
    assert ledger["residual"] == 0
    assert ledger["panorama"] == panorama_b
    assert ledger["transport_total"] == len(payload)
    assert ledger["pose_present"] is False


def test_malformed_truncated_transport_fails() -> None:
    frames, masks = _toy()
    view, _b = _jpeg_background(frames)
    payload = serialize_setting(
        background=view,
        frames=frames,
        masks=masks,
        predictor=PREDICTOR_PER_FRAME,
        residual=None,
    )
    with pytest.raises((ValueError, OSError, zipfile.BadZipFile)):
        reconstruct_serialized_client(payload[:64], require_compressed=True)


def test_uncoded_residual_is_rejected() -> None:
    frames, masks = _toy()
    view, _b = _jpeg_background(frames)
    raw = TransmittedResidual(
        bitstream=b"",
        codec_name="raw",
        is_coded=False,
        raw_frames=np.zeros_like(frames),
        shape=tuple(frames.shape),
    )
    with pytest.raises(ValueError, match="unencoded fallback"):
        serialize_setting(
            background=view,
            frames=frames,
            masks=masks,
            predictor=PREDICTOR_PER_FRAME,
            residual=raw,
        )


def test_ordinary_standalone_pixels_match() -> None:
    frames, masks = _toy()
    view, _b = _jpeg_background(frames)
    from src.pipeline.reconstruction.background import BackgroundResolver

    bg, _ = BackgroundResolver().frames_for(view, frame_count=2, height=32, width=32)
    ordinary = ordinary_composite(
        np.asarray(bg), frames, masks, PREDICTOR_PER_FRAME, residual=None
    )
    payload = serialize_setting(
        background=view,
        frames=frames,
        masks=masks,
        predictor=PREDICTOR_PER_FRAME,
        residual=None,
    )
    standalone = reconstruct_standalone(payload)
    assert ordinary.shape == (2, 32, 32, 3)
    np.testing.assert_array_equal(ordinary, standalone)


def test_refuse_overwrite_on_probe_report(tmp_path: Path) -> None:
    (tmp_path / "probe_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        refuse_overwrite(tmp_path)


def test_short_rgb24_dump_fails_closed() -> None:
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(DecodeCountError):
        require_exact_count(frames[:1], expected=2, height=8, width=8, source="short")


def test_residual_on_changes_pixels_and_r_positive() -> None:
    try:
        tools.resolve_ffmpeg()
    except FileNotFoundError:
        pytest.skip("ffmpeg not available")
    frames, masks = _toy()
    view, panorama_b = _jpeg_background(frames)
    from src.pipeline.reconstruction.background import BackgroundResolver

    bg, _ = BackgroundResolver().frames_for(view, frame_count=2, height=32, width=32)
    off = ordinary_composite(np.asarray(bg), frames, masks, PREDICTOR_PER_FRAME, residual=None)
    residual_frames = np.clip(frames.astype(np.int16) - off.astype(np.int16) + 128, 0, 255).astype(
        np.uint8
    )
    transmitted, _decoded = encode_residual_to_bitstream(
        residual_frames,
        EncodeRequest(codec_name="avc", rate_control=RateControl.CRF, rate=23),
        mode="clipped",
        fps=12.0,
    )
    assert transmitted.byte_count > 0
    payload = serialize_setting(
        background=view,
        frames=frames,
        masks=masks,
        predictor=PREDICTOR_PER_FRAME,
        residual=transmitted,
    )
    ledger = reconcile_ledger(
        payload,
        panorama_b=panorama_b,
        actor_reference_f=client_placements(frames, masks, PREDICTOR_PER_FRAME)[2],
        residual_r=int(transmitted.byte_count),
    )
    assert ledger["residual"] == transmitted.byte_count
    on_pixels = ordinary_composite(
        np.asarray(bg), frames, masks, PREDICTOR_PER_FRAME, residual=transmitted
    )
    standalone = reconstruct_standalone(payload)
    np.testing.assert_array_equal(on_pixels, standalone)
    assert not np.array_equal(on_pixels, off)
