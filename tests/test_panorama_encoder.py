from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.config import PointstreamConfig
from src.shared.schemas import (
    CameraPose,
    EncodedChunkPayload,
    PanoramaPacket,
    VideoChunk,
)
from src.shared import synthesis_engine as se
from src.transport.panorama_encoder import (
    JpegPanoramaEncoder,
    PngPanoramaEncoder,
    RoiVideoPanoramaEncoder,
    apply_panorama_delta,
    build_panorama_encoder,
    compute_panorama_delta,
    read_panorama_pixels_from_path,
    round_trip_panorama,
    round_trip_panorama_delta,
)


def _random_bgr_image(height: int = 48, width: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _photo_like_bgr_image(height: int = 64, width: int = 96, seed: int = 0) -> np.ndarray:
    """A smooth, spatially-correlated synthetic image -- still-image/video
    codecs compress this the way they compress a real photographed panorama
    (i.i.d. random noise, by contrast, is close to worst-case for any lossy
    codec and isn't representative of the real workload)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            128 + 100 * np.sin(xx / 12.0) * np.cos(yy / 18.0),
            128 + 100 * np.cos(xx / 20.0 + 1.0),
            128 + 100 * np.sin(yy / 10.0 + 2.0),
        ],
        axis=-1,
    )
    noise = rng.normal(0.0, 4.0, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _small_perturbation(base: np.ndarray, max_abs_delta: int, seed: int) -> np.ndarray:
    """A `current` panorama that only drifts a little from `base` -- the
    realistic case for panorama+delta (scoreboard/crowd changes between
    sub-chunks of the same point), unlike two fully independent images."""
    rng = np.random.default_rng(seed)
    delta = rng.integers(-max_abs_delta, max_abs_delta + 1, size=base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def test_compute_and_apply_panorama_delta_are_exact_inverses_for_small_changes() -> None:
    previous = _random_bgr_image(seed=1)
    # Representable range is +/-127 per channel (see compute_panorama_delta's
    # docstring) -- stay comfortably inside it, matching the realistic
    # "background barely changed" case this rung targets.
    current = _small_perturbation(previous, max_abs_delta=100, seed=2)

    diff = compute_panorama_delta(current_bgr=current, previous_bgr=previous)
    reconstructed = apply_panorama_delta(previous_bgr=previous, diff_bgr=diff)

    # No lossy compression in between: the diff/apply arithmetic must be a
    # bit-exact inverse (the Residual Guarantee needs this identity to hold
    # even once a real codec's quantization is layered on top).
    assert np.array_equal(reconstructed, current)


def test_compute_and_apply_panorama_delta_clips_large_jumps() -> None:
    """Documents the known limitation (see compute_panorama_delta's
    docstring): a per-channel jump bigger than +/-127 is not perfectly
    invertible through the uint8 diff representation."""
    previous = _random_bgr_image(seed=1)
    current = _random_bgr_image(seed=2)  # fully independent -> some channels jump > 127

    diff = compute_panorama_delta(current_bgr=current, previous_bgr=previous)
    reconstructed = apply_panorama_delta(previous_bgr=previous, diff_bgr=diff)

    assert not np.array_equal(reconstructed, current)


def test_delta_shape_mismatch_raises() -> None:
    previous = _random_bgr_image(height=48, width=64)
    current = _random_bgr_image(height=32, width=32)
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_panorama_delta(current_bgr=current, previous_bgr=previous)


def test_round_trip_panorama_delta_matches_manual_reconstruction() -> None:
    previous = _random_bgr_image(seed=3)
    current = _random_bgr_image(seed=4)
    encoder = JpegPanoramaEncoder(quality=80)

    encoded_bytes, reconstructed = round_trip_panorama_delta(
        current_bgr=current, previous_bgr=previous, encoder=encoder
    )

    # Manually redo what an independent "client" would do from encoded_bytes
    # + the same previous panorama, and assert it lands on the identical
    # pixels the "server" used for its residual (bit-identical symmetry).
    decoded_diff = encoder.decode_bytes(encoded_bytes)
    manual_reconstruction = apply_panorama_delta(previous_bgr=previous, diff_bgr=decoded_diff)
    assert np.array_equal(manual_reconstruction, reconstructed)


def test_roi_video_panorama_encoder_round_trips(tmp_path: Path) -> None:
    encoder = RoiVideoPanoramaEncoder(crf=28, preset="ultrafast")
    image = _photo_like_bgr_image(height=64, width=96, seed=5)

    encoded_bytes, decoded = round_trip_panorama(image, encoder)
    assert isinstance(encoded_bytes, bytes)
    assert len(encoded_bytes) > 0
    assert decoded.shape == image.shape

    # Codec is lossy but a smooth, photo-like image should compress cleanly.
    mean_abs_error = float(np.mean(np.abs(decoded.astype(np.int16) - image.astype(np.int16))))
    assert mean_abs_error < 15.0

    out_path = tmp_path / "panorama.mp4"
    out_path.write_bytes(encoded_bytes)
    read_back = read_panorama_pixels_from_path(out_path)
    assert read_back.shape == image.shape


def test_roi_video_codec_id_distinguishes_crf() -> None:
    low_crf = RoiVideoPanoramaEncoder(crf=18)
    high_crf = RoiVideoPanoramaEncoder(crf=40)
    assert low_crf.codec_id != high_crf.codec_id


def test_build_panorama_encoder_supports_roi_video() -> None:
    config = PointstreamConfig(panorama_codec="roi-video", panorama_roi_crf=25, panorama_roi_preset="veryfast")
    encoder = build_panorama_encoder(config=config)
    assert isinstance(encoder, RoiVideoPanoramaEncoder)
    assert "crf25" in encoder.codec_id


def test_read_panorama_pixels_from_path_dispatches_by_suffix(tmp_path: Path) -> None:
    image = _random_bgr_image(height=32, width=32, seed=6)

    jpeg_path = tmp_path / "panorama.jpg"
    jpeg_bytes = JpegPanoramaEncoder(quality=90).encode_bytes(image)
    jpeg_path.write_bytes(jpeg_bytes)
    decoded_jpeg = read_panorama_pixels_from_path(jpeg_path)
    assert decoded_jpeg.shape == image.shape

    png_path = tmp_path / "panorama.png"
    png_bytes = PngPanoramaEncoder(compression=1).encode_bytes(image)
    png_path.write_bytes(png_bytes)
    decoded_png = read_panorama_pixels_from_path(png_path)
    assert np.array_equal(decoded_png, image)  # PNG is lossless


def _payload_with_panorama(
    *,
    chunk_id: str,
    scene_id: str | None,
    panorama_uri: str,
    panorama_mode: str,
    panorama_image: np.ndarray | None,
) -> EncodedChunkPayload:
    from src.shared.schemas import BallPacket, ResidualPacket, TensorSpec

    chunk = VideoChunk(
        chunk_id=chunk_id,
        source_uri="memory://source",
        start_frame_id=0,
        fps=25.0,
        num_frames=1,
        width=32,
        height=24,
        scene_id=scene_id,
    )
    panorama = PanoramaPacket(
        chunk_id=chunk_id,
        panorama_uri=panorama_uri,
        frame_width=32,
        frame_height=24,
        camera_poses=[CameraPose(frame_id=0, tx=0.0, ty=0.0, tz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)],
        panorama_image=panorama_image.tolist() if panorama_image is not None else None,
        homography_matrices=[np.eye(3, dtype=np.float32).tolist()],
        selected_frame_indices=[0],
        panorama_mode=panorama_mode,  # type: ignore[arg-type]
    )
    ball = BallPacket(
        chunk_id=chunk_id,
        object_id="ball_0",
        trajectory_spec=TensorSpec(name="ball", shape=[1, 1, 4], dtype="torch.float32"),
        events=[],
        states=[],
    )
    residual = ResidualPacket(chunk_id=chunk_id, codec="libx265", residual_video_uri="memory://residual")
    return EncodedChunkPayload(
        chunk=chunk, panorama=panorama, actors=[], actor_references=[], rigid_objects=[], ball=ball, residual=residual
    )


def test_synthesis_engine_reconstructs_delta_chunk_bit_identically_to_encoder(tmp_path: Path) -> None:
    """The critical background-layer rung-2 symmetry check (report 10 5.3,
    CLAUDE.md's Residual Guarantee): an independent "client" SynthesisEngine
    given only the transmitted sidecar bytes must reconstruct the exact same
    pixels the "server" used to compute its residual against."""
    encoder = JpegPanoramaEncoder(quality=70)
    scene_id = "scene0000"

    full_panorama = _random_bgr_image(height=24, width=32, seed=10)
    encoded_full, server_decoded_full = round_trip_panorama(full_panorama, encoder)

    next_panorama = _random_bgr_image(height=24, width=32, seed=11)
    encoded_delta, server_decoded_delta = round_trip_panorama_delta(
        current_bgr=next_panorama, previous_bgr=server_decoded_full, encoder=encoder
    )

    full_path = tmp_path / "panorama_full.jpg"
    full_path.write_bytes(encoded_full)
    delta_path = tmp_path / "panorama_delta.jpg"
    delta_path.write_bytes(encoded_delta)

    client_engine = se.SynthesisEngine(config=PointstreamConfig(), device="cpu")

    payload_full = _payload_with_panorama(
        chunk_id="s0000c0000",
        scene_id=scene_id,
        panorama_uri=str(full_path),
        panorama_mode="full",
        panorama_image=None,
    )
    client_full = client_engine._resolve_panorama_image(payload_full)
    assert np.array_equal(client_full, server_decoded_full)

    payload_delta = _payload_with_panorama(
        chunk_id="s0000c0001",
        scene_id=scene_id,
        panorama_uri=str(delta_path),
        panorama_mode="delta",
        panorama_image=None,
    )
    client_delta = client_engine._resolve_panorama_image(payload_delta)
    assert np.array_equal(client_delta, server_decoded_delta)


def test_synthesis_engine_rejects_delta_with_no_prior_scene_state(tmp_path: Path) -> None:
    encoder = JpegPanoramaEncoder(quality=70)
    diff_image = _random_bgr_image(seed=12)
    encoded_bytes = encoder.encode_bytes(diff_image)
    delta_path = tmp_path / "panorama_delta.jpg"
    delta_path.write_bytes(encoded_bytes)

    engine = se.SynthesisEngine(config=PointstreamConfig(), device="cpu")
    payload = _payload_with_panorama(
        chunk_id="s0001c0000",
        scene_id="scene_never_seen",
        panorama_uri=str(delta_path),
        panorama_mode="delta",
        panorama_image=None,
    )
    with pytest.raises(ValueError, match="protocol violation"):
        engine._resolve_panorama_image(payload)
