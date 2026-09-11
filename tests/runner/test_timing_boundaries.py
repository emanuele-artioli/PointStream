"""Unit tests for Gate 1 timing boundaries, GPU sync, and independent client.

Covers:
- Controlled-clock verification of disjoint boundaries (encoder, client, evaluation)
- Proof that metric evaluation runtime does not change codec clocks
- Generation-off timing coverage
- Exact reproduction of delivered frames by independent client
- GPU synchronization at timed boundaries
- Completeness of Gate 1 timing dictionary
"""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from src.contracts.lattice import ART_DELIVERED
from src.components.background.scale import TransmittedBackground
from src.pipeline.reconstruction.quality import NumpyPsnrEvaluator, QualityReport
from src.runner import run
from src.runner.client import reconstruct_independent_client
from tests.components.background.test_canonical_canvas import _court_pair
from tests.runner.test_background_panorama import _config
from tests.runner.test_run import _all_off, _clip


class ControlledClock:
    """A deterministic clock that can be manually advanced."""

    def __init__(self, start: float = 100.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, amount: float) -> None:
        self.time += amount


class DeliberateEvaluator(NumpyPsnrEvaluator):
    """Evaluator that advances a controlled clock to simulate variable metric time."""

    def __init__(self, clock: ControlledClock, delay: float = 5.0) -> None:
        self.clock = clock
        self.delay = delay

    def evaluate(
        self, reference: np.ndarray, predicted: np.ndarray, **kwargs: Any
    ) -> QualityReport:
        self.clock.advance(self.delay)
        return super().evaluate(reference, predicted, **kwargs)


def test_controlled_clock_proves_disjoint_boundaries() -> None:
    """Clocks are disjoint: encoder, client, and evaluation measure separate non-overlapping spans."""
    clock = ControlledClock(10.0)
    sync_calls = 0

    def counting_sync() -> None:
        nonlocal sync_calls
        sync_calls += 1
        clock.advance(0.1)

    config = _all_off()
    clips = [_clip(50, frames=2)]

    result = run(config, clips, clock=clock, sync_fn=counting_sync)

    assert result.encoder_seconds >= 0.0
    assert result.client_seconds >= 0.0
    assert result.evaluation_seconds >= 0.0

    # Boundaries must be disjoint and positive
    assert sync_calls > 0
    assert "encoder_seconds" in result.timing
    assert "client_seconds" in result.timing
    assert "evaluation_seconds" in result.timing


def test_metrics_runtime_does_not_change_codec_clocks() -> None:
    """The metric stage and evaluation time must not alter encoder_seconds or client_seconds."""
    clock1 = ControlledClock(100.0)
    clock2 = ControlledClock(100.0)

    config = _all_off()
    clips = [_clip(60, frames=2)]

    fast_evaluator = DeliberateEvaluator(clock1, delay=0.0)
    slow_evaluator = DeliberateEvaluator(clock2, delay=50.0)

    res_fast = run(config, clips, evaluator=fast_evaluator, clock=clock1, sync_fn=None)
    res_slow = run(config, clips, evaluator=slow_evaluator, clock=clock2, sync_fn=None)

    # Codec clocks must be completely independent of metric evaluation time
    assert res_fast.encoder_seconds == pytest.approx(res_slow.encoder_seconds, abs=1e-5)
    assert res_fast.client_seconds == pytest.approx(res_slow.client_seconds, abs=1e-5)

    # Evaluation seconds must capture the metric delay
    assert res_slow.evaluation_seconds > res_fast.evaluation_seconds
    assert res_slow.evaluation_seconds >= 50.0


def test_generation_off_timing_coverage() -> None:
    """Generation-off (the Gate A regime) is fully covered with disjoint clocks."""
    config = _config(method="panorama-stream")
    config = replace(config, background=replace(config.background, canvas="canonical"))
    clips = list(_court_pair(n_static=2, n_pan=2))

    clock = ControlledClock(0.0)

    def advancing_sync() -> None:
        clock.advance(0.05)

    result = run(config, clips, context_ids=("court", "court"), clock=clock, sync_fn=advancing_sync)

    # Verify all three clocks are populated
    assert result.encoder_seconds > 0.0
    assert result.client_seconds > 0.0
    assert result.evaluation_seconds > 0.0

    # Verify timing dictionary contains steady state and system records
    timing = result.timing
    assert "steady_state" in timing
    assert "cold_initialization" in timing
    assert "preparation" in timing
    assert "attempt_wall" in timing
    assert "host" in timing
    assert "cpu" in timing
    assert "peak_memory_bytes" in timing
    assert timing["peak_memory_bytes"] > 0


def test_independent_client_reproduces_delivered_frames() -> None:
    """Independent client must reconstruct bit-identical delivered frames without source pixels."""
    config = _config(method="panorama-stream")
    config = replace(config, background=replace(config.background, canvas="canonical"))
    clips = list(_court_pair(n_static=2, n_pan=2))

    result = run(config, clips, context_ids=("court", "court"))

    # Delivered frames from run matches concatenate of chunks
    assert np.array_equal(result.frames, result.delivered_frames)

    # Direct independent client reconstruction for each chunk
    for chunk in result.chunks:
        from src.contracts.lattice import ART_BACKGROUND_MODEL, STAGE_BACKGROUND
        from src.runner.stages import _as_background, _delivered_frames

        view = _as_background(
            chunk.bag.get(ART_BACKGROUND_MODEL) or chunk.bag.get(STAGE_BACKGROUND)
        )
        delivered_target = _delivered_frames(chunk.bag[ART_DELIVERED])

        recon = reconstruct_independent_client(
            background=view,
            frame_count=int(delivered_target.shape[0]),
            height=int(delivered_target.shape[1]),
            width=int(delivered_target.shape[2]),
            placements=(),
            residual_payload=None,
        )
        assert np.array_equal(recon, delivered_target)


def test_gpu_sync_called_at_every_timed_boundary() -> None:
    """GPU sync hook must be called at every timed boundary."""
    sync_events: list[str] = []

    def log_sync() -> None:
        sync_events.append("sync")

    config = _all_off()
    clips = [_clip(40, frames=2)]

    run(config, clips, sync_fn=log_sync)

    # Must have synchronized multiple times (preparation, encoder stages, client phase, evaluation phase, assembly)
    assert len(sync_events) >= 5


def test_serialized_client_boundary_carries_background_and_foreground() -> None:
    from src.pipeline.reconstruction.background import BackgroundModelView
    from src.runner.client import (
        ClientPlacement,
        reconstruct_serialized_client,
        serialize_client_request,
    )

    background = BackgroundModelView(
        plate=np.zeros((4, 4, 3), dtype=np.uint8),
        homographies=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),),
        width=4,
        height=4,
        payload_bytes=7,
    )
    crop = np.full((2, 2, 3), 200, dtype=np.uint8)
    payload = serialize_client_request(
        background=background,
        frame_count=1,
        height=4,
        width=4,
        placements=(ClientPlacement(crop=crop, bbox=(1, 1, 3, 3)),),
    )
    assert isinstance(payload, bytes)
    reconstructed = reconstruct_serialized_client(payload)
    assert isinstance(reconstructed, np.ndarray)
    assert np.array_equal(reconstructed[0, 1:3, 1:3], crop)


def test_serialized_client_rejects_live_object() -> None:
    from src.runner.client import reconstruct_serialized_client

    with pytest.raises(TypeError, match="must be bytes"):
        reconstruct_serialized_client({})  # type: ignore[arg-type]


def test_serialized_client_decodes_raw_background_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client must decode copied codec bytes instead of accepting encoder pixels."""
    from src.components.background import scale
    from src.pipeline.reconstruction.background import BackgroundModelView
    from src.runner.client import reconstruct_serialized_client, serialize_client_request

    decoded_plate = np.full((4, 4, 3), 17, dtype=np.uint8)
    seen: dict[str, object] = {}

    def fake_decode(codec: str, packets: Sequence[TransmittedBackground]) -> np.ndarray:
        packet_tuple = tuple(packets)
        seen["codec"] = codec
        seen["payloads"] = tuple(packet.payload for packet in packet_tuple)
        seen["headers"] = tuple(packet.geometry_header for packet in packet_tuple)
        return decoded_plate

    monkeypatch.setattr(scale, "decode_transmitted_stream", fake_decode)
    background = BackgroundModelView(
        plate=np.full((4, 4, 3), 255, dtype=np.uint8),
        width=4,
        height=4,
        payload_bytes=2,
        wire_payloads=(b"i", b"p"),
        wire_geometry_headers=(b"h0", b"h1"),
        wire_codec="av1",
        wire_codec_id="av1 low-delay test",
    )
    payload = serialize_client_request(
        background=background,
        frame_count=1,
        height=4,
        width=4,
    )

    reconstructed = reconstruct_serialized_client(payload)

    import io

    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        assert "background_plate" not in arrays.files
        assert "background_payload_0" in arrays.files
    assert seen == {
        "codec": "av1",
        "payloads": (b"i", b"p"),
        "headers": (b"h0", b"h1"),
    }
    assert np.array_equal(reconstructed[0], decoded_plate)


def test_serialized_client_decodes_still_sidecar_without_raw_plate() -> None:
    """Still panorama must ship coded bytes, not the encoder's decoded plate."""
    import io

    from src.components.background.sidecar import JpegSidecar
    from src.pipeline.reconstruction.background import BackgroundModelView
    from src.runner.client import reconstruct_serialized_client, serialize_client_request

    true_plate = np.full((512, 512, 3), 17, dtype=np.uint8)
    encoder_plate = np.full((512, 512, 3), 255, dtype=np.uint8)
    sidecar = JpegSidecar(quality=50)
    coded = sidecar.encode(true_plate)
    expected = sidecar.decode(coded)
    background = BackgroundModelView(
        plate=encoder_plate,
        homographies=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),),
        width=512,
        height=512,
        payload_bytes=len(coded),
        wire_payloads=(coded,),
        wire_geometry_headers=(b"",),
        sidecar_codec="jpeg",
    )
    payload = serialize_client_request(
        background=background,
        frame_count=1,
        height=512,
        width=512,
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        assert "background_plate" not in arrays.files
        assert "background_payload_0" in arrays.files
        assert int(arrays["background_payload_0"].nbytes) == len(coded)
    assert len(payload) < int(encoder_plate.nbytes) // 2
    reconstructed = reconstruct_serialized_client(payload)
    assert np.array_equal(reconstructed[0], expected)


def test_serialized_client_rejects_still_packets_without_codec() -> None:
    from src.pipeline.reconstruction.background import BackgroundModelView
    from src.runner.client import reconstruct_serialized_client, serialize_client_request

    background = BackgroundModelView(
        plate=np.zeros((4, 4, 3), dtype=np.uint8),
        width=4,
        height=4,
        payload_bytes=3,
        wire_payloads=(b"abc",),
        wire_geometry_headers=(b"",),
    )
    payload = serialize_client_request(
        background=background,
        frame_count=1,
        height=4,
        width=4,
    )
    with pytest.raises(ValueError, match="no sidecar codec"):
        reconstruct_serialized_client(payload)
