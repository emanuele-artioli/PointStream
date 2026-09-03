"""Required behaviour for background.transport_scale (BP53).

Behaviour
1. Scale 1.0 does not resample; codec payloads match the old path; reconstruction
   pixels match; the geometry header is charged as metadata.
2. Static and panning geometry stay in original canonical coordinates after a
   half-scale round trip (translated marker centroid, not merely equal shapes).
3. A decoded half-scale chain restores identically on encoder and fresh-client
   paths.
4. Odd sizes floor to a positive even coded raster; too-small rasters refuse.
5. A context reset starts a new chain; a changed scale cannot resume a chain.
6. Snapshot/restore is equivalent at the same scale.
7. Header bytes are counted once, as metadata, and are not mixed into the AV1
   payload.

Plausible misuse
8. Unsupported scales and non-stream methods with scale 0.5 are rejected.

Deliberately not tested: libaom fidelity, 4K rate, BD-rate, quarter-scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background.scale import (
    HEADER_BYTES,
    TransportScaleError,
    coded_dimensions,
    downsample_plate,
    restore_plate,
    unpack_header,
)
from src.contracts import config as cfg
from src.contracts.domain import BACKGROUND_PANORAMA_FULL, BACKGROUND_PANORAMA_STREAM
from src.contracts.errors import ConfigError
from src.pipeline.reconstruction.background import warp_plate


def _model(**overrides: object):
    from src.components.background.strategy import bind as bind_background

    settings: dict[str, object] = {"method": BACKGROUND_PANORAMA_STREAM}
    settings.update(overrides)
    return bind_background(cfg.load({"background": settings}))


def _translated_plate(height: int = 80, width: int = 96, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Unique colour blocks plus a saturated marker whose centroid is known."""
    plate = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            plate[y : y + 8, x : x + 8] = (
                (17 * (x + 3) + dx) % 180 + 40,
                (13 * (y + 5) + dy) % 180 + 40,
                90,
            )
    top, left = 16 + dy, 24 + dx
    plate[top : top + 12, left : left + 12] = (255, 0, 0)
    return plate


def _marker_centroid(image: np.ndarray) -> tuple[float, float]:
    red = (
        (image[:, :, 0] >= 200)
        & (image[:, :, 1] <= 40)
        & (image[:, :, 2] <= 40)
    )
    ys, xs = np.nonzero(red)
    assert len(xs) > 0, "marker vanished; a coordinate shift would hide here"
    return float(xs.mean()), float(ys.mean())


def test_scale_one_does_not_resample_and_charges_the_header() -> None:
    plate = _translated_plate()
    array, header = downsample_plate(plate, 1.0)
    assert array.shape == plate.shape
    assert np.array_equal(array, plate)
    restored = restore_plate(array, header)
    assert np.array_equal(restored, plate)
    packed = header.pack()
    assert len(packed) == HEADER_BYTES
    assert unpack_header(packed) == header


def test_invalid_scales_and_method_combinations_are_rejected() -> None:
    with pytest.raises(ConfigError, match="transport_scale"):
        cfg.load({"background": {"method": BACKGROUND_PANORAMA_STREAM, "transport_scale": 0.25}})
    with pytest.raises(ConfigError, match="transport_scale"):
        cfg.load({"background": {"method": BACKGROUND_PANORAMA_STREAM, "transport_scale": 2.0}})
    with pytest.raises(ConfigError, match="transport_scale"):
        cfg.load({"background": {"method": BACKGROUND_PANORAMA_FULL, "transport_scale": 0.5}})
    with pytest.raises(TransportScaleError):
        coded_dimensions(64, 48, 0.25)


def test_odd_sizes_floor_to_positive_even_coded_rasters() -> None:
    assert coded_dimensions(15, 11, 0.5) == (6, 4)
    assert coded_dimensions(4, 4, 0.5) == (2, 2)
    with pytest.raises(TransportScaleError, match="too small"):
        coded_dimensions(3, 3, 0.5)
    with pytest.raises(TransportScaleError, match="too small"):
        coded_dimensions(2, 2, 0.5)


def test_half_scale_restore_keeps_static_and_translated_marker_geometry() -> None:
    static = _translated_plate(dx=0, dy=0)
    shifted = _translated_plate(dx=8, dy=4)
    identity = ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),)
    expected_static = _marker_centroid(static)
    expected_shift = _marker_centroid(shifted)
    for plate, expected in ((static, expected_static), (shifted, expected_shift)):
        coded, header = downsample_plate(plate, 0.5)
        assert coded.shape[1] == header.coded_width
        assert coded.shape[0] == header.coded_height
        restored = restore_plate(coded, header)
        assert restored.shape == plate.shape
        warped = warp_plate(
            restored, identity, height=plate.shape[0], width=plate.shape[1], frame_count=1
        )[0]
        cx, cy = _marker_centroid(warped)
        assert abs(cx - expected[0]) <= 2.0
        assert abs(cy - expected[1]) <= 2.0


@pytest.mark.integration
class TestScaledStreamCodec:
    def test_scale_one_codec_payload_matches_the_unscaled_path(self) -> None:
        plates = [_translated_plate(), _translated_plate(dx=6, dy=2)]
        old = _model()
        new = _model(transport_scale=1.0)
        for plate in plates:
            previous = old.transmit(plate)
            current = new.transmit(plate)
            assert previous.payload == current.payload
            assert current.geometry_header
            assert len(current.geometry_header) == HEADER_BYTES
            assert np.array_equal(old.decode_payload(previous), new.decode_payload(current))

    def test_half_scale_encoder_and_client_restore_identically(self) -> None:
        plates = [_translated_plate(), _translated_plate(dx=6, dy=2)]
        model = _model(transport_scale=0.5)
        artifacts = [model.transmit(plate) for plate in plates]
        encoder = model.decode_payload(artifacts[-1])
        client = model.client_plate(artifacts[-1])
        assert encoder is not None
        assert encoder.shape == plates[-1].shape
        assert np.array_equal(encoder, client)
        header = unpack_header(artifacts[-1].geometry_header)
        assert header.scale == 0.5
        coded = model._transmitter.reconstructions[-1]
        assert coded.shape == (header.coded_height, header.coded_width, 3)

    def test_context_reset_does_not_mix_coded_history(self) -> None:
        first = _translated_plate()
        second = _translated_plate(dx=4, dy=0)
        third = np.zeros((64, 80, 3), dtype=np.uint8)
        third[10:22, 10:22] = (255, 0, 0)
        model = _model(transport_scale=0.5)
        a = model.transmit(first, context_id="court")
        b = model.transmit(second, context_id="court")
        c = model.transmit(third, context_id="replay")
        assert a.mode == "full"
        assert b.mode == "stream"
        assert c.mode == "full"
        assert unpack_header(c.geometry_header).original_width == 80
        assert unpack_header(a.geometry_header).original_width == 96

    def test_geometry_header_is_metadata_not_codec_payload(self) -> None:
        from src.pipeline.reconstruction.background import BackgroundModelView
        from src.runner.accounting import sizes_bytes

        plate = _translated_plate()
        model = _model(transport_scale=0.5)
        artifact = model.transmit(plate)
        assert artifact.payload[:4] != b"PSBG"
        assert len(artifact.geometry_header) == HEADER_BYTES
        view = BackgroundModelView(
            plate=model.decode_payload(artifact),
            homographies=(),
            mode="full",
            width=artifact.width,
            height=artifact.height,
            payload_bytes=len(artifact.payload),
            geometry_header_bytes=len(artifact.geometry_header),
        )
        # The header length is the exact charged delta; panorama is codec bytes only.
        assert view.payload_bytes == len(artifact.payload)
        assert view.geometry_header_bytes == HEADER_BYTES
        ledger = sizes_bytes(
            source=int(plate.nbytes),
            panorama=int(view.payload_bytes or 0),
            metadata=int(view.geometry_header_bytes),
        )
        assert ledger.metadata == HEADER_BYTES
        assert ledger.panorama == len(artifact.payload)
        assert ledger.parts_sum == ledger.metadata + ledger.panorama

    def test_snapshot_roundtrip_and_changed_scale_rejection(self) -> None:
        plate = _translated_plate()
        model = _model(transport_scale=0.5)
        artifact = model.transmit(plate)
        restored = model.decode_payload(artifact)
        state = model.export_stream_state()
        assert state is not None
        clone = _model(transport_scale=0.5)
        clone.import_stream_state(state)
        assert np.array_equal(clone.decode_payload(artifact), restored)
        other = _model(transport_scale=1.0)
        with pytest.raises(ValueError, match="transport_scale"):
            other.import_stream_state(state)

    def test_runner_restores_canonical_size_and_charges_the_header(self) -> None:
        from dataclasses import replace

        from src.runner import run
        from tests.components.background.test_canonical_canvas import _court_pair
        from tests.runner.test_background_panorama import _config

        static, panning = _court_pair(n_static=2, n_pan=3)
        config = _config(method="panorama-stream")
        config = replace(
            config,
            background=replace(
                config.background,
                canvas="canonical",
                transport_scale=0.5,
            ),
        )
        result = run(config, [static, panning], context_ids=("court", "court"))
        assert result.frames.shape[1:3] == static.shape[1:3]
        assert result.sizes.metadata >= 2 * HEADER_BYTES
        assert result.sizes.panorama > 0
        assert result.symmetry is not None
