"""Background strategy and sidecar are independent axes, and panoramas need a plane."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background import REGISTRY as BACKGROUND
from src.components.background.plate import build_plate
from src.components.background.sidecar import ALL_SIDECAR_CODECS, build_sidecar, normalize_sidecar
from src.components.background.strategy import bind
from src.components.background.types import MODE_DELTA, MODE_FULL, MODE_NONE
from src.components.rigid import REGISTRY as RIGID
from src.contracts import config
from src.contracts.config import BackgroundConfig, PointstreamConfig, validate_backends
from src.contracts.domain import (
    BACKGROUND_NONE,
    BACKGROUND_PANORAMA_DELTA,
    BACKGROUND_PANORAMA_FULL,
    GENERAL,
)
from src.contracts.errors import ConfigError, ConfigValueError, UnknownBackendError

_REGISTRIES = {"background": BACKGROUND, "rigid": RIGID}


def _photo_plate(height: int = 32, width: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            128 + 80 * np.sin(xx / 8.0) * np.cos(yy / 10.0),
            128 + 80 * np.cos(xx / 12.0 + 1.0),
            128 + 80 * np.sin(yy / 7.0 + 2.0),
        ],
        axis=-1,
    )
    noise = rng.normal(0.0, 3.0, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _perturb(plate: np.ndarray, seed: int, scale: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delta = rng.integers(-scale, scale + 1, size=plate.shape, dtype=np.int16)
    return np.clip(plate.astype(np.int16) + delta, 0, 255).astype(np.uint8)


class TestStrategyAndSidecarAreIndependent:
    def test_panorama_delta_with_roi_video_constructs_and_validates(self) -> None:
        loaded = config.load(
            {"background": {"method": BACKGROUND_PANORAMA_DELTA, "codec": "roi-video"}}
        )
        validate_backends(loaded, registries=_REGISTRIES)
        backend = bind(loaded)
        assert backend.method == BACKGROUND_PANORAMA_DELTA
        assert backend.codec_name == "roi-video"
        assert "roi-video" in backend.codec_id

    def test_each_strategy_is_registered_and_constructs(self) -> None:
        for name in (
            BACKGROUND_PANORAMA_FULL,
            BACKGROUND_PANORAMA_DELTA,
            BACKGROUND_NONE,
        ):
            assert name in BACKGROUND
            built = BACKGROUND.build(name, codec="png")
            assert built.method == name  # type: ignore[attr-defined]

    def test_sidecar_choice_does_not_change_the_strategy(self) -> None:
        plate = _photo_plate()
        jpeg = BACKGROUND.build("panorama-full", codec="jpeg", jpeg_quality=50)
        png = BACKGROUND.build("panorama-full", codec="png")
        jpeg_art = jpeg.transmit(plate, chunk_id="c0")  # type: ignore[union-attr]
        png_art = png.transmit(plate, chunk_id="c0")  # type: ignore[union-attr]
        assert jpeg_art.method == png_art.method == BACKGROUND_PANORAMA_FULL
        assert jpeg_art.mode == png_art.mode == MODE_FULL
        assert jpeg_art.codec != png_art.codec
        assert jpeg_art.payload != png_art.payload
        assert jpeg_art.codec_id != png_art.codec_id

    def test_unknown_sidecar_is_rejected_inside_the_component(self) -> None:
        with pytest.raises(ConfigValueError, match="sidecar codec"):
            normalize_sidecar("webp")
        with pytest.raises(ConfigValueError, match="sidecar codec"):
            BACKGROUND.build("panorama-full", codec="webp")
        assert "jpeg" in ALL_SIDECAR_CODECS
        assert "png" in ALL_SIDECAR_CODECS
        assert "roi-video" in ALL_SIDECAR_CODECS


class TestPanoramaNeedsAPlanarCamera:
    def test_contract_rejects_panorama_under_general(self) -> None:
        with pytest.raises(ConfigError, match="parallax"):
            config.load({"domain": "general", "background": {"method": BACKGROUND_PANORAMA_FULL}})
        GENERAL.assert_background_valid(BACKGROUND_NONE)

    def test_component_refuses_to_bind_a_panorama_under_general(self) -> None:
        cfg = PointstreamConfig(
            domain="general",
            background=BackgroundConfig(method=BACKGROUND_PANORAMA_FULL),
        )
        with pytest.raises(ConfigValueError, match="parallax"):
            bind(cfg)
        with pytest.raises(ConfigValueError, match="parallax"):
            BACKGROUND.build("panorama-delta", domain=GENERAL)

    def test_none_under_general_binds(self) -> None:
        loaded = config.load(
            {"domain": "general", "background": {"method": BACKGROUND_NONE}}
        )
        validate_backends(loaded, registries=_REGISTRIES)
        backend = bind(loaded)
        artifact = backend.encode_frames(np.zeros((2, 16, 16, 3), dtype=np.uint8))
        assert artifact.mode == MODE_NONE
        assert artifact.deferred_to_residual
        assert artifact.cost().byte_count == 0


class TestDeltaVersusFull:
    def test_single_chunk_delta_is_byte_identical_to_full(self) -> None:
        plate = _photo_plate(seed=3)
        full = BACKGROUND.build("panorama-full", codec="png")
        delta = BACKGROUND.build("panorama-delta", codec="png")
        full_art = full.transmit(plate, scene_id="s0", chunk_id="c0")  # type: ignore[union-attr]
        delta_art = delta.transmit(plate, scene_id="s0", chunk_id="c0")  # type: ignore[union-attr]
        assert full_art.payload == delta_art.payload
        assert full_art.mode == delta_art.mode == MODE_FULL
        assert full_art.cost().byte_count == delta_art.cost().byte_count

    def test_second_chunk_of_a_scene_can_differ(self) -> None:
        first = _photo_plate(seed=4)
        second = _perturb(first, seed=5)
        full = BACKGROUND.build("panorama-full", codec="png")
        delta = BACKGROUND.build("panorama-delta", codec="png")

        first_full = full.transmit(first, scene_id="s0", chunk_id="c0")  # type: ignore[union-attr]
        first_delta = delta.transmit(first, scene_id="s0", chunk_id="c0")  # type: ignore[union-attr]
        assert first_full.payload == first_delta.payload

        decoded = delta.decode_payload(first_delta)  # type: ignore[union-attr]
        second_full = full.transmit(second, scene_id="s0", chunk_id="c1")  # type: ignore[union-attr]
        second_delta = delta.transmit(  # type: ignore[union-attr]
            second,
            previous_decoded=decoded,
            scene_id="s0",
            previous_scene_id="s0",
            chunk_id="c1",
        )
        assert second_delta.mode == MODE_DELTA
        assert second_full.mode == MODE_FULL
        assert second_delta.payload != second_full.payload

    def test_a_new_scene_sends_full_even_under_delta(self) -> None:
        plate_a = _photo_plate(seed=6)
        plate_b = _photo_plate(seed=7)
        delta = BACKGROUND.build("panorama-delta", codec="png")
        first = delta.transmit(plate_a, scene_id="s0", chunk_id="c0")  # type: ignore[union-attr]
        decoded = delta.decode_payload(first)  # type: ignore[union-attr]
        second = delta.transmit(  # type: ignore[union-attr]
            plate_b,
            previous_decoded=decoded,
            scene_id="s1",
            previous_scene_id="s0",
            chunk_id="c1",
        )
        assert second.mode == MODE_FULL


class TestPlateBuilder:
    def test_masked_median_keeps_a_stationary_blob_out_of_the_plate(self) -> None:
        """A blob in 4 of 5 frames biases an unmasked median; masks recover the court."""
        height, width = 24, 32
        frames = np.full((5, height, width, 3), 40, dtype=np.uint8)
        masks = np.zeros((5, height, width), dtype=np.uint8)
        for index in range(4):
            frames[index, 8:16, 10:18] = 250
            masks[index, 8:16, 10:18] = 255
        leaked, _ = build_plate(frames, masks=None)
        recovered, homographies = build_plate(frames, masks=masks)
        assert leaked[12, 14].mean() > 200
        assert recovered[12, 14].mean() < 80
        assert len(homographies) == 5


class TestUnknownBackend:
    def test_validate_backends_rejects_an_unregistered_method(self) -> None:
        loaded = config.load({"background": {"method": BACKGROUND_PANORAMA_FULL}})
        broken = loaded.with_(background=BackgroundConfig(method="panorama-static"))
        with pytest.raises(ConfigError):
            validate_backends(broken, registries=_REGISTRIES)
        with pytest.raises(UnknownBackendError, match="background"):
            BACKGROUND.spec("panorama-static")


@pytest.mark.integration
def test_roi_video_sidecar_encodes_a_plate() -> None:
    plate = _photo_plate(height=32, width=48, seed=9)
    sidecar = build_sidecar("roi-video", roi_crf=40, roi_preset="ultrafast")
    payload = sidecar.encode(plate)
    decoded = sidecar.decode(payload)
    assert len(payload) > 0
    assert decoded.shape[0] == plate.shape[0]
    assert decoded.shape[1] == plate.shape[1]
    error = float(np.mean(np.abs(decoded.astype(np.int16) - plate.astype(np.int16))))
    assert error < 25.0
