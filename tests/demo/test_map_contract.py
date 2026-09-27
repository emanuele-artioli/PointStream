from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demo.evaluation.wire_accounting import payload_kbps
from demo.pipeline.maps.contract import MapStream, OverlayPayloadError, write_sidecar
from demo.pipeline.maps.encode import pack_binary_mask, unpack_binary_mask


def _stream(**overrides: object) -> MapStream:
    base: dict[str, object] = dict(
        map="canny",
        backend="opencv-canny-50-150",
        payload_path="/tmp/canny.payload.bin",
        payload_bytes=1000,
        preview_path="/tmp/preview_canny.mp4",
        preview_bytes=99999,
        duration_s=2.0,
        n_frames=60,
        fps=30.0,
        extract_ms_p50=1.0,
        extract_ms_p95=2.0,
        pack_ms_p50=0.2,
        codec_ms_p50=0.0,
        decode_ms_p50=0.1,
        gpu="cpu",
    )
    base.update(overrides)
    return MapStream(**base)  # type: ignore[arg-type]


def test_payload_kbps_ignores_preview_bytes() -> None:
    stream = _stream(payload_bytes=2500, preview_bytes=1_000_000, duration_s=2.0)
    assert stream.payload_kbps == payload_kbps(2500, 2.0)
    assert stream.payload_kbps == pytest.approx(10.0)
    sidecar = stream.to_sidecar()
    assert sidecar["payload_kbps"] == pytest.approx(10.0)
    assert "preview_bytes" in sidecar
    assert sidecar["payload_kbps"] != pytest.approx(payload_kbps(1_000_000, 2.0))


def test_overlay_mp4_rejected_as_payload() -> None:
    with pytest.raises(OverlayPayloadError):
        _stream(payload_path="/tmp/preview_edges.mp4", kind="native")
    with pytest.raises(OverlayPayloadError):
        _stream(payload_path="/tmp/edges.mp4", kind="overlay")


def test_teleop_badge_uses_serial_p50() -> None:
    assert _stream().teleop_ok is True
    slow = _stream(extract_ms_p50=40.0, pack_ms_p50=5.0, codec_ms_p50=10.0, decode_ms_p50=1.0)
    assert slow.teleop_ok is False


def test_write_sidecar_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sidecar.json"
    write_sidecar(_stream(payload_path=str(tmp_path / "payload.bin")), path)
    text = path.read_text()
    assert "payload_kbps" in text
    assert "teleop_ok" in text


def test_pack_binary_roundtrip() -> None:
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[4:10, 8:20] = 1
    blob = pack_binary_mask(mask)
    out = unpack_binary_mask(blob, 32, 48)
    assert out.shape == mask.shape
    assert np.array_equal(out, mask)
