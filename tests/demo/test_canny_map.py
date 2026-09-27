from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from demo.evaluation.wire_accounting import payload_kbps
from demo.pipeline.maps.canny import (
    DEFAULT_HI,
    DEFAULT_LO,
    HYSTERESIS_PAIRS,
    PAYLOAD_FORMAT,
    PAYLOAD_MAGIC,
    extract_canny_frame,
    extract_canny_frames,
    main,
    pack_canny_payload,
    unpack_canny_payload,
    write_canny_map,
)
from demo.pipeline.maps.contract import MapStream, OverlayPayloadError, read_sidecar


def _box_frames(n: int = 3, contrast: int = 255) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(n):
        img = np.zeros((48, 64), dtype=np.uint8)
        img[8 + i : 32, 10 : 50] = contrast
        frames.append(img)
    return frames


def test_default_hysteresis_pair() -> None:
    assert (DEFAULT_LO, DEFAULT_HI) == (50, 150)
    assert HYSTERESIS_PAIRS == ((50, 150), (100, 200), (150, 250))


def test_extract_canny_frames_binary_hw() -> None:
    frames = _box_frames()
    masks = extract_canny_frames(frames, lo=50, hi=150)
    assert len(masks) == 3
    for mask, src in zip(masks, frames):
        assert mask.shape == src.shape
        assert mask.ndim == 2
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        assert int(mask.sum()) > 0


def test_extract_is_not_rgb_overlay() -> None:
    bgr = np.zeros((32, 40, 3), dtype=np.uint8)
    bgr[6:26, 8:30] = (0, 200, 40)
    mask = extract_canny_frame(bgr, lo=50, hi=150)
    assert mask.ndim == 2
    assert mask.shape == (32, 40)


def test_higher_hysteresis_drops_weak_edges() -> None:
    weak = _box_frames(n=1, contrast=40)[0]
    low = extract_canny_frame(weak, lo=50, hi=150)
    high = extract_canny_frame(weak, lo=150, hi=250)
    assert int(low.sum()) > 0
    assert int(high.sum()) < int(low.sum())


def test_pack_rejects_color_overlay() -> None:
    overlay = np.zeros((16, 20, 3), dtype=np.uint8)
    overlay[..., 2] = 255
    with pytest.raises(ValueError, match="HxW binary"):
        pack_canny_payload([overlay])


def test_chain_codec_roundtrip_covers_every_pixel() -> None:
    from demo.pipeline.maps.canny import FLAG_CHAIN, _decode_chain_plane, _encode_chain_plane

    mask = np.zeros((48, 96), dtype=np.uint8)
    for i in range(70):
        mask[12 + (i // 20), 8 + i] = 1
    mask[30, 40] = 1
    mask[31, 41] = 1
    blob = _encode_chain_plane(mask)
    restored, offset = _decode_chain_plane(blob, 0, 48, 96)
    assert offset == len(blob)
    assert np.array_equal(restored, mask)

    packed = pack_canny_payload([mask, mask], include_chain=True)
    flags = int.from_bytes(packed[20:24], "little")
    assert flags in {0, 1, 2, FLAG_CHAIN}
    for got, want in zip(unpack_canny_payload(packed), (mask, mask)):
        assert np.array_equal(got, want)


def test_payload_roundtrip_and_framing() -> None:
    masks = extract_canny_frames(_box_frames(), lo=50, hi=150)
    blob = pack_canny_payload(masks)
    assert blob.startswith(PAYLOAD_MAGIC)
    assert int.from_bytes(blob[4:8], "little") == 3
    restored = unpack_canny_payload(blob)
    assert len(restored) == len(masks)
    for got, want in zip(restored, masks):
        assert np.array_equal(got, want)


def test_unpack_legacy_v1_per_frame_zstd() -> None:
    from demo.pipeline.maps.canny import _HEADER_V1, _LEN
    from demo.pipeline.maps.encode import pack_binary_mask

    masks = extract_canny_frames(_box_frames(), lo=50, hi=150)
    height, width = masks[0].shape
    chunks = [_HEADER_V1.pack(PAYLOAD_MAGIC, 1, len(masks), height, width)]
    for mask in masks:
        packed = pack_binary_mask(mask)
        chunks.append(_LEN.pack(len(packed)))
        chunks.append(packed)
    restored = unpack_canny_payload(b"".join(chunks))
    for got, want in zip(restored, masks):
        assert np.array_equal(got, want)


def test_write_canny_map_sidecar_and_preview(tmp_path: Path) -> None:
    frames = _box_frames()
    stream = write_canny_map(frames, tmp_path, lo=50, hi=150, fps=10.0, n_warmup=0, n_runs=1)
    payload_path = tmp_path / "payload.bin"
    preview_dir = tmp_path / "preview"
    sidecar_path = tmp_path / "sidecar.json"

    assert payload_path.is_file()
    assert sidecar_path.is_file()
    assert payload_path.name == "payload.bin"
    assert not payload_path.name.startswith("preview_")
    pngs = sorted(preview_dir.glob("*.png"))
    assert len(pngs) == len(frames)

    sidecar = read_sidecar(sidecar_path)
    assert sidecar["map"] == "canny"
    assert sidecar["backend"].startswith("opencv-canny-50-150")
    assert sidecar["kind"] == "native"
    assert sidecar["n_frames"] == 3
    assert sidecar["fps"] == pytest.approx(10.0)
    assert sidecar["duration_s"] == pytest.approx(0.3)
    assert sidecar["payload_bytes"] == payload_path.stat().st_size
    assert sidecar["preview_bytes"] == sum(p.stat().st_size for p in pngs)
    assert sidecar["payload_kbps"] == pytest.approx(
        payload_kbps(sidecar["payload_bytes"], sidecar["duration_s"])
    )
    assert sidecar["payload_kbps"] != pytest.approx(
        payload_kbps(sidecar["preview_bytes"], sidecar["duration_s"])
    )
    assert sidecar["payload_format"] == PAYLOAD_FORMAT
    assert sidecar["gpu"] == stream.gpu
    assert "extract_ms_p50" in sidecar
    assert "pack_ms_p50" in sidecar

    packed = payload_path.read_bytes()
    restored = unpack_canny_payload(packed)
    assert restored[0].ndim == 2
    vis = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
    assert vis is not None
    assert vis.ndim == 3
    assert vis.shape[2] == 4
    assert set(np.unique(vis[:, :, 3])).issubset({0, 220})


def test_payload_path_is_native_not_overlay(tmp_path: Path) -> None:
    stream = write_canny_map(_box_frames(n=1), tmp_path, n_warmup=0, n_runs=1)
    sidecar = read_sidecar(tmp_path / "sidecar.json")
    assert Path(sidecar["payload_path"]).name == "payload.bin"
    assert not Path(sidecar["payload_path"]).name.startswith("preview_")
    assert stream.kind == "native"
    with pytest.raises(OverlayPayloadError):
        MapStream(
            map="canny",
            backend="opencv-canny-50-150",
            payload_path=str(tmp_path / "preview_canny.mp4"),
            payload_bytes=1,
            preview_path=str(tmp_path / "preview"),
            preview_bytes=1,
            duration_s=1.0,
            n_frames=1,
            fps=30.0,
            extract_ms_p50=0.0,
            extract_ms_p95=0.0,
            pack_ms_p50=0.0,
            codec_ms_p50=0.0,
            decode_ms_p50=0.0,
            gpu="cpu",
        )


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


def test_unpack_legacy_v2_packbits_volume() -> None:
    from demo.pipeline.maps.canny import FLAG_XOR_DELTA, _HEADER_V2, _compress_volume

    masks = extract_canny_frames(_box_frames(), lo=50, hi=150)
    height, width = masks[0].shape
    prev = masks[0]
    chunks = [np.packbits(prev.ravel()).tobytes()]
    for mask in masks[1:]:
        chunks.append(np.packbits(np.bitwise_xor(mask, prev).ravel()).tobytes())
        prev = mask
    blob = _HEADER_V2.pack(PAYLOAD_MAGIC, 2, len(masks), height, width, FLAG_XOR_DELTA) + _compress_volume(
        b"".join(chunks)
    )
    restored = unpack_canny_payload(blob)
    for got, want in zip(restored, masks):
        assert np.array_equal(got, want)


def test_canny_ladder_240p_smaller_than_1080p(tmp_path: Path) -> None:
    from demo.pipeline.maps.canny import write_canny_ladder

    frames = []
    for i in range(4):
        img = np.zeros((180, 320), dtype=np.uint8)
        img[20 + i : 140, 30:280] = 255
        frames.append(img)
    streams = write_canny_ladder(frames, tmp_path / "canny", fps=10.0, n_warmup=0, n_runs=1)
    by_map = {s.map: s for s in streams}
    assert "canny" in by_map
    assert "canny_1080" not in by_map or by_map["canny"].payload_bytes <= by_map["canny_1080"].payload_bytes
    assert by_map["canny"].extra["pack_height"] == 180
    hi = by_map.get("canny_240_hi")
    if hi is not None:
        assert hi.payload_bytes <= by_map["canny"].payload_bytes * 2
