from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demo.pipeline.maps.contract import MapStream, write_sidecar
from demo.pipeline.maps.depth import (
    DINOV3_SKIP_REASON,
    colorize_turbo,
    main as depth_main,
    quantize_depth_u8,
)
from demo.pipeline.maps.encode import try_encode_gray_av1, write_u8_stack


def test_require_lists_missing_and_refuses_download() -> None:
    from demo.pipeline.maps.model_paths import MODELS, require

    if MODELS.get("yolo26s_depth") is not None:
        pytest.skip("yolo26s_depth is present")
    with pytest.raises(FileNotFoundError, match="Do not auto-download") as exc:
        require("yolo26s_depth")
    msg = str(exc.value)
    assert "yolo26s_depth" in msg
    assert "Known missing keys" in msg


def test_quantize_uses_1_99_percentile() -> None:
    rng = np.arange(10000, dtype=np.float32).reshape(100, 100)
    u8, lo, hi = quantize_depth_u8(rng)
    assert lo == pytest.approx(float(np.percentile(rng, 1)))
    assert hi == pytest.approx(float(np.percentile(rng, 99)))
    assert u8.dtype == np.uint8
    assert u8.min() == 0
    assert u8.max() == 255


def test_quantize_nan_and_constant() -> None:
    dead = np.full((4, 4), np.nan, dtype=np.float32)
    u8, lo, hi = quantize_depth_u8(dead)
    assert u8.shape == (4, 4)
    assert int(u8.max()) == 0
    assert hi > lo
    const = np.full((8, 8), 3.5, dtype=np.float32)
    u8c, _, _ = quantize_depth_u8(const)
    assert set(np.unique(u8c).tolist()) <= {0, 255}


def test_turbo_preview_is_color_not_payload() -> None:
    gray = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8)
    bgr = colorize_turbo(gray)
    assert bgr.ndim == 3 and bgr.shape[2] == 3
    assert not np.array_equal(bgr[:, :, 0], gray)


def test_u8_stack_roundtrip(tmp_path: Path) -> None:
    frames = [np.full((4, 6), i, dtype=np.uint8) for i in range(3)]
    npy = tmp_path / "depth_u8.npy"
    raw = tmp_path / "depth_u8.bin"
    write_u8_stack(frames, npy, raw)
    loaded = np.load(npy)
    assert loaded.shape == (3, 4, 6)
    assert loaded.dtype == np.uint8
    assert raw.stat().st_size == 3 * 4 * 6


def test_try_encode_gray_av1_returns_none_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("ffmpeg gray AV1 failed")

    monkeypatch.setattr("demo.pipeline.maps.encode.encode_gray_av1", _boom)
    out = try_encode_gray_av1([np.zeros((8, 8), dtype=np.uint8)], tmp_path / "x.mp4")
    assert out is None


def test_depth_cli_exits_2_when_weights_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(_key: str) -> Path:
        raise FileNotFoundError(
            "MODELS['yolo26s_depth'] is missing under /home/itec/emanuele/Models. "
            "Known missing keys: ['sam31', 'yolo26s_depth']. Do not auto-download."
        )

    monkeypatch.setattr("demo.pipeline.maps.depth.require", _missing)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    code = depth_main(["--clip", str(clip), "--out", str(tmp_path / "out")])
    assert code == 2
    err = capsys.readouterr().err
    assert "yolo26s_depth" in err
    assert "Do not auto-download" in err
    assert "MISSING" in err or "missing keys" in err.lower()


def test_dinov3_skips_without_dpt_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    fake.write_bytes(b"not-a-real-ckpt")
    monkeypatch.setitem(__import__("demo.pipeline.maps.model_paths", fromlist=["MODELS"]).MODELS, "dinov3_vits", fake)
    out = tmp_path / "dinov3"
    code = depth_main(["--clip", str(tmp_path / "clip.mp4"), "--out", str(out), "--backend", "dinov3"])
    assert code == 0
    skipped = (out / "skipped.json").read_text()
    assert "DPT" in skipped
    assert DINOV3_SKIP_REASON.split("Skip")[0][:40] in skipped or "ViT-S" in skipped
    assert "skipped" in skipped.lower()


def test_dinov3_missing_weights_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.pipeline.maps.model_paths as mp

    monkeypatch.setitem(mp.MODELS, "dinov3_vits", None)
    code = depth_main(["--clip", str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o"), "--backend", "dinov3"])
    assert code == 2
    assert "dinov3_vits" in capsys.readouterr().err


def test_depth_sidecar_rejects_preview_payload(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        MapStream(
            map="depth",
            backend="yolo26s-depth.pt",
            payload_path=str(tmp_path / "preview_depth.mp4"),
            payload_bytes=10,
            preview_path=str(tmp_path / "preview_depth.mp4"),
            preview_bytes=10,
            duration_s=1.0,
            n_frames=1,
            fps=30.0,
            extract_ms_p50=1.0,
            extract_ms_p95=2.0,
            pack_ms_p50=0.1,
            codec_ms_p50=0.0,
            decode_ms_p50=0.1,
            gpu="cpu",
        )


def test_depth_native_sidecar_writes_kbps(tmp_path: Path) -> None:
    payload = tmp_path / "payload_depth.av1.mp4"
    payload.write_bytes(b"not-av1-but-native")
    stream = MapStream(
        map="depth",
        backend="yolo26n-depth.pt",
        payload_path=str(payload),
        payload_bytes=payload.stat().st_size,
        preview_path=str(tmp_path / "preview_depth.mp4"),
        preview_bytes=99,
        duration_s=2.0,
        n_frames=60,
        fps=30.0,
        extract_ms_p50=1.0,
        extract_ms_p95=2.0,
        pack_ms_p50=0.1,
        codec_ms_p50=0.0,
        decode_ms_p50=0.1,
        gpu="cpu",
        extra={"quantize": "percentile_1_99"},
    )
    path = write_sidecar(stream, tmp_path / "depth.json")
    text = path.read_text()
    assert '"map": "depth"' in text
    assert "payload_kbps" in text
    assert "yolo26n-depth.pt" in text
    assert "percentile_1_99" in text


@pytest.mark.integration
def test_depth_yolo_cli_integration() -> None:
    pytest.importorskip("ultralytics")
    from demo.pipeline.maps.model_paths import MODELS

    if MODELS.get("yolo26s_depth") is None:
        pytest.skip("yolo26*-depth.pt not in MODELS")
    pytest.skip("requires a real clip on gpu1")
