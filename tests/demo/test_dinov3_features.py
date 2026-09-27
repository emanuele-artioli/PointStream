"""Pack/PCA helpers for DINOv3 patch maps — no real backbone required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demo.pipeline.maps.contract import MapStream, write_sidecar
from demo.pipeline.maps.dinov3_features import (
    INT8_SCALE,
    POOLED_HW,
    extra_from_features,
    local_hf_config_dir,
    missing_code_message,
    pack_int8_features,
    pca_to_rgb,
    quantize_int8,
    spatial_pool,
    tokens_to_patch_map,
    unpack_int8_features,
    upsample_rgb,
    zscore_per_channel,
)


def _fake_features(hp: int = 16, wp: int = 24, c: int = 8, t: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    # Spatial ramp on channel 0 so PCA is structured, not white noise.
    yy, xx = np.mgrid[0:hp, 0:wp]
    ramp = (xx.astype(np.float32) / max(wp - 1, 1)) + (yy.astype(np.float32) / max(hp - 1, 1))
    feat = rng.normal(0.0, 0.05, size=(t, hp, wp, c)).astype(np.float32)
    feat[..., 0] += ramp
    return feat


def test_spatial_pool_identity_and_down_up() -> None:
    feat = _fake_features(32, 32, 6, t=1)[0]
    pooled = spatial_pool(feat, 32, 32)
    assert pooled.shape == (32, 32, 6)
    np.testing.assert_allclose(pooled, feat, atol=1e-5)

    down = spatial_pool(_fake_features(64, 64, 5, t=1)[0], 32, 32)
    assert down.shape == (32, 32, 5)

    up = spatial_pool(_fake_features(8, 10, 4, t=1)[0], 32, 32)
    assert up.shape == (32, 32, 4)


def test_pack_int8_handles_vits_channel_count() -> None:
    rng = np.random.default_rng(1)
    feat = rng.normal(size=(8, 8, 384)).astype(np.float32)
    blob = pack_int8_features(feat)
    q, _scale = unpack_int8_features(blob)
    assert q.shape == (1, 32, 32, 384)
    assert q.dtype == np.int8


def test_pack_int8_roundtrip_on_fake_hwc() -> None:
    feat = _fake_features()
    blob = pack_int8_features(feat)
    assert isinstance(blob, (bytes, bytearray))
    assert len(blob) < feat.nbytes
    quantized, scale = unpack_int8_features(blob)
    assert quantized.dtype == np.int8
    assert quantized.shape == (feat.shape[0], POOLED_HW[0], POOLED_HW[1], feat.shape[-1])
    assert scale == pytest.approx(INT8_SCALE)
    assert int(quantized.min()) >= -127
    assert int(quantized.max()) <= 127

    single = pack_int8_features(feat[0])
    q1, _ = unpack_int8_features(single)
    assert q1.shape == (1, 32, 32, feat.shape[-1])


def test_zscore_then_int8_matches_ticket_scale() -> None:
    feat = spatial_pool(_fake_features(16, 16, 3, t=1)[0], 32, 32)
    z = zscore_per_channel(feat)
    assert z.mean() == pytest.approx(0.0, abs=1e-5)
    q = quantize_int8(z)
    expected = np.clip(np.rint(z * INT8_SCALE), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(q, expected)


def test_pca_rgb_of_random_features() -> None:
    feat = _fake_features(20, 12, 7, t=4)
    rgb, components, mean = pca_to_rgb(feat)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (4, 20, 12, 3)
    assert components.shape == (3, 7)
    assert mean.shape == (7,)
    # Channel-0 ramp should dominate PC visualization: not all pixels equal.
    assert rgb.std() > 5.0
    up = upsample_rgb(rgb[0], 80, 96)
    assert up.shape == (80, 96, 3)
    assert up.dtype == np.uint8


def test_sidecar_extra_has_c_hp_wp_pooled_int8(tmp_path: Path) -> None:
    feat = [_fake_features(14, 18, 9, t=1)[0], _fake_features(14, 18, 9, t=1)[0]]
    extra = extra_from_features(feat)
    assert extra == {"C": 9, "Hp": 14, "Wp": 18, "pooled": [32, 32], "dtype": "int8"}
    payload = pack_int8_features(np.stack(feat, axis=0))
    payload_path = tmp_path / "feat_32x32_int8.bin"
    payload_path.write_bytes(payload)
    preview = tmp_path / "preview_pca.mp4"
    preview.write_bytes(b"not-counted")
    stream = MapStream(
        map="dino_feat",
        backend="dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        payload_path=str(payload_path),
        payload_bytes=payload_path.stat().st_size,
        preview_path=str(preview),
        preview_bytes=preview.stat().st_size,
        duration_s=2.0,
        n_frames=2,
        fps=30.0,
        extract_ms_p50=1.0,
        extract_ms_p95=2.0,
        pack_ms_p50=0.2,
        codec_ms_p50=0.0,
        decode_ms_p50=0.1,
        gpu="cpu",
        extra=extra,
    )
    sidecar_path = tmp_path / "sidecar.json"
    write_sidecar(stream, sidecar_path)
    text = sidecar_path.read_text()
    assert '"C": 9' in text
    assert '"Hp": 14' in text
    assert '"Wp": 18' in text
    assert '"pooled": [' in text
    assert '"dtype": "int8"' in text
    assert stream.payload_kbps == pytest.approx((payload_path.stat().st_size * 8.0) / 2.0 / 1000.0)
    assert "not-counted" not in str(stream.payload_kbps)


def test_missing_code_message_names_pth() -> None:
    pth = Path("/home/itec/emanuele/Models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    msg = missing_code_message(pth)
    assert str(pth) in msg
    assert "dinov3 code is not" in msg
    assert "Hub" in msg


def test_drop_cls_and_register_tokens() -> None:
    hp, wp, c = 4, 4, 8
    n_reg = 4
    cls = np.zeros((1, c), dtype=np.float32)
    reg = np.ones((n_reg, c), dtype=np.float32)
    patches = np.arange(hp * wp * c, dtype=np.float32).reshape(hp * wp, c)
    tokens = np.concatenate([cls, reg, patches], axis=0)
    out = tokens_to_patch_map(tokens, height=64, width=64, patch=16, n_register=n_reg)
    assert out.shape == (hp, wp, c)
    np.testing.assert_allclose(out.reshape(-1, c), patches)


def test_local_hf_config_dir_requires_config_json(tmp_path: Path) -> None:
    pth = tmp_path / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    pth.write_bytes(b"x")
    assert local_hf_config_dir(pth) is None
    (tmp_path / "config.json").write_text("{}")
    assert local_hf_config_dir(pth) == tmp_path


def test_cli_exit_2_when_pth_present_without_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from demo.pipeline.maps import dinov3_features as mod

    pth = tmp_path / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    pth.write_bytes(b"not-a-real-checkpoint")
    monkeypatch.setattr(mod, "require", lambda key: pth)
    monkeypatch.setattr(mod, "_facebook_code_available", lambda: False)
    monkeypatch.setattr(mod, "local_hf_config_dir", lambda _p: None)
    code = mod.main(["--clip", str(tmp_path / "clip.mp4"), "--out", str(tmp_path / "out")])
    assert code == 2
