import hashlib
from fractions import Fraction
from pathlib import Path

import pytest

from experiments.visor import h5
from src.pointstream import bitstream as bs


def test_points_parse_and_name() -> None:
    assert h5.parse_point("dcvc@21/4") == (21, 4)
    assert h5.point_key("dcvc@0/2") == "dcvc-0_2"
    assert h5.coded_size(4) == (480, 272) and h5.coded_size(1) == (1920, 1080)
    with pytest.raises(SystemExit):
        h5.parse_point("svt@21/3")


def av1_ivf(frames: int) -> bytes:
    td = bytes([bs.OBU_TEMPORAL_DELIMITER << 3 | 0x02, 0])
    frame = bytes([6 << 3 | 0x02]) + bs.leb128(40) + bytes(40)
    return bs.obus_to_ivf((td + frame) * frames, 1920, 1080, Fraction(50))


def published(tmp: Path, name: str, data: bytes) -> tuple[str, str]:
    (tmp / name).write_bytes(data)
    return name, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("route", ["enh", "scene"])
def test_container_carries_the_published_streams(tmp_path: Path, route: str) -> None:
    if route == "enh":
        bg_file, bg_sha = published(tmp_path, "bg.ivf", av1_ivf(6))
        bg = {"key": "enh-1080-48", "route": "enh", "codec": "enh", "coded_size": [1920, 1080],
              "model_sha256": "11" * 32}
    else:
        bg_file, bg_sha = published(tmp_path, "bg.psdc", b"PSDC" + bytes(300))
        bg = {"key": "scene-540-13", "route": "scene", "codec": "scene", "coded_size": [960, 544],
              "model_sha256": "22" * 32}
    bg.update(root=str(tmp_path), bitstream=bg_file, bitstream_sha256=bg_sha)
    fg_file, fg_sha = published(tmp_path, "fg.psdc", b"PSDC" + bytes(range(200)))
    fg = {"key": "dcvc-21_4", "scale": 4, "coded_size": [480, 272], "bitstream": fg_file, "bitstream_sha256": fg_sha,
          "slots": {"hand": {"sha256": "33" * 32}, "object": None}}
    w = {"fps": Fraction(50), "frames": 6}
    c = h5.container_for(w, bg, fg, tmp_path)
    data = c.pack()
    checks = h5.bitstream_check(c, data, bg, fg, tmp_path)
    assert all(checks.values()), checks
    parts = c.breakdown()
    assert sum(parts.values()) == len(data)
    if route == "enh":  # AV1 is billed without IVF's framing, as B2 counts SVT-AV1
        assert parts["background"] < (tmp_path / bg_file).stat().st_size
    assert [m.kind for m in c.models()] == ["background", "foreground"]


def foreground_result(**change: object) -> dict:
    rows = []
    for scale in (4, 2, 1):
        for qp, kbps in ((0, 50.0), (21, 20.0), (42, 8.0)):
            rows.append({"scale": scale, "qp": qp, "kbps": kbps * (5 - scale), "deterministic": True,
                         "decoder_matches_encoder_intra": True, "psnr_f": 30.0 + (42 - qp) / 10,
                         "slots": {"hand": {"sha256": "x"} if scale in (2, 4) else None},
                         "changed_by_slots": 0.05 if scale in (2, 4) else 0.0, "decoded_frame_sha256": ["a"] * 240})
    result = {"kind": "foreground", "points": rows, "frames": 240, "limit_frames": 0, "jpeg_gate_holds": True,
              "mask_cost": {"bytes": 1000, "share": 0.3}}
    result.update(change)
    return result


def test_foreground_validator() -> None:
    assert all(h5.validate_foreground(foreground_result()).values())
    bad = foreground_result()
    bad["points"][1]["kbps"] = 999.0  # QP 21 dearer than QP 0
    assert not h5.validate_foreground(bad)["rate_falls_with_qp"]
    bad = foreground_result()
    bad["points"][0]["deterministic"] = False
    assert not h5.validate_foreground(bad)["every_point_coded"]


def scores(wpsnr: float = 32.0) -> dict:
    m = {"wpsnr": wpsnr, "wlpips": 0.2, "wdists": 0.15, "psnr_fg": 30.0}
    return {h5.DENSE: dict(m), h5.FILL: dict(m), "frame": {"psnr": 33.0}}


def test_score_validator_checks_anchor_reproduction() -> None:
    row = {"bytes": 30, "breakdown": {"header": 20, "background": 10}, "kbps": 0.06, "scores": scores()}
    anchor = {"deterministic": True, "stream_sha256": "s", "kbps": 100.0, "scores": scores(32.0),
              "b2": {"stream_sha256": "s", "kbps": 100.0, "wpsnr": 32.01}}
    result = {"rows": [row], "anchors": [anchor], "pairs_planned": 1, "jpeg_gate_holds": True, "limit_frames": 0,
              "routes": ["scene"], "bitstream_checks": {"scene": {"pair": "p", "segments_round_trip": True}},
              "background_f_is_foreground_f": True}
    assert all(h5.validate_score(result).values())
    anchor["b2"]["wpsnr"] = 32.5
    assert not h5.validate_score(result)["anchors_reproduce_b2"]
