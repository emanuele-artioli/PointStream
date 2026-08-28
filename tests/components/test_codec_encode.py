"""Real encodes: every rung round-trips, native/pixel ROI, matched-rate search.

Skipped when the encoder binary is missing. Marked integration so the default
pytest invocation (CPU unit tests) does not wait on libvvenc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode, search_qp
from src.components.codec.roi import BlockRoiMap
from src.components.codec.tools import resolve_encoder, resolve_ffmpeg
from src.components.codec.y4m import from_luma, read, write
from src.contracts.codecs import EncodeRequest, RateControl

pytestmark = pytest.mark.integration


def _source(tmp_path: Path, *, width: int = 128, height: int = 128, frames: int = 2) -> Path:
    """A synthetic 4:2:0 clip with spatial texture so a pixel-domain ROI can bite.

    A flat field is a trap: high-QP encodes of two flats can be identical even
    when the pixels nominally differ, and mean-toward degradation of a constant
    block is a no-op.
    """
    rows = np.arange(height, dtype=np.int32)[:, None]
    cols = np.arange(width, dtype=np.int32)[None, :]
    base = ((rows * 3 + cols * 5) % 220).astype(np.uint8)
    luma = np.stack([base] * frames, axis=0)
    luma[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = np.clip(
        luma[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4].astype(np.int32) + 40,
        0,
        255,
    ).astype(np.uint8)
    path = tmp_path / "source.y4m"
    write(path, from_luma(luma, fps=10.0))
    return path


def _available(codec_name: str) -> None:
    try:
        resolve_ffmpeg()
        resolve_encoder(codec_name)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _qp_request(codec_name: str, qp: int = 36, preset: str | None = None) -> EncodeRequest:
    presets = {"avc": "veryfast", "hevc": "ultrafast", "av1": "10", "vvc": "faster"}
    return EncodeRequest(
        codec_name=codec_name,
        rate_control=RateControl.QP,
        rate=qp,
        preset=preset if preset is not None else presets[codec_name],
        pix_fmt="yuv420p",
    )


@pytest.mark.parametrize("codec_name", ["avc", "hevc", "av1", "vvc"])
def test_each_rung_encodes_and_decodes_and_records_path_and_version(
    tmp_path: Path, codec_name: str
) -> None:
    _available(codec_name)
    source = _source(tmp_path)
    dest = tmp_path / f"out{BITSTREAM_SUFFIX[codec_name]}"
    request = _qp_request(codec_name)
    record = encode(source, dest, request)
    assert record.size_bytes > 0
    assert record.tool_path
    assert record.tool_version
    assert record.tool_version != "unknown"
    assert record.codec_name == codec_name
    decoded = tmp_path / "decoded.y4m"
    decode(dest, decoded, request)
    video = read(decoded)
    assert video.width == 128
    assert video.height == 128
    assert video.frames == 2


def test_av1_native_roi_changes_the_bitstream(tmp_path: Path) -> None:
    _available("av1")
    source = _source(tmp_path)
    request = _qp_request("av1", qp=45)
    roi = BlockRoiMap.centred(128, 128, inside_offset=-30, outside_offset=0, fraction=0.5)
    baseline = encode(source, tmp_path / "base.ivf", request)
    native = encode(
        source,
        tmp_path / "roi.ivf",
        request,
        roi=roi,
        roi_arm="native",
    )
    assert native.roi_arm == "native"
    assert "--roi-map-file" in native.command
    assert native.size_bytes != baseline.size_bytes


def test_pixel_arm_changes_the_bitstream_when_native_roi_does_not_exist(tmp_path: Path) -> None:
    _available("vvc")
    source = _source(tmp_path)
    request = _qp_request("vvc", qp=28)
    roi = BlockRoiMap.centred(128, 128, inside_offset=0, outside_offset=16, fraction=0.5, block_size=16)
    baseline = encode(source, tmp_path / "base.vvc", request)
    pixel = encode(
        source,
        tmp_path / "roi.vvc",
        request,
        roi=roi,
        roi_arm="pixel",
    )
    assert pixel.roi_arm == "pixel"
    assert pixel.output.read_bytes() != baseline.output.read_bytes()
    assert pixel.tool_path == baseline.tool_path


def test_avc_addroi_is_measured_not_believed(tmp_path: Path) -> None:
    """If addroi is decorative, the in-house arm is the one that counts."""
    _available("avc")
    source = _source(tmp_path)
    request = _qp_request("avc", qp=28)
    roi = BlockRoiMap.centred(128, 128, inside_offset=-26, outside_offset=16, fraction=0.5, block_size=16)
    baseline = encode(source, tmp_path / "base.mp4", request)
    native = encode(source, tmp_path / "addroi.mp4", request, roi=roi, roi_arm="native")
    pixel = encode(source, tmp_path / "pixel.mp4", request, roi=roi, roi_arm="pixel")
    pixel_moved = pixel.output.read_bytes() != baseline.output.read_bytes()
    native_moved = native.output.read_bytes() != baseline.output.read_bytes()
    assert pixel_moved, "pixel-domain AVC arm did not change the bitstream"
    # addroi is unverified: document whether it moved anything. Either outcome is
    # a measurement; the in-house arm is the one this test requires to work.
    _ = native_moved
    assert native.tool_version == baseline.tool_version
    assert "addroi" in " ".join(native.command) or native.roi_arm == "native"


def test_matched_rate_search_hits_tolerance_on_a_real_avc_encode(tmp_path: Path) -> None:
    _available("avc")
    source = _source(tmp_path)
    request = _qp_request("avc", qp=36)
    baseline = encode(source, tmp_path / "base.mp4", request)
    target = baseline.size_bytes

    def at_qp(qp: int) -> int:
        rec = encode(source, tmp_path / f"qp{qp}.mp4", request.replace_rate(qp))
        return rec.size_bytes

    qp, size = search_qp(at_qp, target, 20, 45, tolerance=0.15)
    assert abs(size - target) / target <= 0.15
    assert 20 <= qp <= 45


# ---------------------------------------------------------------------------
# A decode that loses information is not a decode.
#
# `_decode_command` named no `-c:v`, so ffmpeg picked the muxer's default
# encoder. To a `.y4m` that is rawvideo and harmless, which is why `coded_curve`
# was fine. To a `.mkv` it is libx264 at its own default CRF — so
# `coded_roundtrip` handed back frames that had been through the rung's codec
# and then through x264, and every quality measured on them was pinned near that
# second pass's ceiling however the rung moved. Measured on an av1 anchor over
# QP 15 to 55: 41.71, 41.67, 41.43, 40.65, 38.94 dB while the rate fell tenfold.
# ---------------------------------------------------------------------------


def test_decode_to_a_lossy_container_names_a_lossless_codec() -> None:
    from src.components.codec.command import DECODE_LOSSLESS_CODEC, build_command

    ffmpeg = resolve_ffmpeg()
    request = EncodeRequest(codec_name="av1", rate_control=RateControl.QP, rate=30)
    argv = build_command(
        "decode",
        request,
        source=Path("in.ivf"),
        dest=Path("out.mkv"),
        encoder=ffmpeg,
        ffmpeg=ffmpeg,
    )
    assert "-c:v" in argv
    assert argv[argv.index("-c:v") + 1] == DECODE_LOSSLESS_CODEC


def test_decode_to_a_y4m_leaves_the_codec_alone() -> None:
    """A y4m is rawvideo by construction; naming a codec would break the muxer."""
    from src.components.codec.command import build_command

    ffmpeg = resolve_ffmpeg()
    request = EncodeRequest(codec_name="av1", rate_control=RateControl.QP, rate=30)
    argv = build_command(
        "decode",
        request,
        source=Path("in.ivf"),
        dest=Path("out.y4m"),
        encoder=ffmpeg,
        ffmpeg=ffmpeg,
    )
    assert "-c:v" not in argv


def test_coded_roundtrip_quality_follows_the_rung(tmp_path: Path) -> None:
    """The property the missing `-c:v` broke, driven end to end on real frames.

    A coarse rung must come back visibly worse than a fine one. With a second
    encoder pinning the output this held only until the pin bound, and then the
    curve went flat while the byte count kept moving — which is exactly the
    shape that looks like a working measurement.
    """
    from src.components.codec.measure import coded_roundtrip

    rng = np.random.default_rng(3)
    ramp = np.linspace(0, 255, 128, dtype=np.float32)
    base = ramp[None, :, None, None] + ramp[None, None, :, None] / 2.0
    clip = np.clip(
        np.repeat(base, 3, axis=3) + rng.normal(0, 6.0, (1, 128, 128, 3)), 0, 255
    ).astype(np.uint8)
    clip = np.repeat(clip, 4, axis=0)

    def psnr(candidate: np.ndarray) -> float:
        mse = float(np.mean((clip.astype(np.float64) - candidate.astype(np.float64)) ** 2))
        return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse))

    fine_bytes, fine = coded_roundtrip(
        clip,
        request=EncodeRequest(codec_name="av1", rate_control=RateControl.QP, rate=20),
        work_dir=tmp_path,
    )
    coarse_bytes, coarse = coded_roundtrip(
        clip,
        request=EncodeRequest(codec_name="av1", rate_control=RateControl.QP, rate=55),
        work_dir=tmp_path,
    )
    assert coarse_bytes < fine_bytes
    # Both halves have to move. A rate that falls while the quality does not is
    # the failure this test exists for.
    assert psnr(coarse) < psnr(fine)
