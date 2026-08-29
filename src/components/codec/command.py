"""One command builder for every rung: AVC, HEVC, AV1, VVC, encode and decode.

This is the replacement for the scattered ffmpeg construction sites. Drivers
differ — ffmpeg ``-c:v`` versus a standalone binary — but every encode and
every decode goes through ``build_command``. Callers that assemble argv by
hand are the failure mode this module exists to make unnecessary.

``libsvtav1`` is never emitted. AV1 is ``SvtAv1EncApp``. Asking that binary
(or a sneaked ``libsvtav1`` extra-arg) for ``yuv444p`` raises: the wrapper
accepts the flag, returns success, and writes yuv420p.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from src.contracts.codecs import Driver, EncodeRequest, RateControl
from src.contracts.errors import CodecConstraintError
from src.components.codec.tools import ResolvedTool

Kind = Literal["encode", "decode"]


def build_command(
    kind: Kind,
    request: EncodeRequest,
    *,
    source: Path | str,
    dest: Path | str,
    encoder: ResolvedTool,
    ffmpeg: ResolvedTool | None = None,
    roi_file: Path | str | None = None,
    addroi: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> list[str]:
    """Build an encode or decode argv for ``request``.

    Args:
        kind: ``encode`` or ``decode``. Decode is always ffmpeg.
        request: Validated against codec capabilities before any argv is built.
        source, dest: Input and output paths.
        encoder: The rung's encoder (ffmpeg, kvazaar, or SvtAv1EncApp).
        ffmpeg: Required for decode, and recorded even when the encoder is a
            binary. Encode of ffmpeg rungs uses ``encoder`` as ffmpeg.
        roi_file: Native delta-QP map already formatted for this encoder.
        addroi: ffmpeg ``addroi=...`` filter string for the AVC native arm.
        width, height, fps: Required by kvazaar (no y4m header trust). Optional
            for the others; y4m supplies them.
    """
    request.validate()
    _reject_libsvtav1_yuv444(request)

    if kind == "decode":
        if ffmpeg is None:
            raise ValueError("decode requires ffmpeg")
        return _decode_command(ffmpeg, source=source, dest=dest, pix_fmt=request.pix_fmt)

    caps = request.capabilities
    if caps.driver is Driver.FFMPEG:
        if caps.invocation == "libsvtav1":
            # Defensive: the ladder does not register this invocation. If it
            # ever did, refusing here is cheaper than emitting yuv420p silently.
            raise CodecConstraintError(
                caps.name,
                "invocation",
                "libsvtav1",
                ["SvtAv1EncApp"],
            )
        return _ffmpeg_encode(
            encoder,
            request,
            source=source,
            dest=dest,
            addroi=addroi,
        )
    if caps.name == "hevc":
        return _kvazaar_encode(
            encoder,
            request,
            source=source,
            dest=dest,
            roi_file=roi_file,
            width=width,
            height=height,
            fps=fps,
        )
    if caps.name == "av1":
        if roi_file is not None and not encoder.has("roi-map-file"):
            raise CodecConstraintError(
                "av1",
                "roi-map-file",
                f"{encoder.path} ({encoder.version})",
                ["an SvtAv1EncApp build that lists --roi-map-file in --help"],
            )
        return _svtav1_encode(
            encoder,
            request,
            source=source,
            dest=dest,
            roi_file=roi_file,
            width=width,
            height=height,
        )
    raise CodecConstraintError(
        request.codec_name,
        "driver",
        caps.driver.value,
        ["ffmpeg", "binary"],
    )


def _reject_libsvtav1_yuv444(request: EncodeRequest) -> None:
    """The silent-substitution bug, as a check on extra_args too.

    ``EncodeRequest.validate`` already rejects av1+yuv444p. This catches a
    caller stuffing ``-c:v libsvtav1`` into extra_args on some other codec
    while asking for 4:4:4 — the exact command that used to succeed and lie.
    """
    extras = " ".join(request.extra_args)
    if "libsvtav1" in extras and request.pix_fmt != "yuv420p":
        raise CodecConstraintError(
            request.codec_name,
            "pix_fmt with libsvtav1",
            request.pix_fmt,
            ["yuv420p"],
        )


#: Lossless intermediate for a decode whose container is not a y4m.
#:
#: **A decode that loses information is not a decode.** Without an explicit
#: ``-c:v`` ffmpeg picks the muxer's default encoder, which for Matroska is
#: libx264 at its own default CRF — so `coded_roundtrip` was handing back frames
#: that had been through the rung's codec *and then* through x264. Measured
#: 2026-08-28: an av1 anchor swept over QP 15 to 55 returned 41.71, 41.67,
#: 41.43, 40.65 and 38.94 dB of Y-PSNR, pinned near the x264 pass's own ceiling
#: while its rate fell tenfold. The rung reached the encoder; the quality never
#: reached the measurement.
#:
#: A y4m is rawvideo by construction, so it needs no codec named — which is why
#: `coded_curve`, which decodes to y4m, was never affected and `coded_roundtrip`,
#: which decodes to mkv, was.
DECODE_LOSSLESS_CODEC: Final = "ffv1"


def _decode_command(
    ffmpeg: ResolvedTool,
    *,
    source: Path | str,
    dest: Path | str,
    pix_fmt: str,
) -> list[str]:
    command = [
        ffmpeg.path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-pix_fmt",
        pix_fmt,
    ]
    if Path(dest).suffix.lower() != ".y4m":
        command += ["-c:v", DECODE_LOSSLESS_CODEC]
    command.append(str(dest))
    return command


def _ffmpeg_encode(
    ffmpeg: ResolvedTool,
    request: EncodeRequest,
    *,
    source: Path | str,
    dest: Path | str,
    addroi: str | None,
) -> list[str]:
    caps = request.capabilities
    command = [
        ffmpeg.path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        caps.invocation,
    ]
    if addroi:
        # format= after addroi so libx264 sees 4:2:0 even when the source is RGB.
        command += ["-vf", f"{addroi},format={request.pix_fmt}"]
    if request.preset is not None:
        command += ["-preset", request.preset]
    command += _ffmpeg_rate_args(request)
    # libvvenc only emits 10-bit 4:2:0. Passing -pix_fmt yuv420p either fails or
    # would be a lie; the source y4m is already 4:2:0 and ffmpeg upconverts.
    if caps.name != "vvc":
        command += ["-pix_fmt", request.pix_fmt]
    command += list(request.extra_args)
    command.append(str(dest))
    return command


def _ffmpeg_rate_args(request: EncodeRequest) -> list[str]:
    if request.rate_control is RateControl.LOSSLESS:
        # x264: qp 0 is mathematically lossless at 8-bit.
        return ["-qp", str(request.capabilities.lossless_rate or 0)]
    if request.rate is None:
        return []
    if request.rate_control is RateControl.QP:
        args = ["-qp", str(request.rate)]
        # Perceptual QP adaptation fights a pixel-domain ROI arm and a matched-QP
        # comparison. Turn it off when the request asked for a real QP.
        if request.capabilities.name == "vvc":
            args += ["-qpa", "0"]
        return args
    if request.rate_control is RateControl.CRF:
        return ["-crf", str(request.rate)]
    if request.rate_control is RateControl.BITRATE:
        return ["-b:v", str(request.rate)]
    return []


def _kvazaar_encode(
    encoder: ResolvedTool,
    request: EncodeRequest,
    *,
    source: Path | str,
    dest: Path | str,
    roi_file: Path | str | None,
    width: int | None,
    height: int | None,
    fps: float | None,
) -> list[str]:
    command = [encoder.path, "--input", str(source), "--output", str(dest)]
    if width is not None and height is not None:
        command += ["--input-res", f"{width}x{height}"]
    if fps is not None:
        command += ["--input-fps", _fps_arg(fps)]
    if request.preset is not None:
        command += ["--preset", request.preset]
    if request.rate_control is RateControl.QP and request.rate is not None:
        command += ["--qp", str(request.rate)]
    elif request.rate_control is RateControl.BITRATE and request.rate is not None:
        command += ["--bitrate", str(request.rate)]
    if roi_file is not None:
        command += ["--roi", str(roi_file)]
    command += list(request.extra_args)
    return command


def _svtav1_encode(
    encoder: ResolvedTool,
    request: EncodeRequest,
    *,
    source: Path | str,
    dest: Path | str,
    roi_file: Path | str | None,
    width: int | None,
    height: int | None,
) -> list[str]:
    command = [
        encoder.path,
        "-i",
        str(source),
        "-b",
        str(dest),
        "--progress",
        "0",
    ]
    if width is not None:
        command += ["--width", str(width)]
    if height is not None:
        command += ["--height", str(height)]
    if request.preset is not None:
        command += ["--preset", request.preset]
    command += _svtav1_rate_args(request, roi=roi_file is not None or request.is_roi_arm)
    if roi_file is not None:
        command += ["--roi-map-file", str(roi_file)]
    command += list(request.extra_args)
    return command


def _svtav1_rate_args(request: EncodeRequest, *, roi: bool = False) -> list[str]:
    if request.rate is None:
        return []
    if request.rate_control is RateControl.CRF:
        return ["--crf", str(request.rate)]
    if request.rate_control is RateControl.QP:
        # --crf is --rc 0 --aq-mode 2 --qp x. Without a map, force CQP
        # (--aq-mode 0) so a QP request is actually QP. With a map, leave
        # aq-mode at the encoder default: the 1.8.0 localisation measurement
        # used --rc 0 --qp and default aq-mode 2, and forcing aq-mode 1 on a
        # 2-frame clip made both regions worse.
        if roi:
            return ["--rc", "0", "--qp", str(request.rate)]
        return ["--rc", "0", "--aq-mode", "0", "--qp", str(request.rate)]
    if request.rate_control is RateControl.BITRATE:
        # --tbr is kbps by default; the `b` suffix keeps the contract unit (bps).
        return ["--rc", "1", "--tbr", f"{request.rate}b"]
    return []


def _fps_arg(fps: float) -> str:
    if fps == int(fps):
        return str(int(fps))
    return f"{int(round(fps * 1000))}/1000"
