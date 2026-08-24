"""Encode a clip at several QPs and score PSNR on the whole frame and on regions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.components.codec.encode import BITSTREAM_SUFFIX, EncodeRecord, decode, encode
from src.components.codec.tools import resolve_encoder, resolve_ffmpeg
from src.components.codec.y4m import from_luma, read, write
from src.components.metrics.bd_rate import RDCurve
from src.components.metrics.psnr import PsnrMetric, masked_psnr
from src.contracts.codecs import EncodeRequest, RateControl
from experiments.headroom.remove import as_mask, even_size, rgb_to_luma

DEFAULT_QPS: tuple[int, ...] = (32, 40, 48)
DEFAULT_CODEC = "avc"


def encoders_available(codec_name: str = DEFAULT_CODEC) -> bool:
    try:
        resolve_ffmpeg()
        resolve_encoder(codec_name)
    except FileNotFoundError:
        return False
    return True


def resolved_tools(codec_name: str = DEFAULT_CODEC) -> dict[str, str]:
    """Path and version of the binaries this process will actually run."""
    ffmpeg = resolve_ffmpeg()
    encoder = resolve_encoder(codec_name)
    return {
        "ffmpeg_path": ffmpeg.path,
        "ffmpeg_version": ffmpeg.version,
        "encoder_path": encoder.path,
        "encoder_version": encoder.version,
        "codec_name": codec_name,
    }


def qps_for_codec(codec_name: str, qps: tuple[int, ...]) -> tuple[int, ...]:
    """libvvenc 1.11.0 writes an empty 4K bitstream at QP 48 on smooth fills.

    Original tennis pixels encode at 48; plate/flat do not (exit 0, 0 bytes).
    QP 47 still produces a stream. Keep the three-point curve; do not pretend 48 ran.
    """
    if codec_name != "vvc":
        return qps
    return tuple(47 if qp >= 48 else qp for qp in qps)


def encode_qp_with_vvc_fallback(
    source: Path,
    work_dir: Path,
    codec_name: str,
    qp: int,
    notes: list[str],
    label: str,
) -> tuple[int, Path, EncodeRecord | None]:
    """Encode at ``qp``. If libvvenc writes 0 bytes, step QP down rather than abort.

    Federer plate is empty at 47 and fine at 46. Alcaraz plate is fine at 47.
    The third curve point must exist; pretending QP 48/47 ran is the failure.
    """
    suffix = BITSTREAM_SUFFIX[codec_name]
    floor = 32
    tries = range(qp, floor - 1, -1) if codec_name == "vvc" else (qp,)
    last_error: Exception | None = None
    for try_qp in tries:
        dest = work_dir / f"{codec_name}_qp{try_qp}{suffix}"
        if dest.exists() and dest.stat().st_size > 0:
            if try_qp != qp:
                notes.append(
                    f"{codec_name} {label} QP {qp} empty; using existing QP {try_qp}".strip()
                )
            return try_qp, dest, None
        try:
            record = encode(source, dest, qp_request(codec_name, try_qp))
            if try_qp != qp:
                notes.append(
                    f"{codec_name} {label} QP {qp} wrote 0 bytes with libvvenc; "
                    f"encoded at QP {try_qp}"
                )
            return try_qp, dest, record
        except RuntimeError as exc:
            last_error = exc
            if codec_name != "vvc":
                raise
            print(f"vvc qp={try_qp} empty for {label}; trying lower", flush=True)
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink()
            continue
    raise RuntimeError(
        f"vvc would not emit a bitstream for {label or source} at QP <= {qp}"
    ) from last_error


def qp_request(codec_name: str, qp: int) -> EncodeRequest:
    presets = {"avc": "veryfast", "hevc": "ultrafast", "av1": "10", "vvc": "faster"}
    return EncodeRequest(
        codec_name=codec_name,
        rate_control=RateControl.QP,
        rate=int(qp),
        preset=presets[codec_name],
        pix_fmt="yuv420p",
    )


def encode_luma_curve(
    frames: np.ndarray,
    *,
    work_dir: Path,
    qps: tuple[int, ...] = DEFAULT_QPS,
    codec_name: str = DEFAULT_CODEC,
    masks: np.ndarray | None = None,
    label: str = "",
    fps: float = 25.0,
) -> dict[str, object]:
    """Encode RGB ``frames`` at ``qps``. Quality is PSNR against this clip's luma."""
    if not encoders_available(codec_name):
        raise FileNotFoundError(f"encoder for {codec_name} is not on this host")
    used_qps = qps_for_codec(codec_name, qps)
    notes: list[str] = []
    if used_qps != tuple(qps):
        notes.append(
            f"{codec_name} QPs {tuple(qps)} remapped to {used_qps}: "
            "libvvenc 1.11.0 writes an empty bitstream at QP 48 on some 4K fills"
        )
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    clip = even_size(np.asarray(frames))
    luma = rgb_to_luma(clip)
    source = work_dir / "source.y4m"
    write(source, from_luma(luma, fps=float(fps)))
    metric = PsnrMetric()
    rgb_for_psnr = np.repeat(luma[..., None], 3, axis=-1)
    mask = None
    if masks is not None:
        mask = as_mask(masks, luma.shape[0], luma.shape[1], luma.shape[2])
        mask = mask[:, : luma.shape[1], : luma.shape[2]]
    rates: list[float] = []
    qualities: list[float] = []
    fg_psnr: list[float] = []
    bg_psnr: list[float] = []
    tool: dict[str, str] | None = None
    actual_qps: list[int] = []
    for qp in used_qps:
        used_qp, bitstream, record = encode_qp_with_vvc_fallback(
            source, work_dir, codec_name, qp, notes, label
        )
        actual_qps.append(used_qp)
        request = qp_request(codec_name, used_qp)
        decoded_path = work_dir / f"decoded_qp{used_qp}.y4m"
        if record is None:
            size_bytes = float(bitstream.stat().st_size)
            if tool is None:
                resolved = resolved_tools(codec_name)
                tool = {
                    "encoder_path": resolved["encoder_path"],
                    "encoder_version": resolved["encoder_version"],
                    "ffmpeg_path": resolved["ffmpeg_path"],
                    "ffmpeg_version": resolved["ffmpeg_version"],
                }
            print(f"skip existing {codec_name} qp={used_qp} {bitstream}", flush=True)
        else:
            size_bytes = float(record.size_bytes)
            if tool is None:
                tool = {
                    "encoder_path": record.tool_path,
                    "encoder_version": record.tool_version,
                    "ffmpeg_path": record.ffmpeg_path,
                    "ffmpeg_version": record.ffmpeg_version,
                }
        if not (decoded_path.exists() and decoded_path.stat().st_size > 0):
            decode(bitstream, decoded_path, request)
        decoded = read(decoded_path)
        decoded_rgb = np.repeat(decoded.luma[..., None], 3, axis=-1)
        rates.append(size_bytes)
        qualities.append(float(metric.score(rgb_for_psnr, decoded_rgb)))
        if mask is not None:
            fg_psnr.append(float(masked_psnr(rgb_for_psnr, decoded_rgb, mask)))
            bg_psnr.append(float(masked_psnr(rgb_for_psnr, decoded_rgb, ~mask)))
    if len(rates) >= 2 and not all(rates[i] > rates[i + 1] for i in range(len(rates) - 1)):
        raise RuntimeError(
            f"QP did not move rate as claimed for {label or codec_name}: "
            f"qps={tuple(actual_qps)} rates={tuple(rates)} tool={tool}"
        )
    curve = RDCurve(rates=tuple(rates), qualities=tuple(qualities), label=label)
    return {
        "curve": curve,
        "qps": tuple(actual_qps),
        "fg_psnr": tuple(fg_psnr),
        "bg_psnr": tuple(bg_psnr),
        "tool": tool,
        "notes": notes,
    }
