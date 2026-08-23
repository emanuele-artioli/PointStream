"""Encode a clip at several QPs and score PSNR on the whole frame and on regions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode
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
) -> dict[str, object]:
    """Encode RGB ``frames`` at ``qps``. Quality is PSNR against this clip's luma."""
    if not encoders_available(codec_name):
        raise FileNotFoundError(f"encoder for {codec_name} is not on this host")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    clip = even_size(np.asarray(frames))
    luma = rgb_to_luma(clip)
    source = work_dir / "source.y4m"
    write(source, from_luma(luma, fps=25.0))
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
    for qp in qps:
        request = qp_request(codec_name, qp)
        bitstream = work_dir / f"{codec_name}_qp{qp}{BITSTREAM_SUFFIX[codec_name]}"
        record = encode(source, bitstream, request)
        if tool is None:
            tool = {
                "encoder_path": record.tool_path,
                "encoder_version": record.tool_version,
                "ffmpeg_path": record.ffmpeg_path,
                "ffmpeg_version": record.ffmpeg_version,
            }
        decoded_path = work_dir / f"decoded_qp{qp}.y4m"
        decode(bitstream, decoded_path, request)
        decoded = read(decoded_path)
        decoded_rgb = np.repeat(decoded.luma[..., None], 3, axis=-1)
        rates.append(float(record.size_bytes))
        qualities.append(float(metric.score(rgb_for_psnr, decoded_rgb)))
        if mask is not None:
            fg_psnr.append(float(masked_psnr(rgb_for_psnr, decoded_rgb, mask)))
            bg_psnr.append(float(masked_psnr(rgb_for_psnr, decoded_rgb, ~mask)))
    if len(rates) >= 2 and not all(rates[i] > rates[i + 1] for i in range(len(rates) - 1)):
        raise RuntimeError(
            f"QP did not move rate as claimed for {label or codec_name}: "
            f"qps={qps} rates={tuple(rates)} tool={tool}"
        )
    curve = RDCurve(rates=tuple(rates), qualities=tuple(qualities), label=label)
    return {
        "curve": curve,
        "qps": qps,
        "fg_psnr": tuple(fg_psnr),
        "bg_psnr": tuple(bg_psnr),
        "tool": tool,
    }
