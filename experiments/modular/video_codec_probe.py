"""QP sweep of VVC against AV1 on the source and on the C1 error video.

The two source points already measured (VVC QP 46, AV1 QP 54) are reused when
the bitstream size still matches. Everything else is encoded. Rows are written
after each point so a stopped run keeps what finished.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from experiments.modular.image_codec_probe import FRAMES, MASKS
from experiments.modular.measured_ladder import (
    _yuv420_to_rgb,
    load_sequence,
    measure_native_anchor,
    measure_rungs,
)
from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode
from src.components.codec.y4m import read
from src.components.metrics.psnr import PsnrMetric
from src.contracts.codecs import EncodeRequest, RateControl

OUT = Path("/home/itec/emanuele/pointstream-data/outputs/video-codec-probe/federer007.json")
WORK = Path("/home/itec/emanuele/pointstream-data/outputs/video-codec-probe/federer007")
SOURCE_Y4M = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/measured-tennis/anchors/short/vvc/input.y4m"
)
IMAGE_REPORT = Path("/home/itec/emanuele/pointstream-data/outputs/image-codec-probe/federer007.json")
N_FRAMES = 48
GRID = (("vvc", (32, 40, 46)), ("av1", (46, 50, 54)))
# Measured tennis short window, 22 Sep 2026. Reused only when the file size matches.
REUSED = {
    ("vvc", 46): {"bytes": 112295, "psnr": 31.264},
    ("av1", 54): {"bytes": 316061, "psnr": 36.214},
}
PRESETS = {"av1": "10", "vvc": "faster"}


def _record_image_decision() -> None:
    report = json.loads(IMAGE_REPORT.read_text())
    report["fullres_plate"] = [
        {"codec": "webp", "quality": 40, "bytes": 171882, "psnr": 37.5671},
        {"codec": "vvc", "quality": 32, "bytes": 125020, "psnr": 39.0869},
        {"codec": "vvc", "quality": 40, "bytes": 61631, "psnr": 34.9648},
        {"codec": "av1", "quality": 42, "bytes": 209799, "psnr": 41.0444},
        {"codec": "av1", "quality": 50, "bytes": 136185, "psnr": 38.7755},
    ]
    report["chosen"] = {
        "plate": "webp",
        "plate_qp": 40,
        "sweep_plate_winner": "vvc",
        "sweep_plate_qp": 32,
        "plate_replacement_deferred": True,
        "crop": "av1",
        "crop_qp": 42,
        "reason": (
            "The full-resolution sweep's VVC intra QP 32 point is the deferred "
            "plate replacement. Production keeps PointStream WebP q40 for now. "
            "Crops use AV1 intra QP 42, the useful-band point in the crop sweep."
        ),
    }
    report["one_codec_for_all_signals"] = False
    IMAGE_REPORT.write_text(json.dumps(report, indent=2) + "\n")


def _score_source(codec: str, qp: int, reference: np.ndarray) -> dict:
    reused = REUSED.get((codec, qp))
    bitstream = SOURCE_Y4M.parent.parent / codec / f"{codec}_qp{qp}{BITSTREAM_SUFFIX[codec]}"
    if reused is not None and bitstream.is_file() and bitstream.stat().st_size == reused["bytes"]:
        return {
            "signal": "source",
            "codec": codec,
            "qp": qp,
            "bytes": reused["bytes"],
            "psnr": reused["psnr"],
            "reused": True,
        }
    work = WORK / "source" / codec / f"qp{qp}"
    work.mkdir(parents=True, exist_ok=True)
    request = EncodeRequest(
        codec_name=codec,
        rate_control=RateControl.QP,
        rate=int(qp),
        preset=PRESETS[codec],
        pix_fmt="yuv420p",
    )
    dest = work / f"{codec}_qp{qp}{BITSTREAM_SUFFIX[codec]}"
    encode(SOURCE_Y4M, dest, request, work_dir=work)
    decoded_path = work / "decoded.y4m"
    decode(dest, decoded_path, request)
    decoded = read(decoded_path)
    if decoded.chroma is None:
        raise RuntimeError(f"{dest} decoded without chroma")
    decoded_rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)
    psnr = float(PsnrMetric().score(reference, decoded_rgb))
    return {
        "signal": "source",
        "codec": codec,
        "qp": qp,
        "bytes": dest.stat().st_size,
        "psnr": round(psnr, 4),
        "reused": False,
    }


def _save(report: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    _record_image_decision()
    report = {"source": str(FRAMES), "n_frames": N_FRAMES, "rows": []}
    if OUT.is_file():
        previous = json.loads(OUT.read_text())
        # A previous run may contain error rows from the FFmpeg/libvvenc
        # empty-file defect. They are diagnostics, not operating points; remove
        # them before the direct-vvenc fallback retries the same cells.
        report["rows"] = [
            row for row in previous.get("rows", []) if "error" not in row
        ]
    done = {(row["signal"], row["codec"], row["qp"]) for row in report["rows"] if "error" not in row}

    source = read(SOURCE_Y4M)
    if source.chroma is None:
        raise RuntimeError(f"{SOURCE_Y4M} has no chroma")
    reference = _yuv420_to_rgb(source.luma, source.chroma)
    for codec, qps in GRID:
        for qp in qps:
            key = ("source", codec, qp)
            if key in done:
                continue
            row = _score_source(codec, qp, reference)
            report["rows"].append(row)
            print(row, flush=True)
            _save(report)

    frames, mask = load_sequence(FRAMES, MASKS, N_FRAMES)
    rungs = measure_rungs(frames, mask, include_residuals=False)
    c1 = rungs[1]
    report["c1"] = {
        "total_bytes": c1.total_bytes,
        "bytes_background": c1.bytes_background,
        "bytes_appearance": c1.bytes_appearance,
        "bytes_metadata": c1.bytes_metadata,
        "psnr_overall": c1.psnr_overall,
        "psnr_fg": c1.psnr_fg,
        "psnr_bg": c1.psnr_bg,
        "psnr_weighted": c1.psnr_weighted,
        "plate_codec": "vvc",
        "plate_qp": 32,
        "crop_codec": "av1",
        "crop_qp": 42,
    }
    _save(report)
    print({"c1": report["c1"]}, flush=True)
    error = np.clip(frames.astype(np.int16) - c1.reconstruction.astype(np.int16) + 128, 0, 255)
    error = np.ascontiguousarray(error.astype(np.uint8))
    for codec, qps in GRID:
        for qp in qps:
            key = ("residual_full", codec, qp)
            if key in done:
                continue
            try:
                anchor = measure_native_anchor(
                    error,
                    codec=codec,
                    qp=qp,
                    work_dir=WORK / "residual_full" / codec / f"qp{qp}",
                )
            except RuntimeError as exc:
                row = {
                    "signal": "residual_full",
                    "codec": codec,
                    "qp": qp,
                    "error": str(exc)[:500],
                }
                report["rows"].append(row)
                print(row, flush=True)
                _save(report)
                continue
            work = WORK / "residual_full" / codec / f"qp{qp}"
            decoded = read(work / f"decoded_qp{qp}.y4m")
            if decoded.chroma is None:
                raise RuntimeError(f"{work} decoded without chroma")
            decoded_rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)
            signed = decoded_rgb.astype(np.int16) - 128
            base = c1.reconstruction.astype(np.int16)
            if signed.shape != base.shape:
                signed = signed[: base.shape[0], : base.shape[1], : base.shape[2]]
            corrected = np.clip(base + signed, 0, 255).astype(np.uint8)
            psnr_corrected = float(PsnrMetric().score(frames, corrected))
            row = {
                "signal": "residual_full",
                "codec": codec,
                "qp": qp,
                "bytes": anchor.total_bytes,
                "psnr": round(anchor.psnr_overall, 4),
                "psnr_corrected": round(psnr_corrected, 4),
                "encoder_path": anchor.encoder_path,
                "encoder_version": anchor.encoder_version,
                "ffmpeg_path": anchor.ffmpeg_path,
                "ffmpeg_version": anchor.ffmpeg_version,
            }
            report["rows"].append(row)
            print(row, flush=True)
            _save(report)


if __name__ == "__main__":
    main()
