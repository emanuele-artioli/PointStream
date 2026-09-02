"""AV1/VVC usable-range probe at each encoder's slowest valid preset.

Resolves binaries by path and version, walks each codec's legal QP range at
fixed source resolution and frame rate, and rejects empty, undecodable,
wrong-size, wrong-frame-count or non-monotone points. The smallest valid
bitstream is kept even when it sits below a later BD-rate overlap.

Does not downscale or drop frames. Generation is not involved. Torch is not
imported.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.clip import load_tier_clip
from experiments.tier.low_rate_validate import (
    BOUNDS_PATH,
    DECLARED_FPS,
    HEADLINE_METRICS,
    PRESET_ORDER,
    PRIMARY_ANCHORS,
    PROBE_PATH,
    decode_rejections,
    monotonicity_alarms,
    probe_qps,
)
from src.components.codec.encode import QP_BOUNDS
from src.components.codec.frames import even_size, rgb_to_luma
from src.components.codec.measure import coded_roundtrip
from src.components.codec.tools import resolve_encoder, resolve_ffmpeg
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.metrics import metric as metric_spec


def _y_psnr(reference: np.ndarray, predicted: np.ndarray) -> float:
    ref = rgb_to_luma(np.asarray(reference)).astype(np.float64)
    got = rgb_to_luma(np.asarray(predicted)).astype(np.float64)
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


def _score_headlines(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float | str]:
    """VMAF, Y-PSNR, SSIM. Built per call so a missing VMAF binary is one point."""
    from src.components.metrics.ssim import SsimMetric
    from src.components.metrics.vmaf import VmafMetric

    scores: dict[str, float | str] = {
        "psnr_y": _y_psnr(reference, predicted),
        "ssim": float(SsimMetric().score(reference, predicted)),
    }
    try:
        scores["vmaf"] = float(VmafMetric().score(reference, predicted))
    except (RuntimeError, FileNotFoundError) as exc:
        scores["vmaf_error"] = str(exc)
    return scores


def select_slowest_valid_preset(codec: str, *, fps: float) -> tuple[str, list[str]]:
    """Try documented presets slowest-first on a tiny clip until one encodes.

    The documented slowest name is not always installed (libvvenc has no
    `placebo`). A 4K probe that starts at a rejected preset would record every
    QP as failed and look like a codec-floor result.
    """
    tiny = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    tiny[:, 16:48, 16:48] = 180
    rejected: list[str] = []
    last_error = ""
    for preset in PRESET_ORDER[codec]:
        request = EncodeRequest(
            codec_name=codec,
            rate_control=RateControl.QP,
            rate=int(QP_BOUNDS[codec][1]),
            preset=str(preset),
            pix_fmt="yuv420p",
        )
        request.validate()
        try:
            coded_roundtrip(tiny, request=request, fps=fps)
            return str(preset), rejected
        except Exception as exc:  # noqa: BLE001 — probing which name the binary takes
            rejected.append(str(preset))
            last_error = repr(exc)
            continue
    raise RuntimeError(
        f"no documented preset for {codec!r} encoded a 64x64 probe. "
        f"Rejected {rejected}. Last error: {last_error}"
    )


def _tool_record(codec: str, *, fps: float) -> dict[str, Any]:
    encoder = resolve_encoder(codec)
    ffmpeg = resolve_ffmpeg()
    selected, rejected = select_slowest_valid_preset(codec, fps=fps)
    return {
        "codec": codec,
        "encoder_path": encoder.path,
        "encoder_version": encoder.version,
        "encoder_features": sorted(encoder.features),
        "ffmpeg_path": ffmpeg.path,
        "ffmpeg_version": ffmpeg.version,
        "qp_bounds": list(QP_BOUNDS[codec]),
        "documented_presets_slowest_first": list(PRESET_ORDER[codec]),
        "rejected_presets": rejected,
        "selected_preset": selected,
    }


def probe_codec(
    frames: np.ndarray,
    *,
    codec: str,
    preset: str,
    qps: tuple[int, ...],
    fps: float,
) -> dict[str, Any]:
    """One codec's usable range on ``frames``. Failed QPs stay in the record."""
    source = even_size(np.asarray(frames, dtype=np.uint8))
    points: list[dict[str, Any]] = []
    usable: list[dict[str, Any]] = []

    for qp in qps:
        request = EncodeRequest(
            codec_name=codec,
            rate_control=RateControl.QP,
            rate=int(qp),
            preset=preset,
            pix_fmt="yuv420p",
        )
        request.validate()
        started = time.time()
        point: dict[str, Any] = {
            "qp": int(qp),
            "preset": preset,
            "fps": fps,
        }
        try:
            size, decoded = coded_roundtrip(source, request=request, fps=fps)
            encode_s = time.time() - started
            reasons = decode_rejections(
                bitstream_bytes=int(size),
                source_shape=(
                    int(source.shape[0]),
                    int(source.shape[1]),
                    int(source.shape[2]),
                    int(source.shape[3]),
                ),
                decoded_shape=tuple(int(dim) for dim in decoded.shape),
            )
            point.update(
                {
                    "bytes": int(size),
                    "encode_plus_decode_seconds": round(encode_s, 3),
                    "decoded_shape": list(decoded.shape),
                    "rejections": reasons,
                }
            )
            if reasons:
                point["usable"] = False
            else:
                scores = _score_headlines(source, decoded)
                point["scores"] = scores
                point["usable"] = isinstance(scores.get("vmaf"), float)
                if point["usable"]:
                    usable.append(point)
        except Exception as exc:  # noqa: BLE001 — recorded per QP, not fatal
            point.update(
                {
                    "usable": False,
                    "rejections": [repr(exc)],
                    "encode_plus_decode_seconds": round(time.time() - started, 3),
                }
            )
        points.append(point)
        status = "ok" if point.get("usable") else "rejected"
        print(
            f"  {codec} qp{qp} {status} "
            f"{point.get('bytes', '—')} B  {point.get('encode_plus_decode_seconds')} s"
            f"{'' if not point.get('rejections') else '  ' + '; '.join(point['rejections'])}",
            flush=True,
        )

    alarms: list[str] = []
    if len(usable) < 2:
        alarms.append(f"{codec}: fewer than two usable points ({len(usable)})")
    else:
        vmaf_spec = metric_spec("vmaf")
        alarms.extend(
            monotonicity_alarms(
                [int(item["qp"]) for item in usable],
                [float(item["bytes"]) for item in usable],
                [float(item["scores"]["vmaf"]) for item in usable],
                higher_is_better=vmaf_spec.higher_is_better,
            )
        )
        alarms.extend(
            monotonicity_alarms(
                [int(item["qp"]) for item in usable],
                [float(item["bytes"]) for item in usable],
                [float(item["scores"]["psnr_y"]) for item in usable],
                higher_is_better=True,
            )
        )

    floor = None
    if usable:
        smallest = min(usable, key=lambda item: int(item["bytes"]))
        floor = {
            "qp": smallest["qp"],
            "bytes": smallest["bytes"],
            "scores": smallest["scores"],
        }

    return {
        "codec": codec,
        "preset": preset,
        "n_attempted": len(points),
        "n_usable": len(usable),
        "points": points,
        "smallest_valid": floor,
        "alarms": alarms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="alcaraz_highlights")
    parser.add_argument("--scene", default="scene_000")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--codecs", nargs="+", default=list(PRIMARY_ANCHORS))
    parser.add_argument(
        "--preset",
        default=None,
        help="override the slowest-preset rule. Primary comparison must not.",
    )
    parser.add_argument("--fps", type=float, default=DECLARED_FPS)
    parser.add_argument("--out", default=str(PROBE_PATH))
    args = parser.parse_args(argv)

    if not BOUNDS_PATH.is_file():
        raise SystemExit(
            f"{BOUNDS_PATH} does not exist. Write bounds before the first encode "
            "(python -m experiments.tier.low_rate_bounds)."
        )

    clip = load_tier_clip(video=args.video, scene=args.scene, n_frames=args.frames)
    frames = even_size(np.asarray(clip.frames, dtype=np.uint8))
    print(
        f"{args.video}/{args.scene}: {frames.shape[0]} frames "
        f"{frames.shape[2]}x{frames.shape[1]} @ {args.fps} fps",
        flush=True,
    )

    tools = {name: _tool_record(name, fps=float(args.fps)) for name in args.codecs}
    curves: dict[str, Any] = {}
    for name in args.codecs:
        preset = args.preset or tools[name]["selected_preset"]
        if args.preset:
            print(f"NOTE: preset override {preset!r} for {name}", flush=True)
        print(f"{name}: preset {preset}  {tools[name]['encoder_path']}", flush=True)
        curves[name] = probe_codec(
            frames,
            codec=name,
            preset=str(preset),
            qps=probe_qps(name),
            fps=float(args.fps),
        )

    report = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bounds_file": str(BOUNDS_PATH),
        "clip": clip.describe(),
        "fps": args.fps,
        "resolution_policy": "native; no downscale, no frame drop",
        "headline_metrics": list(HEADLINE_METRICS),
        "primary_quality": "vmaf",
        "preset_policy": (
            "slowest valid preset or full reference configuration"
            if args.preset is None
            else f"OVERRIDDEN to {args.preset}"
        ),
        "tools": tools,
        "curves": curves,
        "alarms": [alarm for curve in curves.values() for alarm in curve["alarms"]],
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    if report["alarms"]:
        print("=== ALARMS ===", flush=True)
        for alarm in report["alarms"]:
            print(f"  ! {alarm}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
