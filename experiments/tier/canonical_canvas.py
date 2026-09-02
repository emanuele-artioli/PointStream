"""Canonical canvas for long compatible scenes — BP44 diagnostic.

Uses the two BP31 scenes (alcaraz_highlights ``scene_000`` static,
``scene_010`` panning) at 24, 32 and 48 frames. Bounds are written before
the first encode.

This mode is **offline / buffered**: the union canvas sees every scene in
the context before any background is coded. It cannot support a live title.

Run: ``python -m experiments.tier.canonical_canvas``
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import cv2
import numpy as np

from experiments.tier.clip import load_tier_clip
from src.components.background.plate import (
    build_plate,
    prepare_canonical_context,
)
from src.components.background.sidecar import build_sidecar
from src.components.background.stream import encode_chain, ffmpeg_provenance, stream_linear
from src.components.codec.frames import rgb_to_luma
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp44-canonical-canvas"
BOUNDS_PATH = OUT_DIR / "bounds-before-run.json"
RESULT_PATH = OUT_DIR / "canonical-canvas.json"

VIDEO = "alcaraz_highlights"
SCENES = ("scene_000", "scene_010")
DURATIONS = (24, 32, 48)

#: Written before the first encode. Two-sided: a result below the floor is as
#: much an alarm as one above the ceiling.
BOUNDS: dict[str, Any] = {
    "reconstruction_mae_vs_independent": {
        "best": 0.0,
        "worst": 3.0,
        "basis": (
            "independent and canonical paths warp the same source through "
            "homographies that differ by a translation. Residual error is "
            "resampling, not a different picture. Above 3 MAE is a geometry bug."
        ),
        "on_breach": "inspect origin shift and homography composition before citing size",
    },
    "canvas_area_over_largest_local": {
        "best": 1.0,
        "worst": 1.25,
        "basis": (
            "BP31 §12 at span 48: panning 2190x3932 vs static 2161x3841 is "
            "x1.038 in area. Same-camera union should sit near the larger local "
            "canvas. Below 1.0 is impossible. Above 1.25 means alignment placed "
            "the scenes far apart or failed open."
        ),
        "on_breach": "check estimate_alignment; an unaligned pad-to-max should still be near x1.04",
    },
    "predictive_bytes_over_independent": {
        "best": 0.20,
        "worst": 1.10,
        "basis": (
            "BP30 sequence ratio was 0.30-0.90 of all-intra on equal-size plates. "
            "Padding can eat some of the gain, so the ceiling sits slightly above "
            "1. Below 0.20 is better than any measured stream. Above 1.10 means "
            "padding plus prediction lost to independent local plates."
        ),
        "on_breach": "separate padding bytes from prediction; do not cite a rate until that split is clear",
    },
    "last_minus_first_psnr_dB": {
        "low": -3.0,
        "high": 1.0,
        "basis": (
            "late-frame quality drop check on warp-back vs source. A drop worse "
            "than 3 dB suggests the homography walked off the canvas. A rise "
            "above 1 dB is not the failure this axis is for, but it is unexpected."
        ),
        "on_breach": "plot per-frame PSNR before calling the canvas a reconstruction win",
    },
}


def _write_bounds() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_before_first_encode": True,
        "mode": "offline/buffered canonical canvas",
        "video": VIDEO,
        "scenes": list(SCENES),
        "durations": list(DURATIONS),
        "bounds": BOUNDS,
        "paths": ps_paths.describe(),
    }
    BOUNDS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = rgb_to_luma(np.asarray(reference)[None, ...])[0].astype(np.float64)
    cand = rgb_to_luma(np.asarray(candidate)[None, ...])[0].astype(np.float64)
    mse = float(np.mean((ref - cand) ** 2))
    if mse <= 0.0:
        return math.inf
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def _warp_back(
    plate: np.ndarray,
    packed: tuple[tuple[float, ...], ...],
    frames: np.ndarray,
) -> np.ndarray:
    height, width = int(frames.shape[1]), int(frames.shape[2])
    out = np.empty_like(frames)
    for index, row in enumerate(packed):
        matrix = np.asarray(row, dtype=np.float64).reshape(3, 3)
        out[index] = cv2.warpPerspective(
            plate,
            np.linalg.inv(matrix),
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
    return out


def _check(name: str, value: float, low: float, high: float) -> dict[str, Any]:
    return {
        "check": name,
        "value": value,
        "band": [low, high],
        "inside_band": low <= value <= high,
    }


def _run_duration(n_frames: int) -> dict[str, Any]:
    clips = [
        load_tier_clip(video=VIDEO, scene=scene, n_frames=n_frames) for scene in SCENES
    ]
    scenes = [np.asarray(clip.frames) for clip in clips]
    masks = [
        np.asarray(clip.union_mask, dtype=np.uint8) * 255 for clip in clips
    ]

    started = time.time()
    independent_plates: list[np.ndarray] = []
    independent_maps: list[tuple[tuple[float, ...], ...]] = []
    independent_bytes = 0
    for frames, mask in zip(scenes, masks, strict=True):
        plate, packed = build_plate(frames, masks=mask)
        independent_plates.append(plate)
        independent_maps.append(packed)
        encoded = encode_chain([plate], codec="av1", crf=38)
        independent_bytes += int(encoded.marginal_bytes)

    canvas, alignments, bounds = prepare_canonical_context(
        scenes, context_id=f"{VIDEO}-point"
    )
    canonical_plates: list[np.ndarray] = []
    canonical_maps: list[tuple[tuple[float, ...], ...]] = []
    for frames, mask, alignment in zip(scenes, masks, alignments, strict=True):
        plate, packed = build_plate(frames, masks=mask, canvas=canvas, alignment=alignment)
        canonical_plates.append(plate)
        canonical_maps.append(packed)

    payloads = stream_linear(canonical_plates, codec="av1", crf=38)
    predictive_bytes = sum(p.byte_count for p in payloads)
    encode_seconds = time.time() - started

    reconstruction_errors = []
    first_psnr: list[float] = []
    last_psnr: list[float] = []
    for frames, local_plate, local_maps, canon_plate, canon_maps in zip(
        scenes, independent_plates, independent_maps, canonical_plates, canonical_maps, strict=True
    ):
        from_independent = _warp_back(local_plate, local_maps, frames)
        from_canonical = _warp_back(canon_plate, canon_maps, frames)
        reconstruction_errors.append(_mae(from_independent, from_canonical))
        first_psnr.append(_psnr(frames[0], from_canonical[0]))
        last_psnr.append(_psnr(frames[-1], from_canonical[-1]))
        # Independent last-minus-first is the control for the canvas-walk-off
        # alarm: if both paths drop, the plate vs source is the cause.

    largest_local = max(item.local_area for item in bounds)
    area_ratio = canvas.area / float(largest_local)
    byte_ratio = predictive_bytes / float(independent_bytes) if independent_bytes else None
    mean_recon = float(sum(reconstruction_errors) / len(reconstruction_errors))
    drift = float(sum(last_psnr) / len(last_psnr) - sum(first_psnr) / len(first_psnr))

    sidecar = build_sidecar("png")
    padding_png_bytes = []
    for local, placed in zip(independent_plates, canonical_plates, strict=True):
        padding_png_bytes.append(len(sidecar.encode(placed)) - len(sidecar.encode(local)))

    checks = [
        _check(
            "reconstruction_mae_vs_independent",
            mean_recon,
            BOUNDS["reconstruction_mae_vs_independent"]["best"],
            BOUNDS["reconstruction_mae_vs_independent"]["worst"],
        ),
        _check(
            "canvas_area_over_largest_local",
            area_ratio,
            BOUNDS["canvas_area_over_largest_local"]["best"],
            BOUNDS["canvas_area_over_largest_local"]["worst"],
        ),
        _check(
            "predictive_bytes_over_independent",
            float(byte_ratio) if byte_ratio is not None else math.nan,
            BOUNDS["predictive_bytes_over_independent"]["best"],
            BOUNDS["predictive_bytes_over_independent"]["worst"],
        ),
        _check(
            "last_minus_first_psnr_dB",
            drift,
            BOUNDS["last_minus_first_psnr_dB"]["low"],
            BOUNDS["last_minus_first_psnr_dB"]["high"],
        ),
    ]
    return {
        "n_frames": n_frames,
        "local_shapes": [list(p.shape) for p in independent_plates],
        "canonical_shape": [canvas.height, canvas.width, 3],
        "canvas": {
            "context_id": canvas.context_id,
            "origin_xy": list(canvas.origin_xy),
            "width": canvas.width,
            "height": canvas.height,
            "area": canvas.area,
            "aligned": canvas.aligned,
        },
        "largest_local_area": largest_local,
        "independent_bytes": independent_bytes,
        "predictive_bytes": predictive_bytes,
        "padding_png_byte_delta": padding_png_bytes,
        "padding_note": (
            "PNG byte delta is a diagnostic for padding itself. The encoded "
            "AV1 bitstream size is the rate."
        ),
        "reconstruction_mae_vs_independent": reconstruction_errors,
        "first_frame_psnr_dB": first_psnr,
        "last_frame_psnr_dB": last_psnr,
        "encode_seconds": round(encode_seconds, 2),
        "checks": checks,
        "alarms": [c["check"] for c in checks if not c["inside_band"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", nargs="*", type=int, default=list(DURATIONS))
    args = parser.parse_args(argv)

    _write_bounds()
    print(f"bounds written to {BOUNDS_PATH} before any encode")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for n_frames in args.durations:
        print(f"duration {n_frames} ...")
        try:
            row = _run_duration(int(n_frames))
        except Exception as exc:  # noqa: BLE001 — report every duration, do not abandon the rest
            failures.append(f"{n_frames}: {type(exc).__name__}: {exc}")
            print(f"  FAILED {n_frames}: {exc}")
            continue
        rows.append(row)
        record_so_far = {
            "video": VIDEO,
            "scenes": list(SCENES),
            "mode": "offline/buffered",
            "durations": rows,
            "failures": failures,
            "submitted": len(args.durations),
            "succeeded": len(rows),
            "failed": len(failures),
        }
        RESULT_PATH.write_text(json.dumps(record_so_far, indent=2) + "\n", encoding="utf-8")
        print(
            f"  canvas {row['canvas']['width']}x{row['canvas']['height']} "
            f"aligned={row['canvas']['aligned']} "
            f"independent={row['independent_bytes']} B "
            f"predictive={row['predictive_bytes']} B "
            f"alarms={row['alarms']}"
        )

    record = {
        "video": VIDEO,
        "scenes": list(SCENES),
        "mode": "offline/buffered",
        "ffmpeg": ffmpeg_provenance(),
        "paths": ps_paths.describe(),
        "bounds_path": str(BOUNDS_PATH),
        "durations": rows,
        "failures": failures,
        "submitted": len(args.durations),
        "succeeded": len(rows),
        "failed": len(failures),
    }
    RESULT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULT_PATH}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
