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
    PAD_FILL,
    build_plate,
    prepare_canonical_context,
)
from src.components.background.sidecar import build_sidecar
from src.components.background.stream import encode_chain, ffmpeg_provenance, stream_linear
from src.components.codec.frames import rgb_to_luma
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp44-canonical-canvas"
BOUNDS_PATH = OUT_DIR / "bounds-before-run.json"
PANNING_BOUNDS_PATH = OUT_DIR / "bounds-before-panning-alarm.json"
RESULT_PATH = OUT_DIR / "canonical-canvas.json"
PANNING_RESULT_PATH = OUT_DIR / "panning-quality.json"

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
            "late-frame quality drop check on warp-back vs source, mean of both "
            "scenes. A pan's last five frames drop ~7 dB against a 24-frame "
            "plate (measured uncoded 2026-09-02); the static scene does not. "
            "That mean therefore sits near -4 dB. It is not a canvas walk-off: "
            "independent and canonical agree, coverage is 1.0, pad is ~0. "
            "Keep reporting it; catch a canvas cause with the independent split."
        ),
        "on_breach": "expected on scene_010; check canonical_minus_independent_last_first_dB before calling it a canvas bug",
    },
}

#: Written before the panning-quality probe. Two-sided. These are not the
#: original last-minus-first band: that one fired, and this probe exists to
#: say whether the canvas is why.
PANNING_BOUNDS: dict[str, Any] = {
    "canonical_minus_independent_last_first_dB": {
        "low": -1.0,
        "high": 1.0,
        "basis": (
            "If the origin shift walked the pan off the canvas, canonical "
            "last-minus-first would drop more than independent. Agreement "
            "within 1 dB means both paths see the same plate-vs-source drop."
        ),
        "on_breach": "the canvas is the cause; inspect origin shift and last-frame coverage",
    },
    "last_frame_coverage": {
        "best": 0.90,
        "worst": 1.0,
        "basis": (
            "BP31 canvas growth at span 24 is about x1.016. If the homographies "
            "are right, the last frame still lands on the plate. Below 0.90 is "
            "a walk-off."
        ),
        "on_breach": "last frame maps off the plate; that is the original bound's feared cause",
    },
    "last_frame_pad_fill_fraction": {
        "best": 0.0,
        "worst": 0.05,
        "basis": (
            "Fraction of last-frame pixels that sample PAD_FILL on the plate. "
            "The original alarm attributed the drop to walking onto pad. Above "
            "5% would drop PSNR the way that bound feared."
        ),
        "on_breach": "late frames are reconstructing from mid-grey pad, not from court texture",
    },
    "independent_first_psnr_dB": {
        "best": 25.0,
        "worst": 45.0,
        "basis": (
            "A registered median plate vs its first source frame is a resampling "
            "floor, typically the low-to-mid 30s on 4K broadcast. Below 25 is a "
            "broken stitch. Above 45 is near-identity, which a 24-frame median is not."
        ),
        "on_breach": "the first-frame floor itself is wrong; do not interpret the late-frame drop",
    },
    "last_homography_scale": {
        "best": 0.90,
        "worst": 1.10,
        "basis": (
            "A pan is a translation. The spurious per-frame zoom BP29 caught "
            "would compound at 24 frames. Outside 0.90-1.10 the motion model "
            "is not a pan."
        ),
        "on_breach": "revisit RANSAC threshold; a zoom-shaped H will smear late frames",
    },
    "registered_minus_unregistered_last_psnr_dB": {
        "best": 1.0,
        "worst": 40.0,
        "basis": (
            "On a pan, an unregistered median smears. Registration should beat "
            "it on the last frame by at least 1 dB if the homography is doing "
            "work. A negative gap means registration is hurting."
        ),
        "on_breach": "registration is not the product the plate claims to be",
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


def _write_panning_bounds() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_before_first_measurement": True,
        "mode": "uncoded plate warp-back vs source (no AV1)",
        "video": VIDEO,
        "scene": "scene_010",
        "nulls": ["independent plate", "register=False", "span=1 (first frame as plate)"],
        "bounds": PANNING_BOUNDS,
        "paths": ps_paths.describe(),
    }
    PANNING_BOUNDS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def _packed_matrix(row: tuple[float, ...]) -> np.ndarray:
    return np.asarray(row, dtype=np.float64).reshape(3, 3)


def _coverage_and_pad(
    plate: np.ndarray,
    homography: np.ndarray,
    frame_height: int,
    frame_width: int,
) -> dict[str, float]:
    """How much of a source frame lands on the plate, and how much on PAD_FILL."""
    plate_h, plate_w = int(plate.shape[0]), int(plate.shape[1])
    ys, xs = np.mgrid[0:frame_height, 0:frame_width]
    pts = np.stack(
        [
            xs.ravel().astype(np.float64),
            ys.ravel().astype(np.float64),
            np.ones(frame_height * frame_width, dtype=np.float64),
        ],
        axis=0,
    )
    mapped = homography @ pts
    weight = mapped[2]
    ok = np.abs(weight) > 1e-8
    x = mapped[0] / np.where(ok, weight, 1.0)
    y = mapped[1] / np.where(ok, weight, 1.0)
    inside = ok & (x >= 0.0) & (x < float(plate_w)) & (y >= 0.0) & (y < float(plate_h))
    xi = np.clip(np.round(x).astype(np.int64), 0, plate_w - 1)
    yi = np.clip(np.round(y).astype(np.int64), 0, plate_h - 1)
    sampled = plate[yi, xi]
    is_pad = np.all(sampled == PAD_FILL, axis=-1) & inside
    coverage = float(inside.mean())
    pad_of_frame = float(is_pad.mean())
    pad_of_inside = float(is_pad[inside].mean()) if bool(inside.any()) else 0.0
    return {
        "coverage": coverage,
        "pad_fill_fraction": pad_of_frame,
        "pad_fill_of_on_canvas": pad_of_inside,
    }


def _homography_scale(matrix: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))


def _centre_shift(
    matrix: np.ndarray, frame_height: int, frame_width: int
) -> list[float]:
    centre = np.array([frame_width / 2.0, frame_height / 2.0, 1.0], dtype=np.float64)
    mapped = matrix @ centre
    mapped = mapped[:2] / mapped[2]
    return [float(mapped[0] - centre[0]), float(mapped[1] - centre[1])]


def _per_frame_psnr(reference: np.ndarray, candidate: np.ndarray) -> list[float]:
    return [
        round(_psnr(reference[index], candidate[index]), 3)
        for index in range(int(reference.shape[0]))
    ]


def _probe_panning(n_frames: int) -> dict[str, Any]:
    clip = load_tier_clip(video=VIDEO, scene="scene_010", n_frames=n_frames)
    frames = np.asarray(clip.frames)
    mask = np.asarray(clip.union_mask, dtype=np.uint8) * 255
    height, width = int(frames.shape[1]), int(frames.shape[2])

    independent, independent_maps = build_plate(frames, masks=mask)
    from_independent = _warp_back(independent, independent_maps, frames)

    canvas, alignments, _bounds = prepare_canonical_context(
        [frames], context_id=f"{VIDEO}-point"
    )
    canonical, canonical_maps = build_plate(
        frames, masks=mask, canvas=canvas, alignment=alignments[0]
    )
    from_canonical = _warp_back(canonical, canonical_maps, frames)

    unregistered, unregistered_maps = build_plate(frames, masks=mask, register=False)
    from_unregistered = _warp_back(unregistered, unregistered_maps, frames)

    first_only, _ = build_plate(frames[:1], masks=mask[:1])
    span1_last = _psnr(frames[-1], first_only)

    last_h = _packed_matrix(independent_maps[-1])
    cover = _coverage_and_pad(independent, last_h, height, width)
    canon_cover = _coverage_and_pad(
        canonical, _packed_matrix(canonical_maps[-1]), height, width
    )

    independent_curve = _per_frame_psnr(frames, from_independent)
    canonical_curve = _per_frame_psnr(frames, from_canonical)
    independent_delta = independent_curve[-1] - independent_curve[0]
    canonical_delta = canonical_curve[-1] - canonical_curve[0]
    registered_last = independent_curve[-1]
    unregistered_last = _psnr(frames[-1], from_unregistered[-1])

    checks = [
        _check(
            "canonical_minus_independent_last_first_dB",
            canonical_delta - independent_delta,
            PANNING_BOUNDS["canonical_minus_independent_last_first_dB"]["low"],
            PANNING_BOUNDS["canonical_minus_independent_last_first_dB"]["high"],
        ),
        _check(
            "last_frame_coverage",
            cover["coverage"],
            PANNING_BOUNDS["last_frame_coverage"]["best"],
            PANNING_BOUNDS["last_frame_coverage"]["worst"],
        ),
        _check(
            "last_frame_pad_fill_fraction",
            cover["pad_fill_fraction"],
            PANNING_BOUNDS["last_frame_pad_fill_fraction"]["best"],
            PANNING_BOUNDS["last_frame_pad_fill_fraction"]["worst"],
        ),
        _check(
            "independent_first_psnr_dB",
            independent_curve[0],
            PANNING_BOUNDS["independent_first_psnr_dB"]["best"],
            PANNING_BOUNDS["independent_first_psnr_dB"]["worst"],
        ),
        _check(
            "last_homography_scale",
            _homography_scale(last_h),
            PANNING_BOUNDS["last_homography_scale"]["best"],
            PANNING_BOUNDS["last_homography_scale"]["worst"],
        ),
        _check(
            "registered_minus_unregistered_last_psnr_dB",
            registered_last - unregistered_last,
            PANNING_BOUNDS["registered_minus_unregistered_last_psnr_dB"]["best"],
            PANNING_BOUNDS["registered_minus_unregistered_last_psnr_dB"]["worst"],
        ),
    ]
    return {
        "n_frames": n_frames,
        "independent_shape": list(independent.shape),
        "canonical_shape": list(canonical.shape),
        "independent_psnr_y_by_frame": independent_curve,
        "canonical_psnr_y_by_frame": canonical_curve,
        "independent_last_minus_first_dB": round(independent_delta, 3),
        "canonical_last_minus_first_dB": round(canonical_delta, 3),
        "last_frame_coverage": cover,
        "canonical_last_frame_coverage": canon_cover,
        "last_homography_scale": _homography_scale(last_h),
        "last_centre_shift_xy": _centre_shift(last_h, height, width),
        "unregistered_last_psnr_dB": round(unregistered_last, 3),
        "span1_last_vs_first_psnr_dB": round(span1_last, 3),
        "checks": checks,
        "alarms": [item["check"] for item in checks if not item["inside_band"]],
    }


def _run_panning_probe(durations: list[int]) -> int:
    _write_panning_bounds()
    print(f"bounds written to {PANNING_BOUNDS_PATH} before any measurement")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for n_frames in durations:
        print(f"panning probe {n_frames} ...")
        try:
            row = _probe_panning(int(n_frames))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{n_frames}: {type(exc).__name__}: {exc}")
            print(f"  FAILED {n_frames}: {exc}")
            continue
        rows.append(row)
        print(
            f"  indep Δ {row['independent_last_minus_first_dB']} dB  "
            f"canon Δ {row['canonical_last_minus_first_dB']} dB  "
            f"coverage {row['last_frame_coverage']['coverage']:.3f}  "
            f"alarms={row['alarms']}"
        )
    record = {
        "video": VIDEO,
        "scene": "scene_010",
        "mode": "uncoded plate warp-back, no AV1",
        "bounds_path": str(PANNING_BOUNDS_PATH),
        "durations": rows,
        "failures": failures,
        "submitted": len(durations),
        "succeeded": len(rows),
        "failed": len(failures),
        "paths": ps_paths.describe(),
    }
    PANNING_RESULT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {PANNING_RESULT_PATH}")
    return 1 if failures else 0


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
    independent_first: list[float] = []
    independent_last: list[float] = []
    for frames, local_plate, local_maps, canon_plate, canon_maps in zip(
        scenes, independent_plates, independent_maps, canonical_plates, canonical_maps, strict=True
    ):
        from_independent = _warp_back(local_plate, local_maps, frames)
        from_canonical = _warp_back(canon_plate, canon_maps, frames)
        reconstruction_errors.append(_mae(from_independent, from_canonical))
        independent_first.append(_psnr(frames[0], from_independent[0]))
        independent_last.append(_psnr(frames[-1], from_independent[-1]))
        first_psnr.append(_psnr(frames[0], from_canonical[0]))
        last_psnr.append(_psnr(frames[-1], from_canonical[-1]))

    largest_local = max(item.local_area for item in bounds)
    area_ratio = canvas.area / float(largest_local)
    byte_ratio = predictive_bytes / float(independent_bytes) if independent_bytes else None
    mean_recon = float(sum(reconstruction_errors) / len(reconstruction_errors))
    drift = float(sum(last_psnr) / len(last_psnr) - sum(first_psnr) / len(first_psnr))
    independent_drift = float(
        sum(independent_last) / len(independent_last)
        - sum(independent_first) / len(independent_first)
    )

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
        _check(
            "canonical_minus_independent_last_first_dB",
            drift - independent_drift,
            PANNING_BOUNDS["canonical_minus_independent_last_first_dB"]["low"],
            PANNING_BOUNDS["canonical_minus_independent_last_first_dB"]["high"],
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
        "independent_first_frame_psnr_dB": independent_first,
        "independent_last_frame_psnr_dB": independent_last,
        "encode_seconds": round(encode_seconds, 2),
        "checks": checks,
        "alarms": [c["check"] for c in checks if not c["inside_band"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", nargs="*", type=int, default=list(DURATIONS))
    parser.add_argument(
        "--probe-panning",
        action="store_true",
        help="uncoded plate warp-back vs source on scene_010; no AV1 stream",
    )
    args = parser.parse_args(argv)

    if args.probe_panning:
        durations = args.durations if args.durations != list(DURATIONS) else [24]
        return _run_panning_probe(durations)

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
