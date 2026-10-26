"""CPU mechanism pilot: binary/key edges versus small ramps, with no oracle decoder."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import resource
import tarfile
import tempfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_foreground_video import PILOTS
from experiments.tennis.k6b_matte_cpu import box_maps
from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from experiments.tennis.k6b_reference_affine import unlift
from experiments.tennis.k6b_reference_video import (
    rgb_yuv,
    code_yuv,
    back_rgb,
    container,
    command,
    ffmpeg_args,
)
from src.codecs import svtav1

WIDTHS = (0, 2, 4)
FAMILIES = ("rgb-key", "luma-mask-only")


def decode_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Both outputs depend only on decoded channels and fixed paid constants.
    return unlift(rgb), rgb.max(axis=-1) > 16


def decode_luma(y: np.ndarray) -> np.ndarray:
    return y >= 128


def alpha_ramp(mask: np.ndarray, width: int) -> np.ndarray:
    import cv2

    if width not in WIDTHS:
        raise ValueError("Unfrozen ramp width")
    if not width:
        return mask.astype(np.float32)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    signed = np.where(mask, inside - 0.5, -(outside - 0.5))
    return np.clip(0.5 + signed / width, 0, 1)


def edge(mask: np.ndarray) -> np.ndarray:
    import cv2

    return mask ^ cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def diagnostics(
    truth: np.ndarray, recovered: np.ndarray, rgb: np.ndarray | None, expected_rgb: np.ndarray
) -> dict[str, Any]:
    import cv2

    counts = {k: 0 for k in ("intersection", "union", "false_positive", "false_negative")}
    distances = []
    instability = []
    color_errors = []
    contamination = []
    previous_error = previous_boundary = None
    for i, (mask, pred) in enumerate(zip(truth, recovered)):
        boundary, decoded = edge(mask), edge(pred)
        counts["intersection"] += int((mask & pred).sum())
        counts["union"] += int((mask | pred).sum())
        counts["false_positive"] += int((~mask & pred).sum())
        counts["false_negative"] += int((mask & ~pred).sum())
        for source, target in ((boundary, decoded), (decoded, boundary)):
            if source.any():
                d = cv2.distanceTransform(
                    (~target).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
                )
                distances.append(
                    d[source] if target.any() else np.full(int(source.sum()), np.hypot(*mask.shape))
                )
        residual = boundary ^ decoded
        if previous_error is not None:
            instability.append(
                float(
                    (residual ^ previous_error).sum() / max(1, (boundary | previous_boundary).sum())
                )
            )
        previous_error, previous_boundary = residual, boundary
        if rgb is not None:
            band = (
                cv2.dilate(boundary.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & mask
            )
            actual = np.where(pred[..., None], rgb[i], 0)
            if band.any():
                color_errors.append(
                    float(
                        np.abs(
                            actual[band].astype(np.float32)
                            - expected_rgb[i][band].astype(np.float32)
                        ).mean()
                    )
                )
            fp = pred & ~mask
            if fp.any():
                contamination.append(float(np.sqrt(np.mean(rgb[i][fp].astype(np.float32) ** 2))))
    ds = np.concatenate(distances) if distances else np.zeros(1)
    return {
        **counts,
        "iou": counts["intersection"] / max(1, counts["union"]),
        "boundary_mean_px": float(ds.mean()),
        "boundary_p95_px": float(np.percentile(ds, 95)),
        "temporal_boundary_error_flips_per_true_boundary_pixel": float(np.mean(instability))
        if instability
        else 0.0,
        "boundary_rgb_mae": float(np.mean(color_errors)) if color_errors else None,
        "false_positive_rgb_rms": float(np.mean(contamination)) if contamination else None,
    }


def ambiguity() -> dict[str, Any]:
    # Even under the [32,255] guard restriction both opaque64 and half-alpha128 are legal.
    first = np.ones(3) * 64
    second = 0.5 * np.ones(3) * 128
    assert np.array_equal(first, second)
    return {
        "same_received_rgb": first.tolist(),
        "hypotheses": [
            {"alpha": 1.0, "foreground_rgb": [64, 64, 64]},
            {"alpha": 0.5, "foreground_rgb": [128, 128, 128]},
        ],
        "unique_alpha_or_color": False,
        "conclusion": "No RGB-only inverse can identify both opacity and arbitrary foreground color, even without a codec.",
    }


def promote(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for family in FAMILIES:
        for width in WIDTHS[1:]:
            pairs = []
            for wid in PILOTS:
                matched = [r for r in rows if r["window"] == wid and r["family"] == family]
                hard = next((r for r in matched if r["width_full_px"] == 0), None)
                soft = next((r for r in matched if r["width_full_px"] == width), None)
                if hard is None or soft is None:
                    continue
                a, b = hard["diagnostics"], soft["diagnostics"]
                pairs.append(
                    soft["container"]["bytes"] <= 0.9 * hard["container"]["bytes"]
                    and b["iou"] >= a["iou"] - 0.005
                    and b["boundary_mean_px"] <= a["boundary_mean_px"] + 0.25
                    and b["temporal_boundary_error_flips_per_true_boundary_pixel"]
                    <= a["temporal_boundary_error_flips_per_true_boundary_pixel"] + 0.005
                )
            result[f"{family}-w{width}"] = {
                "promotes": len(pairs) == 2 and all(pairs),
                "measured_windows": len(pairs),
            }
    return result


def validate(root: Path, full: bool) -> dict[str, Any]:
    import cv2

    r = json.loads((root / "edge-ramp.json").read_text())
    expected = {
        (w, f, s) for w in (PILOTS if full else PILOTS[:1]) for f in FAMILIES for s in WIDTHS
    }
    checks = {
        "cpu_only": r["cuda_visible"] == "" and r["gpu_hours"] == 0,
        "fixed_points": {(x["window"], x["family"], x["width_full_px"]) for x in r["rows"]}
        == expected
        and len(r["rows"]) == len(expected),
        "nonidentifiability": not r["ambiguity"]["unique_alpha_or_color"],
        "excerpt_frames": r["frames"] == (8 if full else 4),
        "zero_codec": all(
            s["hard_mask_exact"] and s["hard_rgb_max_error"] <= 1 and s["anti_leak"]
            for s in r["sources"]
        ),
    }
    if not full:
        checks["smoke_projection_fits_full650"] = r["wall_seconds"] * 4 + 120 < 650
    for i, row in enumerate(r["rows"]):
        c = row["container"]
        path = root / c["file"]
        checks[f"{i}:packet"] = (
            sha(path.read_bytes()) == c["sha256"] and path.stat().st_size == c["bytes"]
        )
        decoded = root / f"validate-{i}.yuv"
        command(
            [
                *ffmpeg_args(),
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rawvideo",
                str(decoded),
            ]
        )
        checks[f"{i}:decode"] = sha(decoded.read_bytes()) == c["track_decoded_sha256"][0]
        # Reconstruct the client mask from only the delivered track and constants.
        # No source matte, alpha ramp, nearest-color field or source crop is loaded.
        if row["family"] == "rgb-key":
            client = root / f"client-{i}"
            client.mkdir()
            pixels, _ = back_rgb({"decoded": str(decoded)}, r["frames"], 960, 540, client)
            _, recovered = decode_rgb(pixels)
            (client / "back.rgb").unlink()
            client.rmdir()
        else:
            y = np.fromfile(decoded, np.uint8).reshape(r["frames"], -1)[:, : 960 * 540]
            recovered = decode_luma(y.reshape(r["frames"], 540, 960))
        output = np.stack(
            [
                cv2.resize(m.astype(np.float32), (1920, 1080), interpolation=cv2.INTER_LINEAR)
                >= 0.5
                for m in recovered
            ]
        )
        checks[f"{i}:client_only_mask"] = array_sha(output) == row["decoded_mask_sha256"]
        decoded.unlink()
        checks[f"{i}:finite"] = all(
            np.isfinite(row["diagnostics"][k])
            for k in (
                "iou",
                "boundary_mean_px",
                "boundary_p95_px",
                "temporal_boundary_error_flips_per_true_boundary_pixel",
            )
        )
    return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport", type=Path)
    p.add_argument("--generated", type=Path, action="append", default=[])
    p.add_argument("--frames", type=int, choices=(4, 8), default=4)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            with tarfile.open(stage / "published.tar") as archive:
                archive.extractall(tmp, filter="data")
            checked = validate(Path(tmp) / "publish", os.environ.get("PS_STAGE") == "full")
        write(Path(os.environ["PS_VALIDATION_PATH"]), checked)
        return 0 if checked["passed"] else 1
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only mechanism check")
    import cv2
    from scipy.ndimage import distance_transform_edt
    from experiments.tennis import k6b

    cv2.setNumThreads(1)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir()
    with tarfile.open(a.transport) as archive:
        archive.extractall(scratch / "transport", filter="data")
    generated = k6b.load_generated([str(p) for p in a.generated], scratch / "generated")
    r: dict[str, Any] = {
        "rows": [],
        "sources": [],
        "frames": a.frames,
        "cuda_visible": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_hours": 0,
        "ambiguity": ambiguity(),
        "widths_full_px": list(WIDTHS),
        "fixed_preset": 4,
        "fixed_crf": 35,
        "code_sha256": sha(Path(__file__).read_bytes()),
        "inputs": {str(p): sha(p.read_bytes()) for p in [a.transport, *a.generated]},
        "tools": {t: svtav1.tool(t) for t in ("ffmpeg", "SvtAv1EncApp", "dav1d")},
        "scope": "Short stored-judged-frame excerpts only, not full240-frame transport rates or neural quality.",
    }
    began = time.monotonic()

    def save():
        r["wall_seconds"] = time.monotonic() - began
        r["cpu_seconds"] = sum(
            x.ru_utime + x.ru_stime
            for x in (
                resource.getrusage(resource.RUSAGE_SELF),
                resource.getrusage(resource.RUSAGE_CHILDREN),
            )
        )
        write(publish / "edge-ramp.json", r)
        write(stage / "edge-ramp.json", r)

    for wid in PILOTS if a.frames == 8 else PILOTS[:1]:
        packet = scratch / "transport/packets" / wid.replace("/", "__")
        meta, units = decode_keypoints((packet / "keypoints.kps").read_bytes())
        _, masks = decode_masks((packet / "exact-rle.matte").read_bytes())
        boxes = box_maps(units)
        times = generated[(units[0]["name"], "match5")]["t"][: a.frames].tolist()
        if len(times) != a.frames or not np.all(np.diff(times) == 3):
            raise ValueError("Expected first regular original judged ticks")
        fps = Fraction(*meta["fps"]) / 3
        original_list = []
        truth_list = []
        half_masks_list = []
        nearest_list = []
        for t in times:
            full_mask = np.repeat(np.repeat(masks[t].any(0), 2, 0), 2, 1)
            canvas = np.zeros((1080, 1920, 3), np.uint8)
            for name in meta["compositing_order"]:
                i = next(
                    i
                    for i, u in enumerate(units)
                    if u["name"] == name or u["name"].endswith("__" + name)
                )
                g = generated[(units[i]["name"], "match5")]
                j = np.flatnonzero(g["t"] == t)
                if len(j) != 1:
                    raise ValueError("Missing saved RGB: do not use target foreground")
                mask = np.repeat(np.repeat(masks[t, i], 2, 0), 2, 1)
                written = k6b.paste(canvas, g["crops"][int(j[0])], boxes[i][t], mask)
                if not np.array_equal(written, mask):
                    raise ValueError("Secondary excerpt lacks crop RGB for exact source mask")
            image = cv2.resize(canvas, (960, 540), interpolation=cv2.INTER_AREA)
            alpha = masks[t].any(0)
            indices = distance_transform_edt(~alpha, return_distances=False, return_indices=True)
            extension = image[tuple(indices)]
            nearest_list.append(extension)
            original_list.append(image)
            truth_list.append(full_mask)
            half_masks_list.append(alpha)
        original = np.stack(original_list)
        truth = np.stack(truth_list)
        half_masks = np.stack(half_masks_list)
        nearest = np.stack(nearest_list)
        hard = np.where(
            half_masks[..., None], np.round(32 + nearest.astype(np.float32) * 223 / 255), 0
        ).astype(np.uint8)
        restored, hard_mask = decode_rgb(hard)
        error = int(
            np.abs(
                restored[half_masks].astype(np.int16) - original[half_masks].astype(np.int16)
            ).max()
        )
        before = array_sha(decode_rgb(hard)[1])
        dummy_truth = ~half_masks
        after = array_sha(decode_rgb(hard)[1])
        assert dummy_truth.shape == half_masks.shape
        source = {
            "window": wid,
            "ticks": times,
            "fps": [fps.numerator, fps.denominator],
            "generated_rgb_sha256": array_sha(original),
            "source_mask_sha256": array_sha(truth),
            "hard_mask_exact": np.array_equal(hard_mask, half_masks),
            "hard_rgb_max_error": error,
            "anti_leak": before == after,
        }
        r["sources"].append(source)
        expected_full = np.stack(
            [cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_CUBIC) for im in nearest]
        )
        for width in WIDTHS:
            alpha = np.stack(
                [
                    cv2.resize(alpha_ramp(m, width), (960, 540), interpolation=cv2.INTER_AREA)
                    for m in truth
                ]
            )
            luma = np.round(alpha * 255).astype(np.uint8)
            for family in FAMILIES:
                tag = wid.replace("/", "__") + f"-{family}-w{width}"
                root = scratch / tag
                root.mkdir()
                if family == "rgb-key":
                    rgb = np.round(
                        alpha[..., None] * (32 + nearest.astype(np.float32) * 223 / 255)
                    ).astype(np.uint8)
                    raw, _ = rgb_yuv(rgb, root)
                    _, zero_mask = decode_rgb(rgb)
                else:
                    raw = root / "source.yuv"
                    np.concatenate(
                        [
                            luma.reshape(a.frames, -1),
                            np.full((a.frames, 960 * 540 // 2), 128, np.uint8),
                        ],
                        1,
                    ).tofile(raw)
                    zero_mask = decode_luma(luma)
                zero_full = np.stack(
                    [
                        cv2.resize(
                            m.astype(np.float32), (1920, 1080), interpolation=cv2.INTER_LINEAR
                        )
                        >= 0.5
                        for m in zero_mask
                    ]
                )
                zero_diag = diagnostics(truth, zero_full, None, expected_full)
                coded = code_yuv(
                    raw, root / "codec", a.frames, 960, 540, 35, fps, preset=4, timeout=100
                )
                if family == "rgb-key":
                    decoded_rgb, _ = back_rgb(coded, a.frames, 960, 540, root)
                    decoded_rgb, decoded_mask = decode_rgb(decoded_rgb)
                else:
                    y = (
                        np.fromfile(coded["decoded"], np.uint8)
                        .reshape(a.frames, -1)[:, : 960 * 540]
                        .reshape(a.frames, 540, 960)
                    )
                    decoded_mask = decode_luma(y)
                    decoded_rgb = None
                full_mask = np.stack(
                    [
                        cv2.resize(
                            m.astype(np.float32), (1920, 1080), interpolation=cv2.INTER_LINEAR
                        )
                        >= 0.5
                        for m in decoded_mask
                    ]
                )
                full_rgb = (
                    None
                    if decoded_rgb is None
                    else np.stack(
                        [
                            cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                            for im in decoded_rgb
                        ]
                    )
                )
                contract = {
                    "window": wid,
                    "family": family,
                    "width_full_px": width,
                    "size": [960, 540],
                    "output_size": [1920, 1080],
                    "ticks": times,
                    "fps": [fps.numerator, fps.denominator],
                    "preset": 4,
                    "crf": 35,
                    "guard_floor": 32,
                    "key_threshold": 16,
                    "luma_threshold": 128,
                    "decoder": "decoded channels and constant metadata only; no original mask/alpha",
                    "separate_mask_only": family == "luma-mask-only",
                }
                delivered = container([coded], publish / (tag + ".mkv"), contract)
                r["rows"].append(
                    {
                        "window": wid,
                        "family": family,
                        "width_full_px": width,
                        "container": delivered,
                        "diagnostics": diagnostics(truth, full_mask, full_rgb, expected_full),
                        "zero_codec": zero_diag,
                        "decoded_mask_sha256": array_sha(full_mask),
                        "excerpt_kbps": delivered["bytes"] * 8 / (a.frames / float(fps)) / 1000,
                    }
                )
                save()
    r["engineering_promotion"] = promote(r["rows"])
    save()
    checked = validate(publish, a.frames == 8)
    write(publish / "validation.json", checked)
    if not checked["passed"]:
        raise ValueError("Edge pilot correctness/scale gate failed")
    print(
        json.dumps(
            {
                "points": len(r["rows"]),
                "seconds": r["wall_seconds"],
                "cpu_seconds": r["cpu_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
