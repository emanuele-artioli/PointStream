"""Registered plate, warp-error residual, and a budgeted appearance arm.

Step 1 codes a registered plate as VVC intra QP 40, charges one float32
homography per frame, pastes one AV1 crop through bbox motion, and codes the
background error of the warped plate as a VVC residual at two QPs. Step 3
admits extra crops only while foreground MSE stays above the ladder threshold
and the appearance payload stays inside 12 kB. The rate-distortion curve and
the 192-frame window run only when a 48-frame point can cross an anchor.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import numpy as np

from experiments.modular.appearance_motion_probe import (
    _bbox_affine,
    _bboxes,
    _box_payload,
    _warp_crop,
)
from experiments.modular.background_registration_probe import _reconstruct
from experiments.modular.image_codec_probe import FRAMES, MASKS
from experiments.modular.measured_ladder import (
    CROP_CODEC,
    CROP_QP,
    RESIDUAL_CODEC,
    RESIDUAL_PRESET,
    _encode_appearance_crop,
    _fg_mse_vs_keyframe,
    _intra_roundtrip,
    _pack_metadata,
    _paste_crop,
    _rgb_to_bgr,
    _roundtrip_clip,
    _webp_roundtrip,
    load_sequence,
    score_regions,
)
from src.components.background.plate import build_plate
from src.components.background.sidecar import build_sidecar

OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/warp-residual/federer007.json"
)
LONG_FRAMES = Path(
    "/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/clips/"
    "alcaraz_highlights/scene_000/window_192"
)
LONG_MASKS = Path(
    "/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/clips/"
    "alcaraz_highlights/scene_000/masks_192.npz"
)
LONG_OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/warp-residual/alcaraz192.json"
)

APPEARANCE_BUDGET = 12_000
MSE_THRESHOLD = 50.0
PROMOTE_PLATE_RESIDUAL_BYTES = 200_000
PACK_WITHIN_BYTES = 5_000

# Weighted scores already measured on this mask. Re-scoring the saved decodes
# would repeat that comparison; the probe cites these anchors.
ANCHORS: dict[str, dict[str, float | int]] = {
    "vvc_qp46": {
        "bytes": 112_295,
        "psnr_weighted": 24.58878360131408,
        "psnr_fg": 21.685556193841492,
        "psnr_bg": 31.362980885416793,
    },
    "av1_qp54": {
        "bytes": 316_061,
        "psnr_weighted": 30.067556734663093,
        "psnr_fg": 27.40115254652083,
        "psnr_bg": 36.28916650699504,
    },
}
LONG_ANCHORS: dict[str, dict[str, float | int]] = {
    "vvc": {"bytes": 258_024, "psnr_overall": 33.94113068630488},
    "av1": {"bytes": 382_825, "psnr_overall": 38.899616835555875},
}


def dominates(
    candidate_bytes: int,
    candidate_weighted: float | None,
    anchor_bytes: int,
    anchor_weighted: float | None,
) -> bool:
    """True when the candidate Pareto-dominates the anchor on rate and quality.

    Fewer or equal bytes and higher or equal weighted PSNR, with at least one
    strict inequality. A missing weighted score does not dominate. Callers
    rely on a point that is only smaller, or only higher quality, returning
    False.
    """
    if candidate_weighted is None or anchor_weighted is None:
        return False
    fewer_or_equal = int(candidate_bytes) <= int(anchor_bytes)
    better_or_equal = float(candidate_weighted) >= float(anchor_weighted)
    strict = int(candidate_bytes) < int(anchor_bytes) or float(candidate_weighted) > float(
        anchor_weighted
    )
    return fewer_or_equal and better_or_equal and strict


def admit_keyframe(
    *,
    spent: int,
    next_cost: int,
    budget: int,
    mse: float,
    threshold: float,
) -> bool:
    """Admit a crop when the warp error exceeds ``threshold`` and the budget holds.

    ``spent`` and ``next_cost`` are appearance payload bytes, not metadata.
    Equality with ``threshold`` does not admit. A caller relies on a crop that
    would pass the threshold and miss the budget returning False.
    """
    if next_cost < 0 or spent < 0 or budget < 0:
        raise ValueError("appearance byte counts must be non-negative")
    return float(mse) > float(threshold) and int(spent) + int(next_cost) <= int(budget)


def foreground_error_split(
    reference_bgr: np.ndarray,
    reconstruction_bgr: np.ndarray,
    mask: np.ndarray,
    covered: np.ndarray,
) -> dict[str, float | None]:
    """Split foreground PSNR into pixels the paste covered and pixels it left.

    ``covered`` is the paste footprint. Uncovered foreground is ``mask`` and
    not ``covered``. An empty subset has PSNR None. A caller relies on a
    perfect covered region and a constant uncovered error producing a finite
    uncovered PSNR and a None covered PSNR.
    """
    reference = np.asarray(reference_bgr)
    reconstruction = np.asarray(reconstruction_bgr)
    fg = np.asarray(mask, dtype=bool)
    wrote = np.asarray(covered, dtype=bool) & fg
    missed = fg & ~np.asarray(covered, dtype=bool)
    fg_count = int(fg.sum())
    return {
        "foreground_pixels": fg_count,
        "covered_fraction": (int(wrote.sum()) / fg_count) if fg_count else None,
        "covered_psnr": _region_psnr(reference, reconstruction, wrote),
        "uncovered_psnr": _region_psnr(reference, reconstruction, missed),
    }


def _region_psnr(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    region: np.ndarray,
) -> float | None:
    if not np.any(region):
        return None
    diff = reference.astype(np.float64) - reconstruction.astype(np.float64)
    mse = float(np.mean(np.square(diff[region])))
    if mse <= 0.0:
        return None
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _pad_even(image: np.ndarray) -> np.ndarray:
    height, width = int(image.shape[0]), int(image.shape[1])
    pad_h, pad_w = height % 2, width % 2
    if pad_h == 0 and pad_w == 0:
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(
        cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE)
    )


def _encode_plate(image_bgr: np.ndarray, codec: str, qp: int) -> tuple[bytes, np.ndarray]:
    """Encode one BGR plate and return it on the padded even canvas."""
    padded = _pad_even(image_bgr)
    if codec == "webp":
        payload, decoded = _webp_roundtrip(padded, qp)
    elif codec == "vvc":
        coder = build_sidecar("vvc", intra_qp=qp)
        payload = coder.encode(padded)
        decoded = coder.decode(payload)
    else:
        payload, decoded = _intra_roundtrip(padded, codec, qp)
    canvas = np.zeros_like(padded)
    height = min(decoded.shape[0], canvas.shape[0])
    width = min(decoded.shape[1], canvas.shape[1])
    canvas[:height, :width] = decoded[:height, :width]
    return payload, canvas


def _score_bgr(
    reference_rgb: np.ndarray,
    reconstruction_bgr: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    overall, fg, bg, weighted = score_regions(
        reference_rgb,
        reconstruction_bgr[..., ::-1],
        mask,
        fg_weight=0.70,
        bg_weight=0.30,
    )
    return {
        "psnr_overall": overall,
        "psnr_fg": fg,
        "psnr_bg": bg,
        "psnr_weighted": weighted,
    }


def _wins(total_bytes: int, weighted: float | None, anchors: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {
        name: dominates(
            total_bytes,
            weighted,
            int(anchor["bytes"]),
            float(anchor["psnr_weighted"]) if "psnr_weighted" in anchor else None,
        )
        for name, anchor in anchors.items()
    }


def _apply_background_residual(
    base_bgr: np.ndarray,
    decoded_error_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    signed = decoded_error_bgr.astype(np.int16) - 128
    corrected = base_bgr.astype(np.int16).copy()
    background = ~mask
    corrected[background] = np.clip(corrected[background] + signed[background], 0, 255)
    return corrected.astype(np.uint8)


def _single_crop_frames(
    plate_frames: np.ndarray,
    crop_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``crop_bgr`` from box 0 onto every frame. Return frames and coverage."""
    frames: list[np.ndarray] = []
    covered: list[np.ndarray] = []
    frame_shape = (int(plate_frames.shape[1]), int(plate_frames.shape[2]))
    for index, box in enumerate(boxes):
        warped = _warp_crop(crop_bgr, _bbox_affine(boxes[0], box), boxes[0], mask[index], frame_shape)
        frame = plate_frames[index].copy()
        valid = np.any(warped != 0, axis=-1) & mask[index]
        frame[valid] = warped[valid]
        frames.append(frame)
        covered.append(valid)
    return np.stack(frames), np.stack(covered)


def _budgeted_frames(
    plate_frames: np.ndarray,
    frames_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    mask: np.ndarray,
    *,
    budget: int = APPEARANCE_BUDGET,
    threshold: float = MSE_THRESHOLD,
) -> dict[str, Any]:
    """Send frame 0, then more crops while MSE and the 12 kB budget allow."""
    first_payload, first_decoded = _encode_appearance_crop(frames_bgr[0], boxes[0])
    payloads = [first_payload]
    decoded = {0: first_decoded}
    keyframes = [0]
    suppressed = 0
    spent = len(first_payload)
    active = 0
    reconstruction: list[np.ndarray] = []
    covered: list[np.ndarray] = []
    frame_shape = (int(plate_frames.shape[1]), int(plate_frames.shape[2]))
    for index, box in enumerate(boxes):
        source_box = boxes[active]
        warped = _warp_crop(
            decoded[active],
            _bbox_affine(source_box, box),
            source_box,
            mask[index],
            frame_shape,
        )
        frame = plate_frames[index].copy()
        valid = np.any(warped != 0, axis=-1) & mask[index]
        frame[valid] = warped[valid]
        if index > 0:
            mse = _fg_mse_vs_keyframe(frames_bgr[index], frame, mask[index])
            if mse > threshold:
                payload, crop = _encode_appearance_crop(frames_bgr[index], box)
                if admit_keyframe(
                    spent=spent,
                    next_cost=len(payload),
                    budget=budget,
                    mse=mse,
                    threshold=threshold,
                ):
                    payloads.append(payload)
                    decoded[index] = crop
                    keyframes.append(index)
                    spent += len(payload)
                    active = index
                    frame = _paste_crop(plate_frames[index], crop, box, fg_mask=mask[index])
                    y1, y2, x1, x2 = box
                    valid = np.zeros(mask[index].shape, dtype=bool)
                    valid[y1:y2, x1:x2] = mask[index, y1:y2, x1:x2]
                else:
                    suppressed += 1
        reconstruction.append(frame)
        covered.append(valid)
    metadata = _pack_metadata([(index, *boxes[index]) for index in keyframes])
    motion_boxes = [boxes[index] for index in range(len(boxes)) if index not in set(keyframes)]
    return {
        "reconstruction": np.stack(reconstruction),
        "covered": np.stack(covered),
        "keyframes": keyframes,
        "suppressed": suppressed,
        "bytes_appearance": sum(len(item) for item in payloads),
        "bytes_metadata": len(metadata),
        "bytes_motion": len(_box_payload(motion_boxes)) if motion_boxes else 0,
    }


def _background_error(frames_bgr: np.ndarray, plate_frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    error = np.clip(frames_bgr.astype(np.int16) - plate_frames.astype(np.int16) + 128, 0, 255)
    error = np.ascontiguousarray(error.astype(np.uint8))
    error[mask] = 128
    return error


def _row(
    name: str,
    *,
    reconstruction_bgr: np.ndarray,
    covered: np.ndarray,
    frames_rgb: np.ndarray,
    frames_bgr: np.ndarray,
    mask: np.ndarray,
    plate_bytes: int,
    map_bytes: int,
    appearance_bytes: int,
    metadata_bytes: int,
    motion_bytes: int,
    residual_bytes: int,
    anchors: dict[str, dict[str, Any]],
    details: dict[str, Any],
) -> dict[str, Any]:
    scores = _score_bgr(frames_rgb, reconstruction_bgr, mask)
    total = (
        plate_bytes + map_bytes + appearance_bytes + metadata_bytes + motion_bytes + residual_bytes
    )
    split = foreground_error_split(frames_bgr, reconstruction_bgr, mask, covered)
    row: dict[str, Any] = {
        "arm": name,
        "bytes_plate": plate_bytes,
        "bytes_maps": map_bytes,
        "bytes_appearance": appearance_bytes,
        "bytes_metadata": metadata_bytes,
        "bytes_motion": motion_bytes,
        "bytes_residual": residual_bytes,
        "total_bytes": total,
        "wins": _wins(total, scores["psnr_weighted"], anchors),
    }
    row.update(scores)
    row.update(split)
    row.update(details)
    return row


def measure_window(
    frames_dir: Path,
    mask_path: Path,
    n_frames: int,
    anchors: dict[str, dict[str, Any]],
    *,
    run_curve: bool,
) -> dict[str, Any]:
    """Run the approved  probe on one window and return the JSON object."""
    frames_rgb, mask = load_sequence(frames_dir, mask_path, n_frames)
    frames_bgr = _rgb_to_bgr(frames_rgb)
    height, width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    print(f"building registered plate for {n_frames} frames", flush=True)
    plate, maps = build_plate(frames_bgr, masks=mask, register=True)
    map_bytes = n_frames * 9 * 4
    boxes = _bboxes(mask)

    print("encoding VVC intra QP 40 plate", flush=True)
    plate_payload, plate_dec = _encode_plate(np.asarray(plate), "vvc", 40)
    plate_frames = _reconstruct(
        plate_dec,
        maps,
        registered=True,
        frame_shape=(height, width),
    )
    crop_payload, crop_dec = _encode_appearance_crop(frames_bgr[0], boxes[0])
    single, single_covered = _single_crop_frames(plate_frames, crop_dec, boxes, mask)
    single_motion = _box_payload(boxes[1:])
    single_meta = _pack_metadata([(0, *boxes[0])])

    print("encoding background warp residual", flush=True)
    error = _background_error(frames_bgr, plate_frames, mask)
    residual_payloads: dict[int, tuple[bytes, np.ndarray]] = {}
    for qp in (46, 40):
        print(f"  residual QP {qp}", flush=True)
        payload, decoded = _roundtrip_clip(
            error[..., ::-1],
            codec=RESIDUAL_CODEC,
            qp=qp,
            preset=RESIDUAL_PRESET,
        )
        residual_payloads[qp] = (payload, decoded)

    rows: list[dict[str, Any]] = []
    rows.append(
        _row(
            "vvc_plate_single_crop",
            reconstruction_bgr=single,
            covered=single_covered,
            frames_rgb=frames_rgb,
            frames_bgr=frames_bgr,
            mask=mask,
            plate_bytes=len(plate_payload),
            map_bytes=map_bytes,
            appearance_bytes=len(crop_payload),
            metadata_bytes=len(single_meta),
            motion_bytes=len(single_motion),
            residual_bytes=0,
            anchors=anchors,
            details={"plate_codec": "vvc", "plate_qp": 40, "residual_qp": None},
        )
    )
    for qp, (payload, decoded) in residual_payloads.items():
        corrected = _apply_background_residual(plate_frames, decoded, mask)
        # Foreground paste is unchanged; rewrite it onto the corrected plate.
        with_fg, covered = _single_crop_frames(corrected, crop_dec, boxes, mask)
        rows.append(
            _row(
                f"vvc_plate_single_crop_bg_residual_qp{qp}",
                reconstruction_bgr=with_fg,
                covered=covered,
                frames_rgb=frames_rgb,
                frames_bgr=frames_bgr,
                mask=mask,
                plate_bytes=len(plate_payload),
                map_bytes=map_bytes,
                appearance_bytes=len(crop_payload),
                metadata_bytes=len(single_meta),
                motion_bytes=len(single_motion),
                residual_bytes=len(payload),
                anchors=anchors,
                details={"plate_codec": "vvc", "plate_qp": 40, "residual_qp": qp},
            )
        )
        print(rows[-1], flush=True)

    cheaper = min(len(item[0]) for item in residual_payloads.values())
    plate_plus = len(plate_payload) + map_bytes + cheaper
    best_weighted = max(float(row["psnr_weighted"] or 0.0) for row in rows)
    vvc_weighted = float(anchors["vvc_qp46"]["psnr_weighted"]) if "vvc_qp46" in anchors else None
    promote_curve = plate_plus < PROMOTE_PLATE_RESIDUAL_BYTES and (
        vvc_weighted is not None and best_weighted >= vvc_weighted
    )
    print(
        f"plate+maps+cheaper residual={plate_plus} promote_curve={promote_curve}",
        flush=True,
    )

    if run_curve and promote_curve and "vvc_qp46" in anchors:
        rows.extend(
            _curve_rows(
                plate=np.asarray(plate),
                maps=maps,
                frames_rgb=frames_rgb,
                frames_bgr=frames_bgr,
                mask=mask,
                boxes=boxes,
                crop_dec=crop_dec,
                crop_bytes=len(crop_payload),
                map_bytes=map_bytes,
                anchors=anchors,
                existing_residual=residual_payloads,
                vvc_plate_bytes=len(plate_payload),
            )
        )

    print("budgeted keyframes", flush=True)
    budgeted = _budgeted_frames(plate_frames, frames_bgr, boxes, mask)
    rows.append(
        _row(
            "vvc_plate_budgeted_keyframes",
            reconstruction_bgr=budgeted["reconstruction"],
            covered=budgeted["covered"],
            frames_rgb=frames_rgb,
            frames_bgr=frames_bgr,
            mask=mask,
            plate_bytes=len(plate_payload),
            map_bytes=map_bytes,
            appearance_bytes=int(budgeted["bytes_appearance"]),
            metadata_bytes=int(budgeted["bytes_metadata"]),
            motion_bytes=int(budgeted["bytes_motion"]),
            residual_bytes=0,
            anchors=anchors,
            details={
                "plate_codec": "vvc",
                "plate_qp": 40,
                "residual_qp": None,
                "keyframes": budgeted["keyframes"],
                "suppressed": budgeted["suppressed"],
                "appearance_budget": APPEARANCE_BUDGET,
                "mse_threshold": MSE_THRESHOLD,
            },
        )
    )
    # The background residual does not depend on the foreground paste.
    for qp, (payload, decoded) in residual_payloads.items():
        corrected = _apply_background_residual(plate_frames, decoded, mask)
        budgeted_fg = _budgeted_frames(corrected, frames_bgr, boxes, mask)
        rows.append(
            _row(
                f"vvc_plate_budgeted_keyframes_bg_residual_qp{qp}",
                reconstruction_bgr=budgeted_fg["reconstruction"],
                covered=budgeted_fg["covered"],
                frames_rgb=frames_rgb,
                frames_bgr=frames_bgr,
                mask=mask,
                plate_bytes=len(plate_payload),
                map_bytes=map_bytes,
                appearance_bytes=int(budgeted_fg["bytes_appearance"]),
                metadata_bytes=int(budgeted_fg["bytes_metadata"]),
                motion_bytes=int(budgeted_fg["bytes_motion"]),
                residual_bytes=len(payload),
                anchors=anchors,
                details={
                    "plate_codec": "vvc",
                    "plate_qp": 40,
                    "residual_qp": qp,
                    "keyframes": budgeted_fg["keyframes"],
                    "suppressed": budgeted_fg["suppressed"],
                    "appearance_budget": APPEARANCE_BUDGET,
                },
            )
        )
        print(rows[-1]["arm"], rows[-1]["total_bytes"], rows[-1]["psnr_weighted"], rows[-1]["wins"], flush=True)

    any_win = any(any(row["wins"].values()) for row in rows)
    within_pack = False
    for row in rows:
        for anchor in anchors.values():
            if "psnr_weighted" not in anchor:
                continue
            if abs(int(row["total_bytes"]) - int(anchor["bytes"])) <= PACK_WITHIN_BYTES:
                within_pack = True
    return {
        "source": str(frames_dir),
        "n_frames": n_frames,
        "metric": {"foreground_weight": 0.70, "background_weight": 0.30},
        "anchors": anchors,
        "plate_shape": list(np.asarray(plate).shape),
        "promote_curve": promote_curve,
        "plate_maps_cheaper_residual_bytes": plate_plus,
        "any_pareto_win": any_win,
        "within_5kb_of_an_anchor": within_pack,
        "rows": rows,
    }


def _curve_rows(
    *,
    plate: np.ndarray,
    maps: tuple[tuple[float, ...], ...],
    frames_rgb: np.ndarray,
    frames_bgr: np.ndarray,
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    crop_dec: np.ndarray,
    crop_bytes: int,
    map_bytes: int,
    anchors: dict[str, dict[str, Any]],
    existing_residual: dict[int, tuple[bytes, np.ndarray]],
    vvc_plate_bytes: int,
) -> list[dict[str, Any]]:
    """At most the remaining curve encodes: WebP plate and residual QP 32."""
    height, width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    rows: list[dict[str, Any]] = []
    plates = {
        ("vvc", 40): (vvc_plate_bytes, None),
    }
    print("encoding WebP q40 registered plate for the curve", flush=True)
    webp_payload, webp_dec = _encode_plate(plate, "webp", 40)
    plates[("webp", 40)] = (len(webp_payload), webp_dec)
    # Reuse the VVC plate frames already warped by reconstructing WebP only.
    webp_frames = _reconstruct(webp_dec, maps, registered=True, frame_shape=(height, width))
    vvc_frames = None
    for codec, qp in (("webp", 40),):
        base = webp_frames
        error = _background_error(frames_bgr, base, mask)
        for residual_qp in (46, 40, 32):
            print(f"curve {codec} plate residual QP {residual_qp}", flush=True)
            payload, decoded = _roundtrip_clip(
                error[..., ::-1],
                codec=RESIDUAL_CODEC,
                qp=residual_qp,
                preset=RESIDUAL_PRESET,
            )
            corrected = _apply_background_residual(base, decoded, mask)
            with_fg, covered = _single_crop_frames(corrected, crop_dec, boxes, mask)
            rows.append(
                _row(
                    f"{codec}_plate_single_crop_bg_residual_qp{residual_qp}",
                    reconstruction_bgr=with_fg,
                    covered=covered,
                    frames_rgb=frames_rgb,
                    frames_bgr=frames_bgr,
                    mask=mask,
                    plate_bytes=plates[(codec, qp)][0],
                    map_bytes=map_bytes,
                    appearance_bytes=crop_bytes,
                    metadata_bytes=len(_pack_metadata([(0, *boxes[0])])),
                    motion_bytes=len(_box_payload(boxes[1:])),
                    residual_bytes=len(payload),
                    anchors=anchors,
                    details={"plate_codec": codec, "plate_qp": qp, "residual_qp": residual_qp},
                )
            )
    # One extra residual QP on the VVC plate, reusing QP 40 and 46.
    if vvc_frames is None and 32 not in existing_residual:
        # The caller already holds the VVC warped frames; this branch encodes
        # only the missing QP against a freshly warped decode of the same plate.
        print("curve VVC plate residual QP 32", flush=True)
        payload, plate_dec = _encode_plate(plate, "vvc", 40)
        base = _reconstruct(plate_dec, maps, registered=True, frame_shape=(height, width))
        error = _background_error(frames_bgr, base, mask)
        residual, decoded = _roundtrip_clip(
            error[..., ::-1],
            codec=RESIDUAL_CODEC,
            qp=32,
            preset=RESIDUAL_PRESET,
        )
        corrected = _apply_background_residual(base, decoded, mask)
        with_fg, covered = _single_crop_frames(corrected, crop_dec, boxes, mask)
        rows.append(
            _row(
                "vvc_plate_single_crop_bg_residual_qp32",
                reconstruction_bgr=with_fg,
                covered=covered,
                frames_rgb=frames_rgb,
                frames_bgr=frames_bgr,
                mask=mask,
                plate_bytes=len(payload),
                map_bytes=map_bytes,
                appearance_bytes=crop_bytes,
                metadata_bytes=len(_pack_metadata([(0, *boxes[0])])),
                motion_bytes=len(_box_payload(boxes[1:])),
                residual_bytes=len(residual),
                anchors=anchors,
                details={"plate_codec": "vvc", "plate_qp": 40, "residual_qp": 32},
            )
        )
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=("short", "long"), default="short")
    args = parser.parse_args()
    if args.window == "short":
        result = measure_window(FRAMES, MASKS, 48, ANCHORS, run_curve=True)
        destination = OUT
    else:
        result = measure_window(LONG_FRAMES, LONG_MASKS, 192, LONG_ANCHORS, run_curve=False)
        destination = LONG_OUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}", flush=True)
    print(
        f"any_pareto_win={result['any_pareto_win']} promote_curve={result['promote_curve']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
