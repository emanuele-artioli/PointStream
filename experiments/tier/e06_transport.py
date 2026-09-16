"""Serialized E06 transport: appearance, masks, panorama side data, residual, ledger."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from experiments.tier.e03b_persist import DecodeCountError, frames_from_rgb24
from src.components.appearance.compressed import CompressedImageAppearance
from src.pipeline.reconstruction.background import MODE_FULL, BackgroundModelView
from src.pipeline.reconstruction.compositor import Placement, composite_clip
from src.pipeline.residual.codec import TransmittedResidual
from src.pipeline.residual.signal import apply_residual
from src.runner.client import (
    ClientPlacement,
    account_serialized_request,
    reconstruct_serialized_client,
    serialize_client_request,
)

PREDICTOR_PER_FRAME = "per_frame_crop"
PREDICTOR_BBOX_RESIZE = "bbox_resized_first_reference"
PREDICTORS = (PREDICTOR_PER_FRAME, PREDICTOR_BBOX_RESIZE)
PredictorName = Literal["per_frame_crop", "bbox_resized_first_reference"]

SIDE_DATA_BYTES = 1742


def require_predictor(name: str) -> PredictorName:
    if name in ("paste", "warped_reference", "warped-reference"):
        raise ValueError(
            f"{name!r} is not an E06 predictor name. "
            f"Use {PREDICTOR_PER_FRAME!r} (current-frame crops) or "
            f"{PREDICTOR_BBOX_RESIZE!r} (first crop resized into later union boxes). "
            "Neither is optical flow, per-object pose warp, or the E05 first-reference paste."
        )
    if name not in PREDICTORS:
        raise ValueError(f"unknown predictor {name!r}; expected one of {PREDICTORS}")
    return name  # type: ignore[return-value]


def decode_appearance(payload: bytes) -> np.ndarray:
    import cv2

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("appearance payload did not decode")
    return np.asarray(decoded, dtype=np.uint8)


def encode_appearance(crop: np.ndarray, *, quality: int = 50) -> bytes:
    encoder = CompressedImageAppearance(quality=quality, format="webp")
    _descriptor, payload = encoder.encode(crop)
    if not payload:
        raise ValueError("empty appearance payload")
    return payload


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2:
        raise ValueError(f"mask must be (H, W); got {selected.shape}")
    rows, cols = np.nonzero(selected)
    if rows.size == 0:
        raise ValueError("empty mask; cannot place an object")
    return (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)


def occupied_boxes(
    frames: np.ndarray, masks: np.ndarray
) -> tuple[tuple[int, tuple[int, int, int, int], np.ndarray], ...]:
    if frames.shape[0] != masks.shape[0]:
        raise ValueError("frames and masks must share a time axis")
    items: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        if not np.any(mask):
            continue
        bbox = bbox_from_mask(mask)
        x1, y1, x2, y2 = bbox
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        items.append((index, bbox, crop))
    if not items:
        raise ValueError("no occupied frames")
    return tuple(items)


def client_placements(
    frames: np.ndarray,
    masks: np.ndarray,
    predictor: str,
    *,
    quality: int = 50,
) -> tuple[tuple[ClientPlacement, ...], dict[str, bytes], int]:
    name = require_predictor(predictor)
    occupied = occupied_boxes(frames, masks)
    if name == PREDICTOR_PER_FRAME:
        placements = []
        appearance_bytes = 0
        for index, bbox, crop in occupied:
            payload = encode_appearance(crop, quality=quality)
            appearance_bytes += len(payload)
            placements.append(
                ClientPlacement(
                    encoded_crop=payload,
                    bbox=bbox,
                    mask=np.asarray(masks[index], dtype=bool),
                    object_id="union",
                    frame_index=index,
                    is_generated=False,
                )
            )
        return tuple(placements), {}, appearance_bytes
    reference = encode_appearance(occupied[0][2], quality=quality)
    placements = tuple(
        ClientPlacement(
            bbox=bbox,
            mask=np.asarray(masks[index], dtype=bool),
            object_id="union",
            frame_index=index,
            is_generated=False,
        )
        for index, bbox, _crop in occupied
    )
    return placements, {"union": reference}, len(reference)


def background_view(
    *,
    bitstream: bytes,
    side: bytes,
    plate: np.ndarray,
    homographies: np.ndarray,
    width: int,
    height: int,
    sidecar_codec: str = "vvc",
    expected_side_bytes: int | None = SIDE_DATA_BYTES,
) -> BackgroundModelView:
    if expected_side_bytes is not None and len(side) != expected_side_bytes:
        raise ValueError(f"panorama side data must be {expected_side_bytes} bytes; got {len(side)}")
    maps = tuple(tuple(float(value) for value in row.reshape(-1)) for row in homographies)
    return BackgroundModelView(
        plate=np.asarray(plate, dtype=np.uint8),
        homographies=maps,
        mode=MODE_FULL,
        scene_id="federer_djokovic/scene_007",
        width=width,
        height=height,
        payload_bytes=len(bitstream),
        geometry_header=side,
        geometry_header_bytes=len(side),
        wire_payloads=(bitstream,),
        wire_geometry_headers=(side,),
        sidecar_codec=sidecar_codec,
    )


def serialize_setting(
    *,
    background: BackgroundModelView,
    frames: np.ndarray,
    masks: np.ndarray,
    predictor: str,
    residual: TransmittedResidual | None,
    quality: int = 50,
) -> bytes:
    placements, references, _actor = client_placements(frames, masks, predictor, quality=quality)
    return serialize_client_request(
        background=background,
        frame_count=int(frames.shape[0]),
        height=int(frames.shape[1]),
        width=int(frames.shape[2]),
        placements=placements,
        residual_payload=residual,
        require_compressed=True,
        references=references or None,
        generator_meta=None,
    )


def reconcile_ledger(
    payload: bytes,
    *,
    panorama_b: int,
    actor_reference_f: int,
    residual_r: int,
) -> dict[str, Any]:
    transport_t = len(payload)
    subledger = account_serialized_request(
        payload,
        residual=residual_r,
        panorama=panorama_b,
        actor_reference=actor_reference_f,
    )
    metadata_m = int(subledger.total)
    unallocated_h = 0
    summed = panorama_b + actor_reference_f + residual_r + metadata_m + unallocated_h
    if summed != transport_t:
        raise ValueError(
            f"ledger does not reconcile: B={panorama_b} F={actor_reference_f} "
            f"R={residual_r} M={metadata_m} H={unallocated_h} sum={summed} T={transport_t}"
        )
    if residual_r < 0:
        raise ValueError("residual bytes cannot be negative")
    return {
        "panorama": panorama_b,
        "actor_reference": actor_reference_f,
        "residual": residual_r,
        "metadata": metadata_m,
        "unallocated_H": unallocated_h,
        "envelope_overhead": int(subledger.envelope_overhead),
        "transport_total": transport_t,
        "metadata_subledger": subledger.as_dict(),
        "pose_present": int(subledger.pose_motion) > 0,
        "raw_parts": (),
        "reconciled": True,
    }


def reconstruct_standalone(payload: bytes) -> np.ndarray:
    frames = np.asarray(
        reconstruct_serialized_client(payload, require_compressed=True),
        dtype=np.uint8,
    )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise DecodeCountError(f"standalone reconstruct shape {frames.shape}")
    return frames


def ordinary_composite(
    background_frames: np.ndarray,
    frames: np.ndarray,
    masks: np.ndarray,
    predictor: str,
    *,
    residual: TransmittedResidual | None,
    quality: int = 50,
) -> np.ndarray:
    placements, references, _actor = client_placements(frames, masks, predictor, quality=quality)
    pipeline: list[Placement] = []
    ref_crops = {key: decode_appearance(value) for key, value in references.items()}
    for item in placements:
        if item.encoded_crop is not None:
            crop = decode_appearance(item.encoded_crop)
        else:
            crop = ref_crops[item.object_id]
        pipeline.append(
            Placement(
                crop=crop,
                bbox=item.bbox,
                mask=item.mask,
                object_id=item.object_id,
                frame_index=item.frame_index,
            )
        )
    base = composite_clip(
        background_frames,
        tuple(pipeline),
        use_heuristic_mask=False,
    )
    if residual is None:
        return np.asarray(base, dtype=np.uint8)
    return np.asarray(apply_residual(base, residual), dtype=np.uint8)


def require_pixel_parity(ordinary: np.ndarray, standalone: np.ndarray, *, source: str) -> None:
    if ordinary.shape != standalone.shape:
        raise DecodeCountError(
            f"{source}: ordinary shape {ordinary.shape} != standalone {standalone.shape}"
        )
    if not np.array_equal(ordinary, standalone):
        raise DecodeCountError(f"{source}: ordinary and standalone pixels differ")


def require_exact_count(frames: np.ndarray, *, expected: int, height: int, width: int, source: str) -> None:
    raw = np.ascontiguousarray(frames).tobytes()
    frames_from_rgb24(
        raw,
        width=width,
        height=height,
        expected_count=expected,
        source=source,
    )
