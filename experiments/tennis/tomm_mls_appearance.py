"""Cold AVIF appearance cache for the native 240-frame MLS receiver."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from experiments.tennis import k6b, w8_payload as wire
from experiments.tennis.tomm_warp_v2_pilot import _crop_from_canvas

QUALITY = 90
SWITCH_MARGIN = 0.10
MAX_AVIF_BYTES = k6b.GEN_H * k6b.GEN_W * 3


def orientation(poses: np.ndarray) -> np.ndarray:
    """Describe signed shoulder and hip spread in torso-length units."""
    torso = poses[:, [5, 6, 11, 12]]
    shoulders = torso[:, :2, :2]
    hips = torso[:, 2:, :2]
    length = np.linalg.norm(shoulders.mean(axis=1) - hips.mean(axis=1), axis=1)
    valid = (torso[:, :, 2] >= 0.5).all(axis=1) & (length > 1.0)
    spread = np.column_stack((shoulders[:, 0, 0] - shoulders[:, 1, 0],
                              0.5 * (hips[:, 0, 0] - hips[:, 1, 0])))
    descriptor = spread / np.maximum(length[:, None], 1.0)
    descriptor[~valid] = np.nan
    return descriptor


def select_reference_ticks(poses: np.ndarray, views: int) -> list[int]:
    """Keep tick zero and greedily maximize the distance to the nearest selected view."""
    descriptors = orientation(poses)
    candidates = np.flatnonzero(np.isfinite(descriptors).all(axis=1) & ((poses[:, :, 2] >= 0.5).sum(axis=1) >= 8))
    selected = [0]
    candidates = candidates[candidates != 0]
    distance = np.linalg.norm(descriptors[candidates] - descriptors[0], axis=1)
    if not np.isfinite(descriptors[0]).all():
        distance = np.linalg.norm(descriptors[candidates], axis=1)
    while len(selected) < views and len(candidates):
        best = int(np.argmax(distance))
        tick = int(candidates[best])
        selected.append(tick)
        candidates = np.delete(candidates, best)
        distance = np.minimum(np.delete(distance, best),
                              np.linalg.norm(descriptors[candidates] - descriptors[tick], axis=1))
    return selected


def view_schedule(poses: np.ndarray, ticks: list[int]) -> list[int]:
    """Pick the nearest usable pose after two consecutive frames beyond the switch margin."""
    descriptors = orientation(poses)
    distances = np.linalg.norm(descriptors[:, None] - descriptors[ticks], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    # Leave the active view in place when the current frame has no usable torso.
    choices = []
    active = 0
    pending = 0
    for row in distances:
        nearest = int(np.argmin(row))
        if row[nearest] + SWITCH_MARGIN < row[active]:
            if nearest == pending:
                active = nearest
            pending = nearest
        else:
            pending = active
        choices.append(active)
    return choices


def encode_views(
    frames: list[np.ndarray], tracks: dict[str, dict[str, Any]], views: int
) -> dict[str, list[tuple[int, bytes]]]:
    """Sender only. Crop source RGB using the same pixel and box geometry as GrabCut."""
    encoded: dict[str, list[tuple[int, bytes]]] = {}
    for role, track in tracks.items():
        ticks = select_reference_ticks(track["selection_kp"], views)
        if len(ticks) != views:
            raise ValueError(f"{role} has too few confidently tracked frames for K={views}")
        encoded[role] = []
        for tick in ticks[1:]:
            box = np.asarray(k6b.int_box(track["boxes"][tick]), dtype=np.float64)
            crop = _crop_from_canvas(frames[tick], box)
            with BytesIO() as dst:
                Image.fromarray(crop).save(dst, format="AVIF", quality=QUALITY, subsampling="4:4:4", max_threads=1)
                encoded[role].append((tick, dst.getvalue()))
    return encoded


def pack_cache(encoded: dict[str, list[tuple[int, bytes]]], carrier_sha256: str, views: int) -> bytes:
    """Serialize additional references as shot-cold PSR sections with implicit crop geometry."""
    sections = [
        (f"{role}.{slot}.{tick}", "reference", payload)
        for role, images in sorted(encoded.items())
        for slot, (tick, payload) in enumerate(images[: views - 1], start=1)
    ]
    return wire.pack(_profile(carrier_sha256, views), sections)


def _profile(carrier_sha256: str, views: int) -> dict[str, Any]:
    return {"profile": "psr-mls-appearance-v1", "carrier_sha256": carrier_sha256,
            "codec": "avif", "quality": QUALITY, "views": views}


def unpack_cache(
    blob: bytes, carrier_sha256: str, views: int, roles: tuple[str, ...], track_frames: int
) -> tuple[dict[str, list[tuple[int, np.ndarray]]], dict[str, int]]:
    """Receiver only. Validate the carrier binding and decode AVIF pixels at their declared slots."""
    if views not in (2, 3) or len(blob) > len(roles) * (views - 1) * MAX_AVIF_BYTES + 4096:
        raise ValueError("invalid appearance cache size or K")
    sections, accounting = wire.unpack(blob, _profile(carrier_sha256, views))
    if accounting["reference"] != sum(map(len, sections.values())) or accounting["frame"] != 0:
        raise ValueError("appearance cache contains non-reference sections")
    by_slot: dict[tuple[str, int], tuple[int, np.ndarray]] = {}
    seen_ticks: set[tuple[str, int]] = set()
    for name, payload in sections.items():
        role, raw_slot, raw_tick = name.split(".")
        slot, tick = int(raw_slot), int(raw_tick)
        if (role not in roles or not 1 <= slot < views or not 0 < tick < track_frames
                or name != f"{role}.{slot}.{tick}" or (role, slot) in by_slot or (role, tick) in seen_ticks
                or not 0 < len(payload) <= MAX_AVIF_BYTES):
            raise ValueError("invalid appearance reference binding")
        with Image.open(BytesIO(payload), formats=["AVIF"]) as image:
            if (image.size != (k6b.GEN_W, k6b.GEN_H) or image.mode != "RGB"
                    or getattr(image, "n_frames", 1) != 1):
                raise ValueError("invalid AVIF appearance crop")
            by_slot[role, slot] = (tick, np.asarray(image))
        seen_ticks.add((role, tick))
    if len(by_slot) != len(roles) * (views - 1):
        raise ValueError("incomplete appearance cache")
    return {role: [by_slot[role, slot] for slot in range(1, views)] for role in roles}, accounting
