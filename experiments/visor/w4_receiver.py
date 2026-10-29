"""W4: the receiver's own foreground mask, uncorrected, in place of F̂ in H5p 4a's composites.

The receiver tracks the foreground with SAM 3.1 from a first-frame union mask sent intra-coded (the seed), with no
later prompt and no correction stream (H5p 4d(b)'s tracker, ``experiments.visor.h5_decoder_mask``). Its masks are on
the 1/s grid (majority-reduced, as the 4d(b) jobs saved them). The MASK segment of the container holds two
configuration bytes (version, scale) and the seed; nothing else is billed for the mask.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.pointstream import mask as mk
from src.pointstream import shape

VERSION = 1


def pack_payload(seed: bytes, scale: int) -> bytes:
    """The MASK segment: version u8, scale u8, then the intra-coded seed (``shape.encode`` of one frame)."""
    return bytes([VERSION, scale]) + seed


def unpack_payload(payload: bytes) -> tuple[int, bytes]:
    """The scale and the seed of a receiver-mask MASK segment."""
    if len(payload) < 3 or payload[0] != VERSION:
        raise ValueError("not a receiver-mask payload")
    return payload[1], payload[2:]


def accuracy(truth: np.ndarray, shown: np.ndarray) -> dict[str, float]:
    """IoU (pooled over the clip and the per-frame mean) and the missed and extra shares of F, for boolean maps."""
    inter = (truth & shown).sum(axis=(1, 2)).astype(np.float64)
    union = (truth | shown).sum(axis=(1, 2)).astype(np.float64)
    frame = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    size = max(int(truth.sum()), 1)
    return {"iou": float(inter.sum() / max(union.sum(), 1.0)), "iou_frame_mean": float(frame.mean()),
            "iou_frame_min": float(frame.min()), "missed_share": float((truth & ~shown).sum() / size),
            "extra_share": float((shown & ~truth).sum() / size)}


def receiver_classes(classes: np.ndarray, scale: int, masks: Path, seed: Path,
                     fps: float) -> tuple[np.ndarray, bytes, dict[str, Any]]:
    """The client's F̂ from the saved tracker output: the class maps at full size (the prediction upsampled as H5p
    4a's F̂ is), the MASK payload (seed plus two configuration bytes) and its record against the true F."""
    frames, height, width = classes.shape
    predicted = np.load(masks)["predicted"]
    if len(predicted) < frames:
        raise ValueError(f"{masks}: {len(predicted)} predicted frames for {frames}")
    predicted = (predicted[:frames] > 0).astype(np.uint8)
    seed_bytes = seed.read_bytes()
    payload = pack_payload(seed_bytes, scale)
    first = shape.decode(seed_bytes)[0]
    reduced = (mk.reduce(classes[:1], scale)[0] > 0).astype(np.uint8)
    shown = mk.upsample(predicted, (width, height))
    truth, got = classes > 0, shown > 0
    record = {"receiver": True, "scale": scale, "bytes": len(payload), "seed_bytes": len(seed_bytes),
              "kbps": len(payload) * 8 / (frames / float(fps)) / 1000,
              "seed_is_first_union_mask": bool(first.shape == reduced.shape and np.array_equal(first, reduced)),
              "prediction_file_sha256": hashlib.sha256(masks.read_bytes()).hexdigest(),
              "seed_file_sha256": hashlib.sha256(seed_bytes).hexdigest(),
              "prediction_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
              **accuracy(truth, got)}
    return shown, payload, record
