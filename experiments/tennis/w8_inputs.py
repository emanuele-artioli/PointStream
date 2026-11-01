"""Decode paid W7 conditioning packets into source-free generator inputs.

Golden prepared arrays are accepted only by the separate validation function.
This module does not inspect video, target masks, or saved generated crops.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_reference

MAX_PACKET = 64 * 1024 * 1024


def archive_members(path: Path, names: list[str]) -> dict[str, bytes]:
    """Read only exact regular members, rejecting duplicate names and unsafe paths."""
    wanted = set(names)
    if len(wanted) != len(names) or not wanted:
        raise ValueError("Missing or duplicate requested members")
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
            raise ValueError("Unsafe archive member name")
    found: dict[str, bytes] = {}
    with tarfile.open(path, "r:*" ) as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            if member.name in found or not member.isfile() or not 0 < member.size <= MAX_PACKET:
                raise ValueError("Duplicate, non-file, or oversized member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable member")
            data = stream.read(MAX_PACKET + 1)
            if len(data) != member.size:
                raise ValueError("Truncated archive member")
            found[member.name] = data
    if found.keys() != wanted:
        raise ValueError("Missing archive member")
    return found


def decode_inputs(keypoints: bytes, references: dict[str, bytes], window: str, judged_bitmap: bytes) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Build only generator arrays and metadata from the selected packet bytes.

    Raises if this implementation cannot honor the paid W7 profile. `judged` is
    supplied by a separately paid, identity-bound packet: the KPS registered
    bitmap denotes scored frames, not generator judged frames. It must still be
    checked against the golden prepared data before reuse.
    """
    from experiments.tennis import k4, k6b

    if keypoints[:4] != b"KPS2":
        raise ValueError("Expected KPS2")
    meta, units = decode_keypoints(keypoints)
    expected = {
        "window": window, "native_frames": 240, "start_tick": 0,
        "model_profile": "C-q4-guidance2-DDIM30", "common_base": "tennis-base@fit-100",
        "generation_size": [k6b.GEN_W, k6b.GEN_H],
        "crop_rule": [k6b.CROP_MARGIN, k6b.SIZE_PERCENTILE, k6b.CENTRE_SMOOTH],
        "context": [24, 4], "seed_rule": "sha256(unit_name)[:7] then seed*1009+run",
    }
    if k4.KP_CONF != 0.5 or any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported renderer or packet contract")
    fps = meta.get("fps")
    if not isinstance(fps, list) or len(fps) != 2 or not all(type(x) is int and x > 0 for x in fps) or meta.get("time_base") != [fps[1], fps[0]]:
        raise ValueError("Unsupported timing")
    try:
        bitmap = bytes.fromhex(meta["registered_frame_bitmap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid registered frame bitmap") from exc
    if len(bitmap) != 30:
        raise ValueError("Invalid registered frame bitmap length")
    if not isinstance(judged_bitmap, bytes) or len(judged_bitmap) != 30:
        raise ValueError("A separately paid 240-frame judged bitmap is required")
    judged = np.unpackbits(np.frombuffer(judged_bitmap, dtype=np.uint8)).astype(bool)
    registered = np.unpackbits(np.frombuffer(bitmap, dtype=np.uint8)).astype(bool)
    if np.any(judged & ~registered):
        raise ValueError("Judged frames are outside the registered frame set")
    names = [unit["name"] for unit in units]
    if len(names) != 2 or len(set(names)) != 2 or set(names) != set(references):
        raise ValueError("Reference/unit binding mismatch")
    order = meta.get("compositing_order")
    if (not isinstance(order, list) or len(order) != 2 or
            any(not isinstance(player, str) or not player for player in order) or
            len(set(order)) != 2 or {k6b.unit_name(window, player) for player in order} != set(names)):
        raise ValueError("Window, player and compositing order binding mismatch")
    out: dict[str, dict[str, np.ndarray]] = {}
    for unit in units:
        name = unit["name"]
        if not isinstance(name, str) or not name or any(c in name for c in "/\\@"):
            raise ValueError("Unsafe unit name")
        refmeta, image = decode_reference(references[name])
        if (refmeta.get("unit") != name or refmeta.get("codec") != ".png" or
                refmeta.get("quality") is not None or refmeta.get("color") != "RGB8" or
                refmeta.get("pixels") != "full_prepared_conditioning_crop" or
                type(refmeta.get("reference_frame")) is not int or not 0 <= refmeta["reference_frame"] < 240):
            raise ValueError("Reference identity or lossless profile mismatch")
        box = np.asarray(refmeta.get("reference_box"), dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
            raise ValueError("Invalid reference geometry")
        if image.shape != (k6b.GEN_H, k6b.GEN_W, 3) or image.dtype != np.uint8:
            raise ValueError("Invalid reference pixels")
        times = np.asarray(unit["t"], dtype=np.int64)
        kp = np.asarray(unit["kp"], dtype=np.float64)
        if not len(times) or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] >= 240 or kp.shape != (len(times), 17, 3) or not np.isfinite(kp).all():
            raise ValueError("Invalid decoded track")
        boxes: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        run_ids: list[int] = []
        cursor = 0
        for run_id, seq in enumerate(k6b.runs(times.tolist())):
            ks = list(kp[cursor:cursor + len(seq)])
            runbox = k6b.run_boxes(ks)
            boxes.extend(runbox)
            poses.extend(k6b.pose_map(k6b.to_crop(k, b)) for k, b in zip(ks, runbox))
            run_ids.extend([run_id] * len(seq))
            cursor += len(seq)
        out[name] = {"reference": image, "reference_box": box, "t": times,
                     "run": np.asarray(run_ids, dtype=np.int64), "box": np.asarray(boxes, dtype=np.float64),
                     "pose": np.stack(poses), "judged": judged[times]}
    bindings = {"window": window, "profile": meta["model_profile"], "context": meta["context"],
                "seeds": {name: int(hashlib.sha256(name.encode()).hexdigest()[:7], 16) for name in names},
                "reference_frames": {name: decode_reference(references[name])[0]["reference_frame"] for name in names},
                "compositing_order": meta.get("compositing_order"), "fps": fps}
    return out, bindings


def validate_golden(decoded: dict[str, dict[str, np.ndarray]], golden: dict[str, Path], keypoints: bytes) -> None:
    """Strict independent check against prepared arrays; never returns golden data."""
    from experiments.tennis.k6b_rate_transport import native_equivalence

    if set(decoded) != set(golden):
        raise ValueError("Golden unit mismatch")
    for name, fields in decoded.items():
        with np.load(golden[name], allow_pickle=False) as source:
            for key, value in fields.items():
                if key not in source or not np.array_equal(value, source[key]):
                    raise ValueError(f"Golden mismatch: {name}:{key}")
    _, tracks = decode_keypoints(keypoints)
    if {track["name"] for track in tracks} != set(decoded):
        raise ValueError("Golden keypoint binding mismatch")
    native_equivalence(tracks, golden)
