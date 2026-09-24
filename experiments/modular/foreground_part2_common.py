"""Fixed Alcaraz 48-frame source and byte-verified background for part 2."""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.modular.background_arms import _intra_still, _render_panorama
from experiments.modular.background_campaign import CLIPS_ROOT, OUT_DIR
from experiments.modular.measured_ladder import load_sequence
from scripts.background_probe import pack_panorama_side_data


EXPECTED_BYTES = 24_648
EXPECTED_PAYLOAD_BYTES = 22_906
EXPECTED_SIDE_BYTES = 1_742
N_FRAMES = 48


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedAlcaraz:
    """Source, anchor mask and selected background in RGB order.

    ``background_rgb`` is a memory-mapped decode from the external data root.
    ``anchor`` and ``background`` are the saved background-campaign rows.
    A caller may score against ``mask`` but must not expose it to a decoder.
    """

    source_rgb: np.ndarray
    mask: np.ndarray
    background_rgb: np.ndarray
    anchor: dict[str, object]
    background: dict[str, object]
    plate_seconds: float


def load_fixed_alcaraz(out_dir: Path) -> FixedAlcaraz:
    """Load the fixed 48-frame Alcaraz handoff, creating one decode cache.

    Args:
        out_dir: External data directory for a decoded background NPY and
            provenance JSON. Existing cache entries must be validated.

    Returns:
        The source and exact-mask anchor plus RGB decoded background.

    Raises:
        RuntimeError: Native output is empty, bytes differ from 24,648 B,
            encoder path differs from the saved row, or a cache is incomplete.

    The caller relies on the same decoded pixels in the Animate Anyone and
    Alcaraz appearance arms. The plate is never rebuilt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = CLIPS_ROOT / "alcaraz_highlights/scene_000"
    source_rgb, mask = load_sequence(source_dir / "window_48", source_dir / "masks_48.npz", N_FRAMES)
    original = json.loads((OUT_DIR / "alcaraz000.json").read_text())
    anchor = next(row for row in original["rows"] if row["representation"] == "source" and row["qp"] == 46)
    background = next(row for row in original["rows"] if row["representation"] == "registered_panorama" and row["qp"] == 46)
    if int(anchor["total_bytes"]) != 65_149 or int(background["total_bytes"]) != EXPECTED_BYTES:
        raise RuntimeError("saved Alcaraz source/background byte contract changed")

    decoded_path = out_dir / "alcaraz000-qp46-background-rgb.npy"
    payload_path = out_dir / "alcaraz000-qp46-plate.vvc"
    side_path = out_dir / "alcaraz000-qp46-panorama.bin"
    meta_path = out_dir / "alcaraz000-qp46-background.json"
    outputs = (decoded_path, payload_path, side_path, meta_path)
    present = tuple(path.exists() for path in outputs)
    if any(present) and not all(present):
        raise RuntimeError("incomplete Alcaraz decoded background cache")

    if all(present):
        meta = json.loads(meta_path.read_text())
        decoded = np.load(decoded_path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(source_rgb.shape)
        if (
            meta.get("total_bytes") != EXPECTED_BYTES
            or meta.get("tool_path") != background["tool_path"]
            or meta.get("tool_version") != background["tool_version"]
            or payload_path.stat().st_size != EXPECTED_PAYLOAD_BYTES
            or side_path.stat().st_size != EXPECTED_SIDE_BYTES
            or _digest(payload_path) != meta.get("payload_sha256")
            or _digest(side_path) != meta.get("side_sha256")
            or _digest(decoded_path) != meta.get("decoded_sha256")
            or tuple(decoded.shape) != expected_shape
            or decoded.dtype != np.uint8
        ):
            raise RuntimeError("Alcaraz decoded background cache fails saved-byte validation")
        return FixedAlcaraz(source_rgb, mask, decoded, anchor, {**background, **meta}, float(original["plate_build_seconds"]))

    cache_path = OUT_DIR / "cache/alcaraz000-n48.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        plate = cache["plate"]
        homographies = cache["homographies"]
        plate_seconds = float(cache["build_seconds"])
    frame_shape = tuple(int(v) for v in source_rgb.shape[1:3])
    side = pack_panorama_side_data(
        homographies,
        plate_shape=tuple(int(v) for v in plate.shape[:2]),
        frame_shape=frame_shape,
        fps=25.0,
    )
    if len(side) != EXPECTED_SIDE_BYTES:
        raise RuntimeError(f"Alcaraz homography side-data moved: {len(side)} B")
    payload, decoded_plate, tool_path, tool_version, encode_seconds, decode_seconds = _intra_still(plate, 46)
    total = len(payload) + len(side)
    if (
        not payload
        or len(payload) != EXPECTED_PAYLOAD_BYTES
        or total != EXPECTED_BYTES
        or tool_path != background["tool_path"]
        or tool_version != background["tool_version"]
    ):
        raise RuntimeError(
            f"Alcaraz QP46 panorama changed: payload={len(payload)} B, "
            f"side={len(side)} B, total={total} B, encoder={tool_path}"
        )
    started = time.perf_counter()
    decoded = _render_panorama(decoded_plate, side, N_FRAMES, frame_shape)
    render_seconds = time.perf_counter() - started
    if decoded.shape != source_rgb.shape or decoded.dtype != np.uint8:
        raise RuntimeError("Alcaraz panorama decode has wrong RGB shape or dtype")
    payload_path.write_bytes(payload)
    side_path.write_bytes(side)
    meta = {
        "total_bytes": total,
        "encoded_payload_bytes": len(payload),
        "side_data_bytes": len(side),
        "tool_path": tool_path,
        "tool_version": tool_version,
        "reencode_seconds": encode_seconds,
        "redecode_seconds": decode_seconds,
        "rerender_seconds": render_seconds,
        "payload_sha256": _digest(payload_path),
        "side_sha256": _digest(side_path),
        "decoded_path": str(decoded_path),
    }
    tmp_path = decoded_path.with_suffix(".npy.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, decoded, allow_pickle=False)
    os.replace(tmp_path, decoded_path)
    meta["decoded_sha256"] = _digest(decoded_path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    mapped = np.load(decoded_path, mmap_mode="r", allow_pickle=False)
    return FixedAlcaraz(source_rgb, mask, mapped, anchor, {**background, **meta}, plate_seconds)
