"""Measure the static-plate registration control on the real 48-frame window.

The modular ladder deliberately used ``register=False``. This probe keeps that
control, then reconstructs a registered plate through its per-frame maps. The
maps are charged as float32 homographies (9 values per frame) so registration
cannot receive free motion information.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import numpy as np

from experiments.modular.image_codec_probe import FRAMES, MASKS
from experiments.modular.measured_ladder import (
    PLATE_CODEC,
    PLATE_QP,
    _plate_roundtrip,
    _rgb_to_bgr,
    _yuv420_to_rgb,
    load_sequence,
    score_regions,
)
from src.components.background.plate import build_plate
from src.components.codec.y4m import read

OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/"
    "background-registration/federer007.json"
)
ANCHOR = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/"
    "measured-tennis/anchors/short/vvc/decoded_qp46.y4m"
)
N_FRAMES = 48


def _reconstruct(
    plate_bgr: np.ndarray,
    maps: tuple[tuple[float, ...], ...],
    *,
    registered: bool,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    height, width = frame_shape
    frames: list[np.ndarray] = []
    for packed in maps:
        if not registered:
            frame = plate_bgr[:height, :width]
        else:
            matrix = np.asarray(packed, dtype=np.float64).reshape(3, 3)
            frame = cv2.warpPerspective(
                plate_bgr,
                np.linalg.inv(matrix),
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        frames.append(frame)
    return np.stack(frames)


def _score(
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


def main() -> None:
    frames_rgb, mask = load_sequence(FRAMES, MASKS, N_FRAMES)
    frames_bgr = _rgb_to_bgr(frames_rgb)
    height, width = frames_rgb.shape[1:3]
    map_bytes = N_FRAMES * 9 * 4
    rows: list[dict[str, object]] = []

    for registered in (False, True):
        plate, maps = build_plate(
            frames_bgr,
            masks=mask,
            register=registered,
        )
        payload, decoded = _plate_roundtrip(plate)
        reconstruction = _reconstruct(
            decoded,
            maps,
            registered=registered,
            frame_shape=(height, width),
        )
        row: dict[str, object] = {
            "arm": "registered_plate" if registered else "unregistered_plate_control",
            "registered": registered,
            "plate_codec": PLATE_CODEC,
            "plate_qp": PLATE_QP,
            "plate_shape": list(plate.shape),
            "plate_bytes": len(payload),
            "map_bytes": map_bytes if registered else 0,
            "total_bytes": len(payload) + (map_bytes if registered else 0),
        }
        row.update(_score(frames_rgb, reconstruction, mask))
        rows.append(row)
        print(row, flush=True)

    anchor_row: dict[str, object] | None = None
    if ANCHOR.is_file():
        decoded_anchor = read(ANCHOR)
        if decoded_anchor.chroma is not None:
            anchor_rgb = _yuv420_to_rgb(decoded_anchor.luma, decoded_anchor.chroma)
            anchor_row = {
                "arm": "vvc_qp46_anchor",
                "bytes": ANCHOR.with_name("vvc_qp46.vvc").stat().st_size,
            }
            anchor_row.update(_score(frames_rgb, anchor_rgb[..., ::-1], mask))

    result = {
        "source": str(FRAMES),
        "n_frames": N_FRAMES,
        "map_encoding": "9 float32 homography values per frame",
        "metric": {"foreground_weight": 0.70, "background_weight": 0.30},
        "rows": rows,
        "anchor": anchor_row,
        "interpretation": (
            "register=False is the existing quality control. A registered plate "
            "must be charged for its per-frame camera maps; it is not comparable "
            "as a free static image."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
