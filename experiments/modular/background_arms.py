"""Step 1 of the 23 September campaign: four background arms.

The 11 September probe is not reused. Its mask came from the long-scene
loader (two objects, mean foreground fraction 0.00937). This step uses
``masks_48.npz``, the mask on the weighted VVC and AV1 anchors.

Encodes are the four arms at VVC QP 46 and QP 40.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from experiments.modular.image_codec_probe import FRAMES, MASKS
from experiments.modular.measured_ladder import _roundtrip_clip, load_sequence, score_regions
from scripts.background_probe import (
    build_common_cleaned_stack,
    charge_side_data,
    pack_panorama_side_data,
    pack_still_or_video_side_data,
    unpack_panorama_side_data,
    warp_plate_to_frame,
)
from src.components.background.sidecar import IntraCodecSidecar
from src.components.background.still import select_best_background_frame
from src.components.codec.tools import resolve_encoder

OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/background-arms/federer007.json"
)
N_FRAMES = 48
QPS = (46, 40)
PRESET = "faster"
CODEC = "vvc"
ANCHORS = {
    "vvc_qp46": {
        "bytes": 112_295,
        "psnr_bg": 31.362980885416793,
        "psnr_weighted": 24.58878360131408,
    },
    "av1_qp54": {
        "bytes": 316_061,
        "psnr_bg": 36.28916650699504,
        "psnr_weighted": 30.067556734663093,
    },
}


def _intra_still(image_rgb: np.ndarray, qp: int) -> tuple[bytes, np.ndarray, str, str, float, float]:
    coder = IntraCodecSidecar(CODEC, qp=qp, preset=PRESET)
    path, version = coder.probe_encoder()
    started = time.perf_counter()
    payload = coder.encode(np.ascontiguousarray(image_rgb[:, :, ::-1]))
    encode_s = time.perf_counter() - started
    started = time.perf_counter()
    decoded_bgr = coder.decode(payload)
    decode_s = time.perf_counter() - started
    decoded = decoded_bgr[:, :, ::-1]
    canvas = np.zeros(image_rgb.shape, dtype=np.uint8)
    height = min(int(decoded.shape[0]), int(canvas.shape[0]))
    width = min(int(decoded.shape[1]), int(canvas.shape[1]))
    canvas[:height, :width] = decoded[:height, :width]
    return payload, canvas, path, version, encode_s, decode_s


def _repeat(image_rgb: np.ndarray, n_frames: int, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    frame = np.zeros((height, width, image_rgb.shape[2]), dtype=np.uint8)
    h = min(height, int(image_rgb.shape[0]))
    w = min(width, int(image_rgb.shape[1]))
    frame[:h, :w] = image_rgb[:h, :w]
    return np.broadcast_to(frame, (n_frames, height, width, 3)).copy()


def _score(reference_rgb: np.ndarray, rendered_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    overall, fg, bg, weighted = score_regions(
        reference_rgb,
        rendered_rgb,
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


def _budgets(total: int) -> dict[str, int]:
    return {name: int(anchor["bytes"]) - total for name, anchor in ANCHORS.items()}


def _row(
    name: str,
    qp: int,
    payload: bytes,
    side_bytes: int,
    rendered_rgb: np.ndarray,
    frames_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    tool_path: str,
    tool_version: str,
    encode_s: float,
    decode_s: float,
    extra: dict[str, object],
) -> dict[str, object]:
    total = len(payload) + side_bytes
    row: dict[str, object] = {
        "representation": name,
        "qp": qp,
        "codec": CODEC,
        "preset": PRESET,
        "tool_path": tool_path,
        "tool_version": tool_version,
        "encoded_payload_bytes": len(payload),
        "side_data_bytes": side_bytes,
        "total_bytes": total,
        "encode_seconds": round(encode_s, 3),
        "decode_seconds": round(decode_s, 3),
        "foreground_budget": _budgets(total),
    }
    row.update(_score(frames_rgb, rendered_rgb, mask))
    row.update(extra)
    return row


def choose_background(rows: list[dict[str, object]], anchor_bytes: int) -> dict[str, object]:
    """Highest background PSNR among arms strictly under ``anchor_bytes``."""
    eligible = [row for row in rows if int(row["total_bytes"]) < anchor_bytes]
    if not eligible:
        return {
            "fits": False,
            "foreground_budget_bytes": 0,
            "reason": "no arm is under the anchor",
        }
    chosen = max(eligible, key=lambda row: float(row["psnr_bg"] or 0.0))
    budget = anchor_bytes - int(chosen["total_bytes"])
    return {
        "fits": budget >= 8_000,
        "representation": chosen["representation"],
        "qp": chosen["qp"],
        "total_bytes": chosen["total_bytes"],
        "psnr_bg": chosen["psnr_bg"],
        "psnr_weighted": chosen["psnr_weighted"],
        "foreground_budget_bytes": budget,
        "reason": (
            "highest background PSNR among arms under the anchor"
            if budget >= 8_000
            else "under the anchor but the remaining budget is below 8 kB"
        ),
    }


def _render_panorama(
    decoded_plate_rgb: np.ndarray,
    side_bytes: bytes,
    n_frames: int,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    homographies, _plate_shape, _frame_shape, _fps = unpack_panorama_side_data(side_bytes)
    height, width = frame_shape
    return np.stack(
        [
            warp_plate_to_frame(decoded_plate_rgb, homographies[index], height=height, width=width)
            for index in range(n_frames)
        ],
        axis=0,
    )


def _write(payload: dict[str, object]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    wall = time.perf_counter()
    frames_rgb, mask = load_sequence(FRAMES, MASKS, N_FRAMES)
    print(
        f"loaded {tuple(frames_rgb.shape)} foreground_fraction={float(mask.mean()):.6f}",
        flush=True,
    )
    print("building plate-inpainted stack", flush=True)
    cleaned_rgb, plate_rgb, homographies, prep = build_common_cleaned_stack(
        frames_rgb,
        mask,
        removal="on",
        register=True,
    )
    best_index, best_mse = select_best_background_frame(frames_rgb, mask)
    print(f"best_frame index={best_index} mse={best_mse:.4f} prep={prep.get('build_seconds')}", flush=True)
    height, width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    frame_shape = (height, width)
    still_side = pack_still_or_video_side_data(frame_shape=frame_shape, n_frames=N_FRAMES, fps=25.0)
    best_side = still_side + int(best_index).to_bytes(2, "little")
    pano_side = pack_panorama_side_data(
        homographies,
        plate_shape=(int(plate_rgb.shape[0]), int(plate_rgb.shape[1])),
        frame_shape=frame_shape,
        fps=25.0,
    )
    video_side = still_side
    encoder = resolve_encoder(CODEC)
    rows: list[dict[str, object]] = []
    document: dict[str, object] = {
        "source": str(FRAMES),
        "mask": str(MASKS),
        "n_frames": N_FRAMES,
        "foreground_fraction": float(mask.mean()),
        "reuse": {
            "sept11_report": (
                "/home/itec/emanuele/pointstream-data/outputs/"
                "development-recovery/wave2-background-probe/probe_report.json"
            ),
            "accepted": False,
            "reason": (
                "The 11 September probe used load_long_scene_clip masks "
                "(two objects, foreground fraction 0.00937). This step uses "
                "masks_48.npz so the budget matches the weighted anchors."
            ),
        },
        "best_frame": {"index": best_index, "background_mse": best_mse},
        "plate_shape": [int(plate_rgb.shape[0]), int(plate_rgb.shape[1])],
        "prep": {key: prep[key] for key in ("build_seconds", "plate_resolution", "total_inpaint_holes", "inpaint_frames") if key in prep},
        "encoder_resolved": {"path": encoder.path, "version": encoder.version},
        "anchors": ANCHORS,
        "rows": rows,
    }

    for qp in QPS:
        for name, image in (
            ("still_frame0", cleaned_rgb[0]),
            ("best_frame", cleaned_rgb[best_index]),
        ):
            print(f"{name} qp {qp}", flush=True)
            payload, decoded, path, version, enc_s, dec_s = _intra_still(image, qp)
            side = still_side if name == "still_frame0" else best_side
            extra: dict[str, object] = {"frame_index": 0 if name == "still_frame0" else best_index}
            if name == "best_frame":
                extra["selection_mse"] = best_mse
            rows.append(
                _row(
                    name,
                    qp,
                    payload,
                    len(side),
                    _repeat(decoded, N_FRAMES, frame_shape),
                    frames_rgb,
                    mask,
                    tool_path=path,
                    tool_version=version,
                    encode_s=enc_s,
                    decode_s=dec_s,
                    extra=extra,
                )
            )
            print(rows[-1], flush=True)
            _write(document)

        print(f"registered_panorama qp {qp}", flush=True)
        payload, decoded_plate, path, version, enc_s, dec_s = _intra_still(plate_rgb, qp)
        started = time.perf_counter()
        rendered = _render_panorama(decoded_plate, pano_side, N_FRAMES, frame_shape)
        dec_s += time.perf_counter() - started
        detail = charge_side_data(
            "registered_panorama",
            N_FRAMES,
            plate_shape=(int(plate_rgb.shape[0]), int(plate_rgb.shape[1])),
            frame_shape=frame_shape,
            homographies=homographies,
        )
        rows.append(
            _row(
                "registered_panorama",
                qp,
                payload,
                int(detail["total_side_data_bytes"]),
                rendered,
                frames_rgb,
                mask,
                tool_path=path,
                tool_version=version,
                encode_s=enc_s,
                decode_s=dec_s,
                extra={"plate_shape": [int(plate_rgb.shape[0]), int(plate_rgb.shape[1])]},
            )
        )
        print(rows[-1], flush=True)
        _write(document)

        print(f"cleaned_video qp {qp}", flush=True)
        started = time.perf_counter()
        payload, decoded_bgr = _roundtrip_clip(cleaned_rgb, codec=CODEC, qp=qp, preset=PRESET)
        encode_s = time.perf_counter() - started
        decoded_rgb = decoded_bgr[..., ::-1]
        if decoded_rgb.shape[1] != height or decoded_rgb.shape[2] != width:
            canvas = np.zeros_like(frames_rgb)
            h = min(height, int(decoded_rgb.shape[1]))
            w = min(width, int(decoded_rgb.shape[2]))
            canvas[:, :h, :w] = decoded_rgb[:, :h, :w]
            decoded_rgb = canvas
        rows.append(
            _row(
                "cleaned_video",
                qp,
                payload,
                len(video_side),
                decoded_rgb[:N_FRAMES],
                frames_rgb,
                mask,
                tool_path=encoder.path,
                tool_version=encoder.version,
                encode_s=encode_s,
                decode_s=0.0,
                extra={},
            )
        )
        print(rows[-1], flush=True)
        _write(document)

    document["choice"] = {
        name: choose_background(rows, int(anchor["bytes"])) for name, anchor in ANCHORS.items()
    }
    document["wall_seconds"] = round(time.perf_counter() - wall, 3)
    _write(document)
    print(f"wrote {OUT}", flush=True)
    print(document["choice"], flush=True)


if __name__ == "__main__":
    main()
