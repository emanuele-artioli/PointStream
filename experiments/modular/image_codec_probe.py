"""Compare JPEG and WebP on a plate, a crop, and a residual from one real window.

Writes a JSON report. Does not change the ladder.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.components.background.plate import build_plate
from src.components.metrics.psnr import PsnrMetric
from src.pipeline.residual.steered_residual import ActorMaskProcessor

FRAMES = Path(
    "/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/clips/"
    "federer_djokovic/scene_007/window_48"
)
MASKS = Path(
    "/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/clips/"
    "federer_djokovic/scene_007/masks_48.npz"
)
QUALITIES = (40, 60, 80)
N_FRAMES = 4


def _encode(image_bgr: np.ndarray, codec: str, quality: int) -> tuple[bytes, np.ndarray]:
    if codec == "jpeg":
        ext, params = ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif codec == "webp":
        ext, params = ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    else:
        raise ValueError(codec)
    ok, buf = cv2.imencode(ext, image_bgr, params)
    if not ok:
        raise RuntimeError(f"{codec} q{quality} failed")
    payload = buf.tobytes()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image_bgr.shape:
        raise RuntimeError(f"{codec} q{quality} decode shape {None if decoded is None else decoded.shape}")
    return payload, decoded


def _psnr(reference: np.ndarray, decoded: np.ndarray) -> float:
    return float(PsnrMetric().score(reference[None], decoded[None]))


def _signals() -> dict[str, np.ndarray]:
    paths = sorted(FRAMES.glob("frame_*.png"))[:N_FRAMES]
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unreadable {path}")
        frames.append(image)
    stack = np.stack(frames)
    masks = np.load(MASKS)["masks"][:N_FRAMES]
    if masks.shape[:3] != stack.shape[:3]:
        raise RuntimeError(f"mask {masks.shape} vs frames {stack.shape}")
    plate, _poses = build_plate(stack, masks, register=False)
    plate_small = cv2.resize(plate, (plate.shape[1] // 2, plate.shape[0] // 2), interpolation=cv2.INTER_AREA)
    bbox = ActorMaskProcessor().compute_tight_bbox(masks[0] > 0, pad=8, even_align=True)
    if bbox is None:
        raise RuntimeError("empty foreground on frame 0")
    y0, y1, x0, x1 = bbox
    crop = stack[0, y0:y1, x0:x1]
    residual = np.clip(stack[1].astype(np.int16) - plate.astype(np.int16) + 128, 0, 255).astype(np.uint8)
    residual[masks[1] > 0] = 128
    return {"plate": plate_small, "crop": crop, "residual": residual}


def _dominates(points_a: list[dict], points_b: list[dict]) -> bool:
    """True when every B point is matched or beaten by some A point, and one is strict."""
    strict = False
    for other in points_b:
        match = False
        for ours in points_a:
            bytes_ok = ours["bytes"] <= other["bytes"]
            psnr_ok = ours["psnr"] >= other["psnr"] - 1e-6
            if bytes_ok and psnr_ok:
                match = True
                if ours["bytes"] < other["bytes"] or ours["psnr"] > other["psnr"] + 0.05:
                    strict = True
                break
        if not match:
            return False
    return strict


def main() -> None:
    started = time.perf_counter()
    signals = _signals()
    rows: list[dict] = []
    for name, image in signals.items():
        for codec in ("jpeg", "webp"):
            for quality in QUALITIES:
                t0 = time.perf_counter()
                payload, decoded = _encode(image, codec, quality)
                rows.append(
                    {
                        "signal": name,
                        "codec": codec,
                        "quality": quality,
                        "bytes": len(payload),
                        "psnr": round(_psnr(image, decoded), 4),
                        "height": int(image.shape[0]),
                        "width": int(image.shape[1]),
                        "seconds": round(time.perf_counter() - t0, 3),
                    }
                )
                print(rows[-1])

    decisions = {}
    for name in signals:
        jpeg = [row for row in rows if row["signal"] == name and row["codec"] == "jpeg"]
        webp = [row for row in rows if row["signal"] == name and row["codec"] == "webp"]
        if _dominates(webp, jpeg):
            winner = "webp"
        elif _dominates(jpeg, webp):
            winner = "jpeg"
        else:
            winner = "split"
        decisions[name] = winner

    same = len(set(decisions.values())) == 1 and "split" not in decisions.values()
    report = {
        "source": "federer_djokovic/scene_007/window_48",
        "n_frames": N_FRAMES,
        "rows": rows,
        "per_signal": decisions,
        "one_codec_for_all_signals": same,
        "chosen": next(iter(decisions.values())) if same else decisions,
        "seconds": round(time.perf_counter() - started, 3),
    }
    out = Path("/home/itec/emanuele/pointstream-data/outputs/image-codec-probe/federer007.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"per_signal": decisions, "one_codec": same, "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
