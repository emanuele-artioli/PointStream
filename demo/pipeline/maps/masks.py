"""Mask maps for the demo gallery, from `src.segmentation`.

Each run stores the lossless masks (``masks.rle``, what training and the matte
read) and bills a colour-painted SVT-AV1 preview (``masks.av1.mp4``, the shared
CRF recipe) so the gallery's kbps stays comparable with the other maps.

CLI::

    PYTHONPATH=. python -m demo.pipeline.maps.masks --backend yoloe-26x --clip PATH --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from demo.pipeline.maps.av1_crf import (
    AV1_CRF,
    AV1_PRESET,
    AV1_SCALE,
    CLASS_COLORS_BGR,
    pipe_bgr_av1,
)
from demo.pipeline.maps.contract import MapStream, write_sidecar
from src.segmentation import ClipMasks, build, load_domain

DOMAIN = "egocentric"
MAP_NAMES = {"sam31": "sam31_masks"}  # every YOLOE size is the gallery's yoloe_masks


def paint(masks: ClipMasks, index: int) -> np.ndarray:
    """BGR frame on black, later classes on top, so the inspector can key it."""
    canvas = np.zeros((masks.height, masks.width, 3), dtype=np.uint8)
    for name in masks.classes:
        canvas[masks.class_mask(index, name)] = CLASS_COLORS_BGR.get(name, (255, 255, 255))
    return canvas


def write_preview(masks: ClipMasks, dest: Path) -> int:
    """Encode the painted masks; returns the number of non-empty frames."""
    proc = pipe_bgr_av1(masks.width, masks.height, masks.fps, dest)
    assert proc.stdin is not None
    nonempty = 0
    try:
        for index in range(len(masks)):
            nonempty += bool(masks.frames[index])
            proc.stdin.write(paint(masks, index).tobytes())
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        code = proc.wait()
    if code != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(
            f"mask AV1 encode failed ({code}): {stderr.decode(errors='replace')[-2000:]}"
        )
    return nonempty


def run_mask_map(
    backend: str, clip: Path | str, out_dir: Path | str, *, max_frames: int | None = None
) -> MapStream:
    clip, out = Path(clip), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    masks = build(backend).segment(clip, load_domain(DOMAIN), max_frames=max_frames)
    masks.save(out)
    preview = out / "masks.av1.mp4"
    nonempty = write_preview(masks, preview)
    timing = masks.meta.get("timing") or {}
    stream = MapStream(
        map=MAP_NAMES.get(backend, "yoloe_masks"),
        backend=backend,
        payload_path=str(preview),
        payload_bytes=preview.stat().st_size,
        preview_path=str(preview),
        preview_bytes=preview.stat().st_size,
        duration_s=len(masks) / masks.fps,
        n_frames=len(masks),
        fps=masks.fps,
        # Throughput (load excluded) is the one number comparable across backends.
        extract_ms_p50=float(timing.get("ms_per_frame") or 0.0),
        extract_ms_p95=float(timing.get("step_ms_p95") or timing.get("ms_per_frame") or 0.0),
        pack_ms_p50=0.0,
        codec_ms_p50=0.0,
        decode_ms_p50=0.0,
        gpu=str(((masks.meta.get("worker_runtime") or {}).get("gpu") or {}).get("name") or ""),
        extra={
            "payload_format": "av1",
            "payload_schema": "pointstream.maps.mask_av1.v1",
            "lossless_masks": str(out / "masks.rle"),
            "av1_scale": AV1_SCALE,
            "av1_preset": AV1_PRESET,
            "av1_crf": AV1_CRF,
            "classes": list(masks.classes),
            "class_prompts": masks.meta.get("prompts"),
            "model_load_s": timing.get("model_load_s"),
            "latency_excludes_load": True,
            "n_nonempty": nonempty,
            "overlay_path": str(preview),
            "overlay_key": "black",
            "overlay": "av1-black-key",
        },
    )
    write_sidecar(stream, out / "sidecar.json")
    return stream


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", default="yoloe-26x")
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    run_mask_map(args.backend, args.clip, args.out, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
