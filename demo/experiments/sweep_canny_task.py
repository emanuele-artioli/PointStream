"""Sweep luma-Canny knobs against YOLOE and DW-Pose boundaries.

Reads the maps-gallery clips and the already extracted YOLOE / pose previews.
Does not download weights. Prints a JSON report. Payload bits are the native
packer (packbits / XOR / sparse / chain), never the preview.

Example::

    PYTHONPATH=. python -m demo.experiments.sweep_canny_task \\
        --clips /home/itec/emanuele/tmp/maps-clips \\
        --maps /home/itec/emanuele/tmp/maps-out \\
        --out /tmp/canny_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from demo.evaluation.wire_accounting import payload_kbps
from demo.pipeline.maps.canny import codec_name, pack_canny_payload, unpack_canny_payload
from demo.pipeline.maps.canny_quality import (
    boundary_band,
    extract_tuned_frame,
    mean_scores,
    score_edges,
    stroke_band,
    union_bands,
)

# (height, lo, hi, blur, fullres_then_down)
SWEEP = []
for _height in (240, 540):
    for _lo, _hi in ((50, 150), (80, 160), (100, 200), (120, 240), (150, 250)):
        for _blur in (0.0, 0.8, 1.6):
            SWEEP.append((_height, _lo, _hi, _blur, False))
BASELINE = (240, 50, 150, 0.0, True)

RECALL_FLOOR_RATIO = 0.92


def _load_gray_alpha(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return image
    return image[:, :, 3]


def _pngs(directory: Path) -> list[Path]:
    frames = sorted(directory.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no pngs in {directory}")
    return frames


def load_targets(maps_root: Path, clip: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Filled YOLOE masks and pose strokes at source resolution.

    Contours are taken after the resize, so a 1px outline is not deleted by
    downsampling.
    """
    yolo_paths = _pngs(maps_root / clip / "yoloe_masks" / "preview_masks")
    pose_paths = _pngs(maps_root / clip / "pose" / "preview")
    n = min(len(yolo_paths), len(pose_paths))
    targets = []
    for index in range(n):
        filled = (_load_gray_alpha(yolo_paths[index]) > 0).astype(np.uint8)
        strokes = (_load_gray_alpha(pose_paths[index]) > 0).astype(np.uint8)
        targets.append((filled, strokes))
    return targets


def _resize_keep(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Downsample a binary mask without dropping thin strokes."""
    if mask.shape == (height, width):
        return (mask > 0).astype(np.uint8)
    scaled = cv2.resize((mask > 0).astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_AREA)
    return (scaled > 0).astype(np.uint8)


def boundary_at(filled: np.ndarray, strokes: np.ndarray, height: int, width: int) -> np.ndarray:
    return union_bands(
        boundary_band(_resize_keep(filled, height, width), radius=1),
        stroke_band(_resize_keep(strokes, height, width), radius=1),
    )


def read_clip(path: Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) < n_frames:
        raise RuntimeError(f"{path} yielded {len(frames)} frames, wanted {n_frames}")
    return frames


def _eval_config(frames, targets, cfg, *, level: int, include_chain: bool) -> dict:
    height, lo, hi, blur, fullres = cfg
    edges = []
    rows = []
    prev = None
    for frame, (filled, strokes) in zip(frames, targets):
        edge = extract_tuned_frame(
            frame,
            lo=lo,
            hi=hi,
            target_height=height,
            blur_sigma=blur,
            fullres_then_down=fullres,
        )
        band = boundary_at(filled, strokes, edge.shape[0], edge.shape[1])
        rows.append(score_edges(edge, band, prev))
        edges.append(edge)
        prev = edge
    payload = pack_canny_payload(edges, level=level, include_chain=include_chain)
    flags = int.from_bytes(payload[20:24], "little")
    if include_chain:
        restored = unpack_canny_payload(payload)
        for got, want in zip(restored, edges):
            if not np.array_equal(got, want):
                raise RuntimeError("canny payload did not round-trip")
    stats = mean_scores(rows)
    n = len(edges)
    duration_s = n / 30.0
    stats.update(
        {
            "height": height,
            "lo": lo,
            "hi": hi,
            "blur": blur,
            "fullres_then_down": fullres,
            "payload_bytes": len(payload),
            "payload_kbps": payload_kbps(len(payload), duration_s),
            "codec": codec_name(flags),
            "pack_h": int(edges[0].shape[0]),
            "pack_w": int(edges[0].shape[1]),
        }
    )
    return stats


def _key(row: dict) -> tuple:
    return (
        int(row["height"]),
        int(row["lo"]),
        int(row["hi"]),
        float(row["blur"]),
        bool(row["fullres_then_down"]),
    )


def choose_winner(per_clip: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    """Cheapest config that holds recall near the dense baseline on every clip."""
    clips = sorted(per_clip)
    baselines = {}
    for clip in clips:
        base = next(row for row in per_clip[clip] if _key(row) == BASELINE)
        baselines[clip] = base
    by_key: dict[tuple, dict] = {}
    for clip in clips:
        floor = baselines[clip]["recall"] * RECALL_FLOOR_RATIO
        for row in per_clip[clip]:
            key = _key(row)
            slot = by_key.setdefault(key, {"clips": {}})
            slot["clips"][clip] = row
            slot["key"] = {
                "height": row["height"],
                "lo": row["lo"],
                "hi": row["hi"],
                "blur": row["blur"],
                "fullres_then_down": row["fullres_then_down"],
            }
    ranked = []
    for slot in by_key.values():
        if len(slot["clips"]) != len(clips):
            continue
        admissible = all(
            slot["clips"][clip]["recall"] + 1e-9 >= baselines[clip]["recall"] * RECALL_FLOOR_RATIO
            for clip in clips
        )
        mean_kbps = float(np.mean([slot["clips"][c]["payload_kbps"] for c in clips]))
        mean_recall = float(np.mean([slot["clips"][c]["recall"] for c in clips]))
        mean_quality = float(np.mean([slot["clips"][c]["quality"] for c in clips]))
        ranked.append(
            {
                **slot["key"],
                "admissible": admissible,
                "mean_kbps": mean_kbps,
                "mean_recall": mean_recall,
                "mean_quality": mean_quality,
                "per_clip_kbps": {c: slot["clips"][c]["payload_kbps"] for c in clips},
                "per_clip_recall": {c: slot["clips"][c]["recall"] for c in clips},
                "per_clip_codec": {c: slot["clips"][c]["codec"] for c in clips},
            }
        )
    admissible = [row for row in ranked if row["admissible"]]
    pool = admissible or ranked
    pool.sort(key=lambda row: (row["mean_kbps"], -row["mean_recall"]))
    winner = pool[0]
    winner["baseline_recall"] = {c: baselines[c]["recall"] for c in clips}
    winner["baseline_kbps"] = {c: baselines[c]["payload_kbps"] for c in clips}
    winner["used_fallback"] = not bool(admissible)
    return winner, pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--level", type=int, default=3, help="zstd level for the sweep ranking")
    parser.add_argument("--chain", action="store_true", help="also try the chain coder during the sweep")
    args = parser.parse_args(argv)

    configs = [BASELINE, *SWEEP]
    # Baseline is also inside SWEEP when fullres is false; keep it explicit and unique.
    seen = set()
    unique = []
    for cfg in configs:
        if cfg in seen:
            continue
        seen.add(cfg)
        unique.append(cfg)

    per_clip: dict[str, list[dict]] = {}
    for clip in ("clip_01", "clip_02", "clip_03"):
        targets = load_targets(args.maps, clip)
        frames = read_clip(args.clips / f"{clip}.mp4", len(targets))
        rows = []
        print(f"{clip}: {len(frames)} frames, {len(unique)} configs", flush=True)
        for index, cfg in enumerate(unique, start=1):
            row = _eval_config(frames, targets, cfg, level=args.level, include_chain=args.chain)
            rows.append(row)
            print(
                f"  {index:02d} h={cfg[0]} {cfg[1]}/{cfg[2]} blur={cfg[3]} "
                f"full={int(cfg[4])} recall={row['recall']:.3f} waste={row['waste']:.3f} "
                f"flicker={row['flicker']:.3f} {row['payload_kbps']:.1f}kbps {row['codec']}",
                flush=True,
            )
        per_clip[clip] = rows

    winner, pool = choose_winner(per_clip)
    # Re-pack the shortlist at the gallery zstd level, including the chain coder.
    shortlist = []
    seen_keys = set()
    for row in pool[:5]:
        key = (
            int(row["height"]),
            int(row["lo"]),
            int(row["hi"]),
            float(row["blur"]),
            bool(row["fullres_then_down"]),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        shortlist.append(key)
    base_key = BASELINE
    if base_key not in seen_keys:
        shortlist.append(base_key)
    final_rows: dict[str, list[dict]] = {}
    for clip in ("clip_01", "clip_02", "clip_03"):
        targets = load_targets(args.maps, clip)
        frames = read_clip(args.clips / f"{clip}.mp4", len(targets))
        final_rows[clip] = [
            _eval_config(frames, targets, cfg, level=19, include_chain=True) for cfg in shortlist
        ]
        print(f"{clip} final zstd19+chain", flush=True)
        for row in final_rows[clip]:
            print(
                f"  h={row['height']} {row['lo']}/{row['hi']} blur={row['blur']} "
                f"recall={row['recall']:.3f} {row['payload_kbps']:.1f}kbps {row['codec']}",
                flush=True,
            )
    final_winner, _final_pool = choose_winner(final_rows)
    report = {
        "recall_floor_ratio": RECALL_FLOOR_RATIO,
        "zstd_level": args.level,
        "include_chain": bool(args.chain),
        "winner_rank": winner,
        "winner": final_winner,
        "clips": {
            clip: sorted(rows, key=lambda row: row["payload_kbps"])
            for clip, rows in per_clip.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("winner", json.dumps(final_winner, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
