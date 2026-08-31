"""Score probe generations on ``reid`` through ``TENNIS_SCALE``.

The LPIPS probe ranks pixel distance to the later frame. Identity is a
different question: does the person in the output match the keyframe, quoted
between the measured same-person and different-person anchors. Bounds for this
pass were written in ``outputs/bp19-conditioning/bounds-before-run.json``
before any BP19 generate.

Crops are the bounding box of the letterboxed player mask. ``reid`` refuses a
mask region; a box is the scope it was calibrated on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.probe.clips import list_clips, load_coding_sample, load_frame
from experiments.probe.engines import CANVAS, DEVICE, SEED, plan_for
from experiments.probe.run import _coding_bundle, donor_appearances
from experiments.probe.score import _letterbox_mask, _mask_bbox
from src.components.generation import REGISTRY as GENERATORS
from src.components.generation._numpy import as_hwc, prepare_letterboxed
from src.components.metrics.comparison import compare_paired
from src.components.metrics.reid import TENNIS_SCALE, ReidMetric
from src.contracts.conditioning import GenerationParams

BOUNDS = Path("outputs/bp19-conditioning/bounds-before-run.json")


def _crop(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prepared = prepare_letterboxed(rgb, None, CANVAS, CANVAS)
    hwc = as_hwc(prepared["appearance"])[..., :3]
    letterboxed = _letterbox_mask(mask, prepared["letterbox"])
    box = _mask_bbox(letterboxed)
    if box is None:
        raise ValueError("empty player mask; reid cannot score a hole")
    x1, y1, x2, y2 = box
    return hwc[y1:y2, x1:x2]


def _crop_on_canvas(canvas_hwc: np.ndarray, rgb_for_box: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prepared = prepare_letterboxed(rgb_for_box, None, CANVAS, CANVAS)
    letterboxed = _letterbox_mask(mask, prepared["letterbox"])
    box = _mask_bbox(letterboxed)
    if box is None:
        raise ValueError("empty player mask; reid cannot score a hole")
    x1, y1, x2, y2 = box
    return as_hwc(canvas_hwc)[y1:y2, x1:x2, :3]


def _mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = float(sum(values) / n)
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, float((var / n) ** 0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("outputs") / "bp19-multi-controlnet" / "reid.json")
    args = parser.parse_args(argv)

    bounds = json.loads(BOUNDS.read_text()) if BOUNDS.is_file() else {}
    print("[reid] bounds were written in", BOUNDS, "before any BP19 generate")
    print(
        "[reid] scale",
        TENNIS_SCALE.source,
        "same_person",
        TENNIS_SCALE.same_person,
        "different_person",
        TENNIS_SCALE.different_person,
    )

    clips = list_clips()
    clips_by_key = {clip.key: clip for clip in clips}
    donors = donor_appearances(clips, 0)
    reid = ReidMetric(device=args.device)
    plan = plan_for("multi-controlnet")
    # `Registry[object].build` is typed `object`; the sibling caller in
    # experiments/probe/run.py annotates the built generator `Any` for the
    # same reason, so this matches rather than inventing a second convention.
    engine: Any = GENERATORS.build(plan.name)
    params = GenerationParams(width=CANVAS, height=CANVAS, steps=plan.steps)

    rows: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        donor_key, _donor_rgb = donors[clip.key]
        donor_frame = load_frame(clips_by_key[donor_key], 0)
        appearance_frame = load_frame(clip, 0)
        appearance_crop = _crop(appearance_frame.appearance_rgb, appearance_frame.object_mask)
        donor_crop = _crop(donor_frame.appearance_rgb, donor_frame.object_mask)
        for offset in plan.offsets:
            sample = load_coding_sample(clip, 0, offset)
            target_crop = _crop(sample.reference_rgb, sample.object_mask)
            t0 = time.perf_counter()
            predicted = engine.generate(
                _coding_bundle(sample), seed=args.seed, device=args.device, params=params
            )
            wall_s = time.perf_counter() - t0
            predicted_crop = _crop_on_canvas(
                np.asarray(predicted), sample.reference_rgb, sample.object_mask
            )
            gt_same = float(reid.score(appearance_crop, target_crop))
            gt_wrong = float(reid.score(donor_crop, target_crop))
            engine_score = float(reid.score(predicted_crop, appearance_crop))
            row = {
                "clip_key": clip.key,
                "offset": offset,
                "gt_same": gt_same,
                "gt_wrong": gt_wrong,
                "engine": engine_score,
                "engine_on_scale": TENNIS_SCALE.describe(engine_score),
                "donor_key": donor_key,
                "wall_s": wall_s,
            }
            rows.append(row)
            print(
                f"[reid] {clip.key} offset={offset} engine={engine_score:.4f} "
                f"gt_same={gt_same:.4f} gt_wrong={gt_wrong:.4f} "
                f"{TENNIS_SCALE.describe(engine_score)} {wall_s:.1f}s"
            )
        payload = _summarise(rows, bounds=bounds, seed=args.seed, device=args.device)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"[reid] checkpointed {len(rows)} rows -> {args.out}")

    print("[reid] done.", args.out)
    return 0


def _summarise(
    rows: list[dict[str, Any]],
    *,
    bounds: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, Any]:
    engine = [row["engine"] for row in rows]
    gt_same = [row["gt_same"] for row in rows]
    gt_wrong = [row["gt_wrong"] for row in rows]
    engine_mean, engine_se = _mean_se(engine)
    same_mean, same_se = _mean_se(gt_same)
    wrong_mean, wrong_se = _mean_se(gt_wrong)
    vs_same = compare_paired("engine", engine, "gt_same", gt_same, higher_is_better=True)
    vs_wrong = compare_paired("engine", engine, "gt_wrong", gt_wrong, higher_is_better=True)
    expected = bounds.get("multi_controlnet", {}).get("reid", {})
    return {
        "citable": False,
        "seed": seed,
        "device": device,
        "n": len(rows),
        "n_clips": len({row["clip_key"] for row in rows}),
        "instrument": {
            "name": "reid",
            "scope": "bbox of letterboxed player mask",
            "higher_is_better": True,
            "scale": {
                "same_person": TENNIS_SCALE.same_person,
                "different_person": TENNIS_SCALE.different_person,
                "source": TENNIS_SCALE.source,
            },
        },
        "bounds_written_before_generate": expected,
        "engine": {
            "mean": engine_mean,
            "stderr": engine_se,
            "on_scale": TENNIS_SCALE.describe(engine_mean),
        },
        "gt_same_person": {"mean": same_mean, "stderr": same_se, "on_scale": TENNIS_SCALE.describe(same_mean)},
        "gt_different_person": {
            "mean": wrong_mean,
            "stderr": wrong_se,
            "on_scale": TENNIS_SCALE.describe(wrong_mean),
        },
        "vs_gt_same": vs_same.describe(),
        "vs_gt_wrong": vs_wrong.describe(),
        "winner_vs_same": vs_same.winner,
        "winner_vs_wrong": vs_wrong.winner,
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
