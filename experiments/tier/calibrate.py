"""Calibrate the metrics a tier asks for, at the resolution it asks for them.

A metric can be perfectly ordered and still be uninterpretable, and this project
has published rankings on two instruments that turned out not to measure what
their name said. So the anchors are part of the measurement, not a follow-up:
identical, a mild perturbation, a severe one, and an unrelated clip, scored by
the same evaluator the run uses, on the same pixels at the same resolution.

The unrelated anchor is another tennis broadcast, not a random field. Random
noise is not a natural image and a feature-space metric has no reason to behave
sensibly on it; the number that matters is what an *irrelevant frame from this
dataset* scores, because that is the floor a real result has to clear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.tier.clip import BP21_CLIPS, load_tier_clip

#: A clip from a different match, used as the unrelated anchor.
UNRELATED_VIDEO = "sinner_alcaraz"
UNRELATED_SCENE = "scene_001"

#: Two frames is enough for an anchor and keeps VMAF's cost bounded. The
#: resolution is *not* reduced: a metric's absolute scale moves with resolution,
#: so an anchor measured on a downscaled frame would not be the scale the run
#: reports in.
ANCHOR_FRAMES = 2


def _unrelated_frames(shape: tuple[int, ...]) -> np.ndarray | None:
    window = BP21_CLIPS / UNRELATED_VIDEO / UNRELATED_SCENE / "window"
    pngs = sorted(window.glob("frame_*.png"))[: shape[0]]
    if len(pngs) < shape[0]:
        return None
    frames = np.stack(
        [cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for path in pngs]
    )
    if frames.shape != tuple(shape):
        return None
    return frames


def anchors(reference: np.ndarray) -> dict[str, np.ndarray]:
    """Identical, mild, severe, unrelated — in that intended order of badness."""
    mild = np.stack([cv2.GaussianBlur(frame, (3, 3), 0.8) for frame in reference])
    severe = np.stack([cv2.GaussianBlur(frame, (21, 21), 9.0) for frame in reference])
    table = {
        "identical": reference.copy(),
        "mild-blur": mild,
        "severe-blur": severe,
    }
    unrelated = _unrelated_frames(reference.shape)
    if unrelated is not None:
        table["unrelated-clip"] = unrelated
    return table


def calibrate(metrics: list[str], reference: np.ndarray) -> dict[str, Any]:
    """Score every anchor with `metrics`, and say whether the ordering held."""
    from src.runner.evaluation import ComponentMetricEvaluator

    evaluator = ComponentMetricEvaluator(metrics)
    table: dict[str, dict[str, float | str]] = {}
    for name, candidate in anchors(reference).items():
        report = evaluator.evaluate(reference, candidate)
        table[name] = {
            item.metric: (
                "inf" if np.isinf(item.value) else round(float(item.value), 5)
            )
            for item in report.scoped
        }

    order = [name for name in ("identical", "mild-blur", "severe-blur", "unrelated-clip") if name in table]
    verdicts: dict[str, Any] = {}
    for metric in evaluator.metric_names:
        values = [table[name][metric] for name in order]
        numeric = [float("inf") if value == "inf" else float(value) for value in values]
        # PSNR, SSIM and VMAF are higher-is-better; LPIPS is a distance. The
        # expected direction is stated per metric rather than inferred, because
        # inferring it from the numbers is how a metric gets declared correct by
        # the very data it is supposed to judge.
        higher_is_better = metric in {"psnr", "ssim", "vmaf"}
        expected = list(reversed(sorted(numeric))) if higher_is_better else sorted(numeric)
        verdicts[metric] = {
            "by_anchor": dict(zip(order, values, strict=True)),
            "direction": "higher-is-better" if higher_is_better else "lower-is-better",
            "ordering_held": numeric == expected,
        }
    return {
        "anchor_frames": int(reference.shape[0]),
        "resolution": f"{reference.shape[2]}x{reference.shape[1]}",
        "unrelated_anchor": f"{UNRELATED_VIDEO}/{UNRELATED_SCENE}",
        "metrics": verdicts,
        "how_to_read": (
            "Quote the unrelated-clip value beside any score from the same "
            "metric. A run that does not clearly beat the unrelated anchor is "
            "not distinguishable from an irrelevant frame."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="*", default=["psnr", "ssim", "vmaf", "lpips"])
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "bp23-tier" / "metric-calibration.json"),
    )
    args = parser.parse_args(argv)

    clip = load_tier_clip(n_frames=ANCHOR_FRAMES)
    outcome = calibrate(list(args.metrics), clip.frames)
    Path(args.out).write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    return 0


__all__ = ["ANCHOR_FRAMES", "anchors", "calibrate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
