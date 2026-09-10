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
from src.contracts import paths as ps_paths

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
    from src.contracts.metrics import metric as metric_spec

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
        spec = metric_spec(metric)
        higher_is_better = spec.higher_is_better
        values = [table[name][metric] for name in order]
        # Direction is a property of the metric contract, not a set of names
        # this file has to remember. Inferring it from the scores is how a
        # metric gets declared correct by the data it is supposed to judge.
        val_id = float("inf") if table["identical"][metric] == "inf" else float(table["identical"][metric])
        val_mild = float("inf") if table["mild-blur"][metric] == "inf" else float(table["mild-blur"][metric])
        val_sev = float("inf") if table["severe-blur"][metric] == "inf" else float(table["severe-blur"][metric])
        val_unr = (float("inf") if table["unrelated-clip"][metric] == "inf" else float(table["unrelated-clip"][metric])) if "unrelated-clip" in table else None

        if higher_is_better:
            dist_ok = (val_id >= val_mild >= val_sev)
            unr_ok = (val_unr is None) or (val_id > val_unr and val_mild > val_unr)
        else:
            dist_ok = (val_id <= val_mild <= val_sev)
            unr_ok = (val_unr is None) or (val_id < val_unr and val_mild < val_unr)

        ordering_held = dist_ok and unr_ok
        verdicts[metric] = {
            "by_anchor": dict(zip(order, values, strict=True)),
            "direction": "higher-is-better" if higher_is_better else "lower-is-better",
            "ordering_held": ordering_held,
            "distortion_ordering_held": dist_ok,
            "unrelated_ordering_held": unr_ok,
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


def synthetic_unrelated_clip(shape: tuple[int, ...]) -> np.ndarray:
    """Generate a deterministic synthetic unrelated scene if broadcast clip is missing."""
    T, H, W, C = shape
    arr = np.zeros((T, H, W, C), dtype=np.uint8)
    for t in range(T):
        y, x = np.ogrid[:H, :W]
        stripe = ((x // 32) + (y // 32) + t) % 2
        arr[t, :, :, 0] = np.uint8(stripe * 220 + 20)
        arr[t, :, :, 1] = np.uint8(((x // 16) % 2) * 180 + 30)
        arr[t, :, :, 2] = np.uint8(((y // 16) % 2) * 200 + 40)
    return arr


def temporal_null_frames(reference: np.ndarray, seed: int = 42) -> np.ndarray:
    """Permute frames along time axis to establish the temporal null floor."""
    count = int(reference.shape[0])
    if count < 2:
        return reference.copy()
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    if np.array_equal(indices, np.arange(count)):
        indices[0], indices[1] = indices[1], indices[0]
    return reference[indices].copy()


def spatial_null_frames(reference: np.ndarray) -> np.ndarray:
    """Constant uniform frame sequence establishing spatial floor."""
    return np.full_like(reference, 128)


def run_full_metric_calibration(
    metrics: list[str],
    reference: np.ndarray,
    *,
    unrelated: np.ndarray | None = None,
) -> dict[str, Any]:
    """Execute complete metric calibration fixture: anchors plus null controls."""
    from src.runner.evaluation import ComponentMetricEvaluator
    from src.contracts.metrics import metric as metric_spec

    # Build anchors
    mild_blur = np.stack([cv2.GaussianBlur(frame, (3, 3), 0.8) for frame in reference])
    severe_blur = np.stack([cv2.GaussianBlur(frame, (21, 21), 9.0) for frame in reference])

    # Add Gaussian noise anchors
    rng = np.random.default_rng(42)
    mild_noise = np.clip(reference.astype(float) + rng.normal(0, 5.0, reference.shape), 0, 255).astype(np.uint8)
    severe_noise = np.clip(reference.astype(float) + rng.normal(0, 30.0, reference.shape), 0, 255).astype(np.uint8)

    unrelated_clip = unrelated
    if unrelated_clip is None:
        unrelated_clip = _unrelated_frames(reference.shape)
    if unrelated_clip is None:
        unrelated_clip = synthetic_unrelated_clip(reference.shape)

    anchor_table: dict[str, np.ndarray] = {
        "identical": reference.copy(),
        "mild-blur": mild_blur,
        "severe-blur": severe_blur,
        "mild-noise": mild_noise,
        "severe-noise": severe_noise,
        "unrelated-clip": unrelated_clip,
    }

    # Null controls
    t_null = temporal_null_frames(reference)
    s_null = spatial_null_frames(reference)
    null_table = {
        "temporal-null-shuffled": t_null,
        "spatial-null-uniform": s_null,
    }

    evaluator = ComponentMetricEvaluator(metrics)
    scores: dict[str, dict[str, float | str]] = {}
    for name, cand in {**anchor_table, **null_table}.items():
        report = evaluator.evaluate(reference, cand)
        scores[name] = {
            item.metric: ("inf" if np.isinf(item.value) else round(float(item.value), 4))
            for item in report.scoped
        }

    # Check partial ordering across blur and noise anchors:
    # Check identity > mild > severe and mild > unrelated separately;
    # do NOT force unrelated below every severe distortion (unrelated natural images can
    # legitimately have higher structural SSIM than heavy gaussian noise; the ordering is partial).
    verdicts: dict[str, Any] = {}
    alarms: list[str] = []

    for metric in evaluator.metric_names:
        spec = metric_spec(metric)
        higher_is_better = spec.higher_is_better

        id_val = float("inf") if scores["identical"][metric] == "inf" else float(scores["identical"][metric])
        mild_blur_val = float("inf") if scores["mild-blur"][metric] == "inf" else float(scores["mild-blur"][metric])
        sev_blur_val = float("inf") if scores["severe-blur"][metric] == "inf" else float(scores["severe-blur"][metric])
        mild_noise_val = float("inf") if scores["mild-noise"][metric] == "inf" else float(scores["mild-noise"][metric])
        sev_noise_val = float("inf") if scores["severe-noise"][metric] == "inf" else float(scores["severe-noise"][metric])
        unr_val = float("inf") if scores["unrelated-clip"][metric] == "inf" else float(scores["unrelated-clip"][metric])

        # Partial ordering:
        # 1. Distortion degradation: identical >= mild >= severe
        # 2. Unrelated discrimination: identical > unrelated and mild > unrelated
        if higher_is_better:
            blur_dist_ok = (id_val >= mild_blur_val >= sev_blur_val)
            blur_unr_ok = (id_val > unr_val and mild_blur_val > unr_val)
            noise_dist_ok = (id_val >= mild_noise_val >= sev_noise_val)
            noise_unr_ok = (id_val > unr_val and mild_noise_val > unr_val)
        else:
            blur_dist_ok = (id_val <= mild_blur_val <= sev_blur_val)
            blur_unr_ok = (id_val < unr_val and mild_blur_val < unr_val)
            noise_dist_ok = (id_val <= mild_noise_val <= sev_noise_val)
            noise_unr_ok = (id_val < unr_val and mild_noise_val < unr_val)

        blur_ok = blur_dist_ok and blur_unr_ok
        noise_ok = noise_dist_ok and noise_unr_ok

        if not blur_dist_ok:
            alarms.append(f"{metric}: blur distortion ordering violated (identical={id_val}, mild={mild_blur_val}, severe={sev_blur_val})")
        if not blur_unr_ok:
            alarms.append(f"{metric}: mild blur failed to discriminate unrelated clip (mild={mild_blur_val}, unrelated={unr_val})")
        if not noise_dist_ok:
            alarms.append(f"{metric}: noise distortion ordering violated (identical={id_val}, mild={mild_noise_val}, severe={sev_noise_val})")
        if not noise_unr_ok:
            alarms.append(f"{metric}: mild noise failed to discriminate unrelated clip (mild={mild_noise_val}, unrelated={unr_val})")

        # Absolute value checks
        if metric == "vmaf":
            if id_val < 95.0 or id_val > 100.0:
                alarms.append(f"VMAF identical score {id_val} outside [95.0, 100.0]")
            if unr_val > 40.0:
                alarms.append(f"VMAF unrelated score {unr_val} > 40.0")
        elif metric == "psnr":
            if id_val != float("inf") and id_val < 80.0:
                alarms.append(f"PSNR identical score {id_val} < 80.0 dB")
            if unr_val > 35.0:
                alarms.append(f"PSNR unrelated score {unr_val} > 35.0 dB (natural clip ceiling)")
        elif metric == "ssim":
            if id_val < 0.999:
                alarms.append(f"SSIM identical score {id_val} < 0.999")
            if unr_val > 0.60:
                alarms.append(f"SSIM unrelated score {unr_val} > 0.60")

        verdicts[metric] = {
            "blur_ordering_held": blur_ok,
            "noise_ordering_held": noise_ok,
            "distortion_ordering_held": blur_dist_ok and noise_dist_ok,
            "unrelated_ordering_held": blur_unr_ok and noise_unr_ok,
            "by_anchor": {name: scores[name][metric] for name in anchor_table},
            "null_controls": {name: scores[name][metric] for name in null_table},
        }

    return {
        "valid": len(alarms) == 0,
        "alarms": alarms,
        "n_frames": int(reference.shape[0]),
        "resolution": f"{reference.shape[2]}x{reference.shape[1]}",
        "metrics": verdicts,
        "how_to_read": (
            "Quote the unrelated-clip and null-control values beside scores. "
            "Runs not beating the unrelated anchor fail floor discrimination."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="*", default=["psnr", "ssim", "vmaf", "lpips"])
    parser.add_argument(
        "--out",
        default=str(ps_paths.outputs() / "bp23-tier" / "metric-calibration.json"),
    )
    args = parser.parse_args(argv)

    clip = load_tier_clip(n_frames=ANCHOR_FRAMES)
    outcome = calibrate(list(args.metrics), clip.frames)
    Path(args.out).write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    return 0


__all__ = [
    "ANCHOR_FRAMES",
    "anchors",
    "calibrate",
    "main",
    "run_full_metric_calibration",
    "spatial_null_frames",
    "synthetic_unrelated_clip",
    "temporal_null_frames",
]


if __name__ == "__main__":
    raise SystemExit(main())
