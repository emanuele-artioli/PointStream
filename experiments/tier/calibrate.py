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

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.tier.clip import BP21_CLIPS, load_tier_clip
from src.contracts import paths as ps_paths

#: A clip from a different match, used as the unrelated anchor.
UNRELATED_VIDEO = "sinner_alcaraz"
UNRELATED_SCENE = "scene_001"

#: Extra natural control (different match) when the BP21 cache is present.
#: Unit tests do not require this path; they use synthetic court-like textures.
ADDITIONAL_UNRELATED_CONTROLS: tuple[tuple[str, str], ...] = (
    ("federer_djokovic", "scene_001"),
)

#: Legacy absolute SSIM ceiling for unrelated content. Wave 2 measured 0.6702
#: on natural tennis-vs-tennis frames. Do not lower this number merely to pass
#: a run; if natural controls cannot support it, record the scale and propose
#: a new policy rather than silently changing Gate B.
SSIM_UNRELATED_LEGACY_CEILING = 0.60

PROPOSED_SSIM_CALIBRATION_POLICY: dict[str, Any] = {
    "name": "ssim_natural_control_scale_v1",
    "status": "proposed",
    "gate_b": (
        "unchanged. experiments.tier.protocol.evaluate_confirmation_protocol "
        "still requires caller-supplied calibrated_metrics evidence; this file "
        "does not relax that gate."
    ),
    "keep": [
        "identity > mild > severe on higher-is-better metrics (blur and noise)",
        "mild > unrelated on higher-is-better metrics (partial order; not severe vs unrelated)",
        "SSIM identical >= 0.999",
        "do not treat SSIM as a temporal-order metric",
        "do not alarm on temporal-null SSIM (framewise metric; high scores expected)",
    ],
    "drop": [
        (
            f"SSIM unrelated absolute ceiling {SSIM_UNRELATED_LEGACY_CEILING:.2f} as a "
            "validity condition. Full-frame SSIM on tennis broadcasts is dominated by "
            "shared court geometry, grass, and camera framing, so a different match "
            "can still score above 0.60 (Wave 2: 0.6702)."
        )
    ],
    "replacement": (
        "Record SSIM on at least two structured natural unrelated controls with "
        "source paths and per-frame SHA-256. Quote the observed unrelated scale "
        "beside every SSIM number. Validity is the partial order "
        "(identity>mild>severe and mild>unrelated), not a pre-registered 0.60 floor."
    ),
    "legacy_ssim_unrelated_ceiling": SSIM_UNRELATED_LEGACY_CEILING,
    "bp21_optional_path": str(BP21_CLIPS / "<video>" / "<scene>" / "window"),
}

#: Two frames is enough for an anchor and keeps VMAF's cost bounded. The
#: resolution is *not* reduced: a metric's absolute scale moves with resolution,
#: so an anchor measured on a downscaled frame would not be the scale the run
#: reports in.
ANCHOR_FRAMES = 2


def _frame_sha256(frame: np.ndarray) -> str:
    arr = np.ascontiguousarray(frame, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _stack_identity(frames: np.ndarray) -> dict[str, Any]:
    stack = np.ascontiguousarray(frames, dtype=np.uint8)
    return {
        "shape": [int(dim) for dim in stack.shape],
        "frame_hashes": [_frame_sha256(stack[index]) for index in range(int(stack.shape[0]))],
        "stack_sha256": hashlib.sha256(stack.tobytes()).hexdigest(),
    }


def _looks_like_broadcast(shape: tuple[int, ...]) -> bool:
    return len(shape) >= 3 and int(shape[1]) >= 720 and int(shape[2]) >= 1280


def load_natural_control(
    video: str,
    scene: str,
    shape: tuple[int, ...],
) -> dict[str, Any] | None:
    """Load a BP21 window with confirmed path and per-frame SHA-256, or None."""
    window = BP21_CLIPS / video / scene / "window"
    record: dict[str, Any] = {
        "video": video,
        "scene": scene,
        "path": str(window),
        "available": False,
        "source": "bp21_cache",
    }
    if not window.is_dir():
        return None
    pngs = sorted(window.glob("frame_*.png"))[: int(shape[0])]
    if len(pngs) < int(shape[0]):
        return None
    first_bgr = cv2.imread(str(pngs[0]), cv2.IMREAD_COLOR)
    if first_bgr is None:
        return None
    first = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    if first.shape != tuple(shape[1:]):
        record["available"] = False
        record["reason"] = f"shape {first.shape} does not match reference {shape[1:]}"
        return record
    frames = [first]
    for path in pngs[1:]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    stack = np.stack(frames)
    if stack.shape != tuple(shape):
        return None
    record.update(_stack_identity(stack))
    record["available"] = True
    record["frame_paths"] = [str(path) for path in pngs]
    record["frames"] = stack
    return record


def _unrelated_frames(shape: tuple[int, ...]) -> np.ndarray | None:
    loaded = load_natural_control(UNRELATED_VIDEO, UNRELATED_SCENE, shape)
    if loaded is None or not loaded.get("available"):
        return None
    return loaded["frames"]


def collect_natural_unrelated_controls(
    shape: tuple[int, ...],
    *,
    exclude_video: str | None = None,
) -> list[dict[str, Any]]:
    """Primary UNRELATED clip plus at least one additional natural control when cached."""
    wanted: list[tuple[str, str]] = [(UNRELATED_VIDEO, UNRELATED_SCENE)]
    wanted.extend(ADDITIONAL_UNRELATED_CONTROLS)
    seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []
    for video, scene in wanted:
        if (video, scene) in seen:
            continue
        if exclude_video and video == exclude_video:
            continue
        seen.add((video, scene))
        loaded = load_natural_control(video, scene, shape)
        if loaded is None:
            records.append(
                {
                    "video": video,
                    "scene": scene,
                    "path": str(BP21_CLIPS / video / scene / "window"),
                    "available": False,
                    "source": "bp21_cache",
                }
            )
            continue
        public = {key: value for key, value in loaded.items() if key != "frames"}
        records.append(public)
    return records


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


def synthetic_court_like_clip(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Structured tennis-court-like frames for SSIM scale tests without BP21 clips.

    Shared green field and white lines make full-frame SSIM against another
    court-like clip sit in the same high band as unrelated broadcast tennis,
    which is the regime the 0.60 ceiling was failing on.
    """
    count, height, width, channels = shape
    rng = np.random.default_rng(seed)
    arr = np.zeros((count, height, width, channels), dtype=np.uint8)
    arr[..., 0] = 36
    arr[..., 1] = 142
    arr[..., 2] = 58
    line = max(1, min(2, height // 32))
    arr[:, height // 2 - line : height // 2 + line, :, :] = 230
    arr[:, :, width // 2 - line : width // 2 + line, :] = 230
    arr[:, line : line * 2, line : width - line, :] = 230
    arr[:, height - line * 2 : height - line, line : width - line, :] = 230
    blob_h = max(2, height // 10)
    blob_w = max(2, width // 12)
    for time_index in range(count):
        row = int(np.clip(height // 2 + rng.integers(-height // 6, height // 6 + 1), 0, height - blob_h))
        col = int(np.clip(width // 3 + rng.integers(-width // 8, width // 8 + 1), 0, width - blob_w))
        arr[time_index, row : row + blob_h, col : col + blob_w, :] = (210, 190, 40)
    return arr


def synthetic_foreign_court_clip(shape: tuple[int, ...], seed: int = 99) -> np.ndarray:
    """Structured court-like frames that are not a copy of `synthetic_court_like_clip`.

    Same grass colour and centre lines, different player placement and no outer
    lines. Full-frame SSIM against the reference court typically lands near the
    Wave 2 unrelated-broadcast observation (~0.67): above the legacy 0.60
    ceiling, below mild blur. Used when BP21 clips are absent.
    """
    count, height, width, channels = shape
    rng = np.random.default_rng(seed)
    arr = np.zeros((count, height, width, channels), dtype=np.uint8)
    arr[..., 0] = 36
    arr[..., 1] = 142
    arr[..., 2] = 58
    line = max(1, min(2, height // 32))
    arr[:, height // 2 - line : height // 2 + line, :, :] = 230
    arr[:, :, width // 2 - line : width // 2 + line, :] = 230
    blob_h = max(2, int(height * 0.1))
    blob_w = max(2, int(width * 0.1))
    for time_index in range(count):
        row = int(rng.integers(0, max(1, height - blob_h)))
        col = int(rng.integers(0, max(1, width - blob_w)))
        arr[time_index, row : row + blob_h, col : col + blob_w, :] = (20, 40, 200)
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
    extra_unrelated: list[np.ndarray] | None = None,
    reference_video: str | None = None,
) -> dict[str, Any]:
    """Execute complete metric calibration fixture: anchors plus null controls.

    Partial order (higher-is-better): identity > mild > severe, and mild > unrelated.
    SSIM is not a temporal-order metric; a shuffled temporal null scoring high is
    expected and is not an alarm. The legacy SSIM unrelated ceiling of 0.60 is
    recorded as a scale finding. It is not lowered to pass a run, and exceeding
    it on natural controls does not by itself invalidate this fixture — see
    ``proposed_policy``. Gate B is not changed here.

    Optional BP21 windows live at ``outputs/bp21-headroom/clips/<video>/<scene>/window``.
    Unit tests may pass synthetic court-like textures instead of that cache.
    """
    from src.runner.evaluation import ComponentMetricEvaluator
    from src.contracts.metrics import metric as metric_spec

    # Build anchors
    mild_blur = np.stack([cv2.GaussianBlur(frame, (3, 3), 0.8) for frame in reference])
    severe_blur = np.stack([cv2.GaussianBlur(frame, (21, 21), 9.0) for frame in reference])

    # Add Gaussian noise anchors
    rng = np.random.default_rng(42)
    mild_noise = np.clip(reference.astype(float) + rng.normal(0, 5.0, reference.shape), 0, 255).astype(np.uint8)
    severe_noise = np.clip(reference.astype(float) + rng.normal(0, 30.0, reference.shape), 0, 255).astype(np.uint8)

    unrelated_source = "caller"
    unrelated_clip = unrelated
    primary_control: dict[str, Any] | None = None
    if unrelated_clip is None:
        primary_control = load_natural_control(UNRELATED_VIDEO, UNRELATED_SCENE, reference.shape)
        if primary_control is not None and primary_control.get("available"):
            unrelated_clip = primary_control["frames"]
            unrelated_source = "bp21_cache"
        else:
            unrelated_clip = synthetic_unrelated_clip(reference.shape)
            unrelated_source = "synthetic"
            primary_control = {
                "video": None,
                "scene": None,
                "path": None,
                "available": True,
                "source": "synthetic",
                **_stack_identity(unrelated_clip),
            }
    else:
        primary_control = {
            "video": None,
            "scene": None,
            "path": None,
            "available": True,
            "source": "caller",
            **_stack_identity(unrelated_clip),
        }

    natural_controls: list[dict[str, Any]] = []
    if primary_control is not None:
        natural_controls.append({key: value for key, value in primary_control.items() if key != "frames"})
    if _looks_like_broadcast(tuple(reference.shape)):
        for record in collect_natural_unrelated_controls(
            tuple(reference.shape), exclude_video=reference_video
        ):
            if record.get("video") == UNRELATED_VIDEO and record.get("scene") == UNRELATED_SCENE:
                continue
            natural_controls.append(record)

    extra_table: dict[str, np.ndarray] = {}
    extra_identities: list[dict[str, Any]] = []
    for index, extra in enumerate(extra_unrelated or []):
        name = f"unrelated-extra-{index}"
        extra_table[name] = extra
        extra_identities.append({"name": name, "source": "caller", **_stack_identity(extra)})

    if _looks_like_broadcast(tuple(reference.shape)):
        for video, scene in ADDITIONAL_UNRELATED_CONTROLS:
            if reference_video and video == reference_video:
                continue
            loaded = load_natural_control(video, scene, reference.shape)
            if loaded is None:
                natural_controls.append(
                    {
                        "video": video,
                        "scene": scene,
                        "path": str(BP21_CLIPS / video / scene / "window"),
                        "available": False,
                        "source": "bp21_cache",
                    }
                )
                continue
            public = {key: value for key, value in loaded.items() if key != "frames"}
            if not any(
                item.get("video") == video and item.get("scene") == scene
                for item in natural_controls
            ):
                natural_controls.append(public)
            if loaded.get("available"):
                name = f"unrelated-{video}-{scene}"
                extra_table[name] = loaded["frames"]
                extra_identities.append({**public, "name": name})

    anchor_table: dict[str, np.ndarray] = {
        "identical": reference.copy(),
        "mild-blur": mild_blur,
        "severe-blur": severe_blur,
        "mild-noise": mild_noise,
        "severe-noise": severe_noise,
        "unrelated-clip": unrelated_clip,
        **extra_table,
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
    scale_findings: list[dict[str, Any]] = []

    for metric in evaluator.metric_names:
        spec = metric_spec(metric)
        higher_is_better = spec.higher_is_better

        id_val = float("inf") if scores["identical"][metric] == "inf" else float(scores["identical"][metric])
        mild_blur_val = float("inf") if scores["mild-blur"][metric] == "inf" else float(scores["mild-blur"][metric])
        sev_blur_val = float("inf") if scores["severe-blur"][metric] == "inf" else float(scores["severe-blur"][metric])
        mild_noise_val = float("inf") if scores["mild-noise"][metric] == "inf" else float(scores["mild-noise"][metric])
        sev_noise_val = float("inf") if scores["severe-noise"][metric] == "inf" else float(scores["severe-noise"][metric])
        unr_val = float("inf") if scores["unrelated-clip"][metric] == "inf" else float(scores["unrelated-clip"][metric])

        extra_unrelated_scores = {
            name: scores[name][metric] for name in extra_table
        }

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

        # Absolute value checks. SSIM unrelated 0.60 is a scale finding, not an
        # alarm: natural tennis-vs-tennis frames have been observed at 0.6702.
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
            ceiling_held = unr_val <= SSIM_UNRELATED_LEGACY_CEILING
            extra_observed = {
                name: (
                    float("inf") if extra_unrelated_scores[name] == "inf"
                    else float(extra_unrelated_scores[name])
                )
                for name in extra_unrelated_scores
            }
            scale_findings.append(
                {
                    "metric": "ssim",
                    "name": "legacy_unrelated_ceiling",
                    "ceiling": SSIM_UNRELATED_LEGACY_CEILING,
                    "observed": unr_val,
                    "held": ceiling_held,
                    "additional_unrelated": extra_observed,
                    "temporal_null": (
                        float("inf")
                        if scores["temporal-null-shuffled"][metric] == "inf"
                        else float(scores["temporal-null-shuffled"][metric])
                    ),
                    "temporal_null_is_alarm": False,
                    "note": (
                        "SSIM is framewise; a shuffled temporal null of similar "
                        "frames scoring high is expected and is not an alarm."
                    ),
                }
            )

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
        "unrelated_source": unrelated_source,
        "unrelated_controls": natural_controls,
        "additional_unrelated_identities": extra_identities,
        "reference_identity": _stack_identity(reference),
        "scale_findings": scale_findings,
        "proposed_policy": PROPOSED_SSIM_CALIBRATION_POLICY,
        "how_to_read": (
            "Quote the unrelated-clip and null-control values beside scores. "
            "Runs not beating the unrelated anchor fail floor discrimination. "
            "SSIM unrelated vs the legacy 0.60 ceiling is a scale finding; "
            "validity is the partial order. Temporal-null SSIM is not an alarm."
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
    "ADDITIONAL_UNRELATED_CONTROLS",
    "ANCHOR_FRAMES",
    "PROPOSED_SSIM_CALIBRATION_POLICY",
    "SSIM_UNRELATED_LEGACY_CEILING",
    "UNRELATED_SCENE",
    "UNRELATED_VIDEO",
    "anchors",
    "calibrate",
    "collect_natural_unrelated_controls",
    "load_natural_control",
    "main",
    "run_full_metric_calibration",
    "spatial_null_frames",
    "synthetic_court_like_clip",
    "synthetic_foreign_court_clip",
    "synthetic_unrelated_clip",
    "temporal_null_frames",
]


if __name__ == "__main__":
    raise SystemExit(main())
