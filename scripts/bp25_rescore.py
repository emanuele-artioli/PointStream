# ruff: noqa: E402 - sys.path bootstrap must run before `from src...`.
"""Calibrate the 4-step stop-eval, then re-score trained IP-Adapter checkpoints.

Bounds are written to disk before the first generation. The ranking protocol
matches PLAN.md §2.10: 20 diffusion steps, 12 probe clips, offsets 1–8,
object-bbox LPIPS against real-image anchors. The 4-step stop-eval is a
separate instrument and is only used in the calibration stage.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path as _Path

_REPO_ROOT = str(_Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pathlib import Path
from typing import Any

import numpy as np

from experiments.probe.clips import (
    CLIP_MODE_OFFSETS,
    list_clips,
    load_coding_sample,
    load_frame,
    with_appearance,
)
from experiments.probe.engines import CANVAS, SEED
from experiments.probe.run import (
    _coding_bundle,
    _score_coding,
    donor_appearances,
    predict_static_copy,
)
from experiments.probe.score import _letterbox_mask, _mask_bbox
from src.components.generation._numpy import as_hwc, prepare_letterboxed
from src.components.generation.controlnet import ControlNetGenerator
from src.components.metrics.comparison import compare_paired
from src.components.metrics.lpips import LpipsMetric
from src.components.metrics.reid import TENNIS_SCALE, ReidMetric
from src.contracts.conditioning import GenerationParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("bp25")

WEIGHTS = Path("assets") / "weights" / "ip-adapter-trained"
STOCK_NUMBER_TO_BEAT = 0.7606
ALARM_PASTE_THROUGH = 0.45
ALARM_WORSE_THAN_UNRELATED = 0.74
ALARM_SAME_PERSON = 0.8663

# Written to disk before any generate. Do not edit after looking at a result.
BOUNDS = {
    "written_before_generate": True,
    "ranking_protocol": (
        "20-step generations vs real-image static-copy and unrelated anchors, "
        "same protocol as PLAN.md §2.10. The 4-step stop-eval is calibrated "
        "separately and is not used to rank models."
    ),
    "object_lpips": {
        "scope": "bbox of letterboxed object mask",
        "lower_is_better": True,
        "static_copy_expected": [0.43, 0.55],
        "unrelated_expected": [0.72, 0.76],
        "pose_controlnet_20_expected": [0.55, 0.66],
        "stock_ip_adapter_20_expected": [0.73, 0.80],
        "stock_ip_adapter_20_published": STOCK_NUMBER_TO_BEAT,
        "trained_expected": [0.50, 0.78],
        "alarm_paste_through_below": 0.45,
        "alarm_worse_than_unrelated_above": 0.74,
        "number_to_beat": STOCK_NUMBER_TO_BEAT,
        "basis": (
            "BP19 L206-211 and PLAN.md §2.10. Pose roster 0.6031; paste ~0.45; "
            "unrelated 0.7358; stock IP-Adapter 0.7606. Below 0.45 is paste-through; "
            "above 0.74 is worse than showing a different player."
        ),
    },
    "calibration_4_vs_20": {
        "question": "Can a 4-step eval distinguish a good generation from a bad one?",
        "same_model_4_vs_20_expected_delta": [0.02, 0.20],
        "basis": (
            "Vanilla SD1.5 needs 20–50 steps. 4-step txt2img from noise should "
            "score worse than 20-step of the same model. If they do not separate "
            "at >=2 SE, the tripwire cannot rank models. Pose-controlnet is "
            "img2img (strength 0.65) so it is a secondary arm, not the primary "
            "known-good for the IP-Adapter instrument."
        ),
    },
    "reid": {
        "scope": "bbox of letterboxed object mask",
        "higher_is_better": True,
        "expected": [0.53, 0.72],
        "alarm_same_person_anchor": 0.8663,
        "different_person_anchor": 0.5315,
        "basis": "BP19 / TENNIS_SCALE. Ceiling is semantic appearance, not identity.",
    },
}


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


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lpips = [float(row["object_lpips"]) for row in rows]
    psnr = [float(row["object_psnr_db"]) for row in rows]
    reid = [float(row["reid"]) for row in rows if row.get("reid") is not None]
    lpips_mean, lpips_se = _mean_se(lpips)
    psnr_mean, psnr_se = _mean_se(psnr)
    out: dict[str, Any] = {
        "n": len(rows),
        "n_clips": len({row["clip_key"] for row in rows}),
        "object_lpips": {"mean": lpips_mean, "stderr": lpips_se, "scope": "bbox of object mask"},
        "object_psnr_db": {"mean": psnr_mean, "stderr": psnr_se, "scope": "object mask"},
    }
    if reid:
        reid_mean, reid_se = _mean_se(reid)
        out["reid"] = {
            "mean": reid_mean,
            "stderr": reid_se,
            "on_scale": TENNIS_SCALE.describe(reid_mean),
            "scope": "bbox of letterboxed object mask",
        }
    return out


def _release(generator: Any) -> None:
    pipe = getattr(generator, "_pipeline", None)
    if generator is not None:
        generator._pipeline = None
    del pipe
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _item_key(arm: str, sample_key: str, offset: int, steps: int | None) -> str:
    return f"{arm}|{sample_key}|offset={offset}|steps={steps}"


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    return {row["item_key"]: row for row in rows}


def _score_prediction(
    sample,
    predicted: np.ndarray,
    *,
    lpips: LpipsMetric,
    reid: ReidMetric | None,
    appearance_crop: np.ndarray | None,
) -> dict[str, Any]:
    score = _score_coding(sample, predicted, lpips_metric=lpips)
    row: dict[str, Any] = {
        "clip_key": sample.key,
        "offset": sample.offset,
        "object_lpips": score.object_lpips,
        "object_psnr_db": score.object_psnr_db,
        "n_object_pixels": score.n_object_pixels,
        "differs_from_input": score.differs_from_input,
        "differs_from_reference": score.differs_from_reference,
        "reid": None,
    }
    if reid is not None and appearance_crop is not None:
        predicted_crop = _crop_on_canvas(
            np.asarray(predicted), sample.reference_rgb, sample.object_mask
        )
        row["reid"] = float(reid.score(predicted_crop, appearance_crop))
        row["reid_on_scale"] = TENNIS_SCALE.describe(row["reid"])
    return row


def _run_copy_arm(
    name: str,
    samples: list,
    *,
    predict,
    lpips: LpipsMetric,
    reid: ReidMetric | None,
    appearance_crops: dict[str, np.ndarray],
    store: dict[str, dict[str, Any]],
    out_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        item_key = _item_key(name, sample.key, sample.offset, None)
        if item_key in store:
            rows.append(store[item_key])
            continue
        predicted = predict(sample)
        row = _score_prediction(
            sample,
            predicted,
            lpips=lpips,
            reid=reid,
            appearance_crop=appearance_crops.get(_clip_id(sample.key)),
        )
        row["arm"] = name
        row["steps"] = None
        row["item_key"] = item_key
        store[item_key] = row
        rows.append(row)
        log.info(
            "[bp25] %s %s lpips=%.4f psnr=%.2f reid=%s",
            name,
            sample.key,
            row["object_lpips"],
            row["object_psnr_db"],
            row["reid"],
        )
        _write(out_path, {"rows": list(store.values())})
    return rows


def _run_generator_arm(
    name: str,
    samples: list,
    *,
    generator: ControlNetGenerator,
    steps: int,
    device: str,
    seed: int,
    lpips: LpipsMetric,
    reid: ReidMetric | None,
    appearance_crops: dict[str, np.ndarray],
    store: dict[str, dict[str, Any]],
    out_path: Path,
    sample_override=None,
) -> list[dict[str, Any]]:
    params = GenerationParams(width=CANVAS, height=CANVAS, steps=steps)
    rows: list[dict[str, Any]] = []
    pending = [
        sample
        for sample in samples
        if _item_key(name, sample.key, sample.offset, steps) not in store
    ]
    for sample in samples:
        item_key = _item_key(name, sample.key, sample.offset, steps)
        if item_key in store:
            rows.append(store[item_key])
            continue
        driven = sample_override(sample) if sample_override is not None else sample
        t0 = time.perf_counter()
        predicted = generator.generate(
            _coding_bundle(driven), seed=seed, device=device, params=params
        )
        wall_s = time.perf_counter() - t0
        row = _score_prediction(
            sample,
            predicted,
            lpips=lpips,
            reid=reid,
            appearance_crop=appearance_crops.get(_clip_id(sample.key)),
        )
        row["arm"] = name
        row["steps"] = steps
        row["item_key"] = item_key
        row["wall_s"] = wall_s
        row["loaded_checkpoint"] = generator.loaded_checkpoint
        store[item_key] = row
        rows.append(row)
        remaining = sum(
            1
            for other in pending
            if _item_key(name, other.key, other.offset, steps) not in store
        )
        log.info(
            "[bp25] %s steps=%s %s lpips=%.4f psnr=%.2f reid=%s %.1fs remaining=%s",
            name,
            steps,
            sample.key,
            row["object_lpips"],
            row["object_psnr_db"],
            row["reid"],
            wall_s,
            remaining,
        )
        _write(out_path, {"rows": list(store.values())})
    return rows


def _clip_id(sample_key: str) -> str:
    """CodingSample.key is the clip key; keep a helper so a later key format change is one line."""
    return sample_key


def _paired(
    name_a: str,
    rows_a: list[dict],
    name_b: str,
    rows_b: list[dict],
    *,
    field: str,
    higher_is_better: bool,
):
    by_b = {(row["clip_key"], row["offset"]): row.get(field) for row in rows_b}
    a_vals = []
    b_vals = []
    for row in rows_a:
        other = by_b.get((row["clip_key"], row["offset"]))
        if other is None or row.get(field) is None:
            continue
        a_vals.append(float(row[field]))
        b_vals.append(float(other))
    if len(a_vals) < 2:
        return None
    return compare_paired(
        name_a, a_vals, name_b, b_vals, higher_is_better=higher_is_better
    )


def _describe(comparison) -> str:
    if comparison is None:
        return "not enough paired items"
    return comparison.describe()


def _alarms(lpips_mean: float, reid_mean: float | None) -> list[str]:
    fired: list[str] = []
    if lpips_mean < ALARM_PASTE_THROUGH:
        fired.append(
            f"LPIPS {lpips_mean:.4f} below paste-through alarm {ALARM_PASTE_THROUGH}"
        )
    if lpips_mean > ALARM_WORSE_THAN_UNRELATED:
        fired.append(
            f"LPIPS {lpips_mean:.4f} above worse-than-unrelated alarm "
            f"{ALARM_WORSE_THAN_UNRELATED}"
        )
    if reid_mean is not None and reid_mean >= ALARM_SAME_PERSON - 0.02:
        fired.append(
            f"reid {reid_mean:.4f} at the same-person anchor {ALARM_SAME_PERSON}"
        )
    return fired


def _samples(clips, offsets: tuple[int, ...]):
    out = []
    for clip in clips:
        for offset in offsets:
            if clip.n_frames > offset:
                out.append(load_coding_sample(clip, 0, offset))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "bp25-ip-adapter")
    parser.add_argument("--n-clips", type=int, default=12)
    parser.add_argument("--calib-offset", type=int, default=8)
    parser.add_argument("--steps-short", type=int, default=4)
    parser.add_argument("--steps-full", type=int, default=20)
    parser.add_argument(
        "--stage",
        choices=("calibrate", "rank", "all"),
        default="all",
    )
    parser.add_argument(
        "--skip-pose",
        action="store_true",
        help="Skip pose-controlnet in calibration (img2img, secondary arm).",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds_path = out_dir / "bounds-before-run.json"
    if not bounds_path.is_file():
        _write(bounds_path, BOUNDS)
        log.info("[bp25] wrote bounds before generate -> %s", bounds_path)
    else:
        log.info("[bp25] bounds already on disk at %s; not rewriting", bounds_path)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.device.startswith("cuda"):
        index = int(args.device.split(":")[1]) if ":" in args.device else 0
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        log.info(
            "[bp25] using %s (%s, %.1f/%.1f GiB free)",
            args.device,
            props.name,
            free / 1024**3,
            total / 1024**3,
        )

    clips = list(list_clips())[: args.n_clips]
    if len(clips) < 8:
        log.warning("[bp25] only %s clips; comparisons with n<8 are underpowered", len(clips))
    calib_samples = _samples(clips, (args.calib_offset,))
    rank_samples = _samples(clips, CLIP_MODE_OFFSETS)
    log.info(
        "[bp25] clips=%s calib_items=%s rank_items=%s",
        len(clips),
        len(calib_samples),
        len(rank_samples),
    )

    lpips = LpipsMetric(device=args.device)
    reid = ReidMetric(device=args.device)
    appearance_crops: dict[str, np.ndarray] = {}
    for clip in clips:
        appearance_frame = load_frame(clip, 0)
        appearance_crops[clip.key] = _crop(
            appearance_frame.appearance_rgb, appearance_frame.object_mask
        )

    donors = donor_appearances(clips, 0)
    store_path = out_dir / "rows.json"
    store = _load_rows(store_path)

    def appearance_crop_for(sample_key: str) -> np.ndarray:
        return appearance_crops[_clip_id(sample_key)]

    # --- photo anchors (no diffusion) ---
    def predict_copy(sample):
        return predict_static_copy(sample.appearance_rgb, CANVAS, CANVAS)

    donor_rgb = {clip.key: donors[clip.key][1] for clip in clips}

    def predict_unrelated(sample):
        rgb = donor_rgb[_clip_id(sample.key)]
        return predict_static_copy(rgb, CANVAS, CANVAS)

    copy_rows = _run_copy_arm(
        "static-copy",
        rank_samples,
        predict=predict_copy,
        lpips=lpips,
        reid=reid,
        appearance_crops=appearance_crops,
        store=store,
        out_path=store_path,
    )
    null_rows = _run_copy_arm(
        "unrelated-image",
        rank_samples,
        predict=predict_unrelated,
        lpips=lpips,
        reid=reid,
        appearance_crops=appearance_crops,
        store=store,
        out_path=store_path,
    )

    identical_rows: list[dict[str, Any]] = []
    for sample in calib_samples:
        item_key = _item_key("identical", sample.key, sample.offset, None)
        if item_key in store:
            identical_rows.append(store[item_key])
            continue
        predicted = predict_static_copy(sample.reference_rgb, CANVAS, CANVAS)
        row = _score_prediction(
            sample,
            predicted,
            lpips=lpips,
            reid=reid,
            appearance_crop=appearance_crop_for(sample.key),
        )
        row["arm"] = "identical"
        row["steps"] = None
        row["item_key"] = item_key
        store[item_key] = row
        identical_rows.append(row)
    _write(store_path, {"rows": list(store.values())})

    calib_report: dict[str, Any] = {}
    if args.stage in {"calibrate", "all"}:
        calib_arms: list[tuple[str, str | None, int]] = [
            ("stock-ip-adapter", None, args.steps_short),
            ("stock-ip-adapter", None, args.steps_full),
        ]
        if not args.skip_pose:
            calib_arms.extend(
                [
                    ("pose-controlnet", "pose", args.steps_short),
                    ("pose-controlnet", "pose", args.steps_full),
                ]
            )
        calib_rows: dict[str, list[dict[str, Any]]] = {}
        for arm_name, variant, steps in calib_arms:
            key = f"{arm_name}@{steps}"
            log.info("[bp25] calibration arm %s", key)
            if variant == "pose":
                generator = ControlNetGenerator(variant="pose", steps=steps)
            else:
                generator = ControlNetGenerator(variant="ip-adapter", steps=steps)
            try:
                calib_rows[key] = _run_generator_arm(
                    arm_name,
                    calib_samples,
                    generator=generator,
                    steps=steps,
                    device=args.device,
                    seed=args.seed,
                    lpips=lpips,
                    reid=reid,
                    appearance_crops=appearance_crops,
                    store=store,
                    out_path=store_path,
                )
            finally:
                _release(generator)

        stock4 = calib_rows[f"stock-ip-adapter@{args.steps_short}"]
        stock20 = calib_rows[f"stock-ip-adapter@{args.steps_full}"]
        calib_copy = [row for row in copy_rows if row["offset"] == args.calib_offset]
        calib_null = [row for row in null_rows if row["offset"] == args.calib_offset]
        same_model = _paired(
            f"stock@{args.steps_short}",
            stock4,
            f"stock@{args.steps_full}",
            stock20,
            field="object_lpips",
            higher_is_better=False,
        )
        vs_copy_4 = _paired(
            f"stock@{args.steps_short}",
            stock4,
            "static-copy",
            calib_copy,
            field="object_lpips",
            higher_is_better=False,
        )
        vs_copy_20 = _paired(
            f"stock@{args.steps_full}",
            stock20,
            "static-copy",
            calib_copy,
            field="object_lpips",
            higher_is_better=False,
        )
        vs_null_4 = _paired(
            f"stock@{args.steps_short}",
            stock4,
            "unrelated",
            calib_null,
            field="object_lpips",
            higher_is_better=False,
        )
        can_rank = same_model is not None and same_model.verdict == "clear"
        calib_report = {
            "identical": _summarise(identical_rows),
            "static-copy": _summarise(calib_copy),
            "unrelated-image": _summarise(calib_null),
            **{key: _summarise(rows) for key, rows in calib_rows.items()},
            "comparisons": {
                "stock_4_vs_20": _describe(same_model),
                "stock_4_vs_static_copy": _describe(vs_copy_4),
                "stock_20_vs_static_copy": _describe(vs_copy_20),
                "stock_4_vs_unrelated": _describe(vs_null_4),
            },
            "verdict": {
                "four_step_can_rank_models": can_rank,
                "reason": (
                    "4-step vs 20-step of the same stock IP-Adapter txt2img pipeline "
                    "separated at >=2 SE. Ranking still uses the 20-step roster "
                    "protocol, because that is what 0.7606 was measured at."
                    if can_rank
                    else (
                        "4-step and 20-step of the same stock IP-Adapter did not "
                        "separate at >=2 SE. The 4-step tripwire cannot rank models."
                    )
                ),
            },
        }
        pose4 = calib_rows.get(f"pose-controlnet@{args.steps_short}", [])
        pose20 = calib_rows.get(f"pose-controlnet@{args.steps_full}", [])
        if pose4 and pose20:
            calib_report["comparisons"]["pose_4_vs_20"] = _describe(
                _paired(
                    f"pose@{args.steps_short}",
                    pose4,
                    f"pose@{args.steps_full}",
                    pose20,
                    field="object_lpips",
                    higher_is_better=False,
                )
            )
            calib_report["comparisons"]["pose_4_vs_stock_4"] = _describe(
                _paired(
                    f"pose@{args.steps_short}",
                    pose4,
                    f"stock@{args.steps_short}",
                    stock4,
                    field="object_lpips",
                    higher_is_better=False,
                )
            )
        _write(out_dir / "calibration.json", calib_report)
        log.info("[bp25] calibration verdict: %s", calib_report["verdict"])

    rank_report: dict[str, Any] = {}
    if args.stage in {"rank", "all"}:
        checkpoints = [
            ("stock-ip-adapter", None),
            ("checkpoint-epoch-1", str(WEIGHTS / "checkpoint-epoch-1")),
            ("checkpoint-epoch-2", str(WEIGHTS / "checkpoint-epoch-2")),
            ("checkpoint-epoch-3", str(WEIGHTS / "checkpoint-epoch-3")),
        ]
        for _arm, ckpt in checkpoints:
            if ckpt is None:
                continue
            adapter = Path(ckpt) / "ip-adapter.bin"
            if not adapter.is_file():
                raise FileNotFoundError(f"trained adapter missing: {adapter}")

        rank_rows: dict[str, list[dict[str, Any]]] = {
            "static-copy": copy_rows,
            "unrelated-image": null_rows,
        }
        for arm_name, ckpt in checkpoints:
            log.info("[bp25] ranking arm %s steps=%s", arm_name, args.steps_full)
            generator = ControlNetGenerator(
                variant="ip-adapter", checkpoint=ckpt, steps=args.steps_full
            )
            try:
                rank_rows[arm_name] = _run_generator_arm(
                    arm_name,
                    rank_samples,
                    generator=generator,
                    steps=args.steps_full,
                    device=args.device,
                    seed=args.seed,
                    lpips=lpips,
                    reid=reid,
                    appearance_crops=appearance_crops,
                    store=store,
                    out_path=store_path,
                )
            finally:
                _release(generator)

        # Shuffled-appearance control on every trained checkpoint, same items.
        for epoch in (1, 2, 3):
            arm_name = f"checkpoint-epoch-{epoch}-shuffled"
            ckpt = str(WEIGHTS / f"checkpoint-epoch-{epoch}")
            log.info("[bp25] shuffled-appearance arm %s", arm_name)

            def _shuffle(sample, _clips=clips, _donors=donors):
                donor_key, rgb = _donors[_clip_id(sample.key)]
                shuffled = with_appearance(sample, rgb)
                return shuffled

            generator = ControlNetGenerator(
                variant="ip-adapter", checkpoint=ckpt, steps=args.steps_full
            )
            try:
                rank_rows[arm_name] = _run_generator_arm(
                    arm_name,
                    rank_samples,
                    generator=generator,
                    steps=args.steps_full,
                    device=args.device,
                    seed=args.seed,
                    lpips=lpips,
                    reid=reid,
                    appearance_crops=appearance_crops,
                    store=store,
                    out_path=store_path,
                    sample_override=_shuffle,
                )
            finally:
                _release(generator)

        comparisons: dict[str, str] = {
            "static_copy_vs_unrelated": _describe(
                _paired(
                    "static-copy",
                    copy_rows,
                    "unrelated",
                    null_rows,
                    field="object_lpips",
                    higher_is_better=False,
                )
            ),
            "stock_vs_static_copy": _describe(
                _paired(
                    "stock-ip-adapter",
                    rank_rows["stock-ip-adapter"],
                    "static-copy",
                    copy_rows,
                    field="object_lpips",
                    higher_is_better=False,
                )
            ),
        }
        appearance_use: dict[str, str] = {}
        for epoch in (1, 2, 3):
            name = f"checkpoint-epoch-{epoch}"
            comparisons[f"{name}_vs_stock"] = _describe(
                _paired(
                    name,
                    rank_rows[name],
                    "stock-ip-adapter",
                    rank_rows["stock-ip-adapter"],
                    field="object_lpips",
                    higher_is_better=False,
                )
            )
            comparisons[f"{name}_vs_static_copy"] = _describe(
                _paired(
                    name,
                    rank_rows[name],
                    "static-copy",
                    copy_rows,
                    field="object_lpips",
                    higher_is_better=False,
                )
            )
            comparisons[f"{name}_vs_unrelated"] = _describe(
                _paired(
                    name,
                    rank_rows[name],
                    "unrelated",
                    null_rows,
                    field="object_lpips",
                    higher_is_better=False,
                )
            )
            vs_shuffled = _paired(
                name,
                rank_rows[name],
                f"{name}-shuffled",
                rank_rows[f"{name}-shuffled"],
                field="object_lpips",
                higher_is_better=False,
            )
            comparisons[f"{name}_vs_shuffled"] = _describe(vs_shuffled)
            comparisons[f"{name}_reid_vs_shuffled"] = _describe(
                _paired(
                    name,
                    rank_rows[name],
                    f"{name}-shuffled",
                    rank_rows[f"{name}-shuffled"],
                    field="reid",
                    higher_is_better=True,
                )
            )
            if vs_shuffled is None:
                appearance_use[name] = "unresolved (not enough paired items)"
            elif vs_shuffled.verdict == "clear" and vs_shuffled.winner == name:
                appearance_use[name] = "uses appearance"
            elif vs_shuffled.verdict == "clear":
                appearance_use[name] = "not using appearance"
            else:
                appearance_use[name] = f"unresolved ({vs_shuffled.describe()})"

        summaries = {name: _summarise(rows) for name, rows in rank_rows.items()}
        fired: dict[str, list[str]] = {}
        for name, summary in summaries.items():
            reid_mean = summary.get("reid", {}).get("mean")
            fired[name] = _alarms(summary["object_lpips"]["mean"], reid_mean)

        rank_report = {
            "protocol": {
                "steps": args.steps_full,
                "n_clips": len(clips),
                "offsets": list(CLIP_MODE_OFFSETS),
                "seed": args.seed,
                "device": args.device,
                "canvas": CANVAS,
                "number_to_beat": STOCK_NUMBER_TO_BEAT,
            },
            "arms": summaries,
            "comparisons": comparisons,
            "alarms": fired,
            "appearance_use": appearance_use,
        }
        _write(out_dir / "ranking.json", rank_report)
        log.info("[bp25] ranking written -> %s", out_dir / "ranking.json")
        for name, summary in summaries.items():
            lp = summary["object_lpips"]
            log.info(
                "[bp25] %s LPIPS %.4f +/- %.4f (n=%s) alarms=%s",
                name,
                lp["mean"],
                lp["stderr"],
                summary["n"],
                fired[name],
            )

    report = {
        "bounds_path": str(bounds_path),
        "calibration": calib_report,
        "ranking": rank_report,
    }
    _write(out_dir / "report.json", report)
    log.info("[bp25] done -> %s", out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
