"""The cross-appearance control: does this engine use the appearance at all?

Hold the model, the pose, the target frame and the metric fixed. Vary **only**
which keyframe the engine is shown — this clip's own, or a donor clip's from a
different video. Score both against the true target. An engine with a working
appearance pathway is worse with the wrong player; one without is indifferent.

**Why this and not the static-copy floor.** A paste is real pixels in the wrong
pose; a generator is synthetic pixels in the right pose, and MSE structurally
favours the former. "Below the floor" and "does not use appearance" are
different claims and only this one settles the second (``PLAN.md`` §2.4).

**The scale is measured, not assumed.** Pasting the right keyframe rather than
the wrong one is worth ~0.285 LPIPS on this probe set. That is what perfect use
of the appearance signal buys on this metric and this task, so an engine's
delta is reported as a share of it.

Clip mode matters here more than anywhere: the delta measured frame-by-frame
says nothing about a pathway that was structurally disabled by driving a
temporal model at T=1.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.probe.bounds import judge_cross_appearance
from experiments.probe.clips import (
    CLIP_MODE_OFFSETS,
    DEFAULT_KEYFRAME,
    CodingSample,
    ProbeClip,
    list_clips,
    load_coding_sequence,
    with_appearance,
)
from experiments.probe.engines import CANVAS, DEVICE, SEED, EnginePlan, plan_for
from experiments.probe.run import (
    _coding_bundle,
    _generate_clip,
    _release,
    _require_sequence_path,
    _write_json,
    build_lpips,
    donor_appearances,
    predict_static_copy,
)
from experiments.probe.score import score_generation
from src.components.metrics.comparison import compare_paired
from src.contracts.conditioning import GenerationParams

OWN = "own-appearance"
WRONG = "wrong-appearance"


@dataclass
class ArmScores:
    """One arm on one clip, averaged over the clip's frames."""

    object_lpips: float | None
    object_psnr_db: float | None
    n_frames: int


@dataclass
class ClipPair:
    clip_key: str
    donor_key: str
    own: ArmScores
    wrong: ArmScores
    delta_lpips: float | None
    delta_psnr_db: float | None
    error: str | None = None


@dataclass
class CrossAppearanceResult:
    engine: str
    drive_mode: str
    seed: int
    device: str
    offsets: list[int]
    keyframe_index: int
    paste_separation_lpips: float | None
    pairs: list[ClipPair] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=dict)


def _generate(
    plan: EnginePlan,
    built: Any,
    samples: Sequence[CodingSample],
    *,
    seed: int,
    device: str,
    params: GenerationParams,
) -> list[np.ndarray]:
    """Clip mode for a temporal plan, frame by frame otherwise.

    Both arms of the control go through this one function, so the correct and
    the wrong appearance can never be driven by different paths — which would
    make the delta a comparison of invocations rather than of appearances.
    """
    if built is None:
        return [
            predict_static_copy(sample.appearance_rgb, CANVAS, CANVAS) for sample in samples
        ]
    if plan.sequence:
        frames, _wall = _generate_clip(
            plan, built, samples, seed=seed, device=device, params=params
        )
        return frames
    return [
        np.asarray(
            built.generate(_coding_bundle(sample), seed=seed, device=device, params=params)
        )
        for sample in samples
    ]


def _arm(
    plan: EnginePlan,
    built: Any,
    samples: Sequence[CodingSample],
    *,
    seed: int,
    device: str,
    params: GenerationParams,
    lpips_metric: Any,
) -> ArmScores:
    predicted = _generate(plan, built, samples, seed=seed, device=device, params=params)
    lpips_values: list[float] = []
    psnr_values: list[float] = []
    for sample, frame in zip(samples, predicted):
        score = score_generation(
            sample.reference_rgb,
            np.asarray(frame),
            object_mask=sample.object_mask,
            canvas_width=CANVAS,
            canvas_height=CANVAS,
            appearance=sample.appearance_rgb,
            lpips_metric=lpips_metric,
        )
        if score.object_lpips is not None:
            lpips_values.append(float(score.object_lpips))
        if np.isfinite(score.object_psnr_db):
            psnr_values.append(float(score.object_psnr_db))
    return ArmScores(
        object_lpips=_mean(lpips_values),
        object_psnr_db=_mean(psnr_values),
        n_frames=len(samples),
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_cross_appearance(
    engine: str,
    *,
    device: str = DEVICE,
    seed: int = SEED,
    out_dir: Path,
    probe_root: Path | None = None,
    keyframe_index: int = DEFAULT_KEYFRAME,
    offsets: tuple[int, ...] = CLIP_MODE_OFFSETS,
    generator: Any = None,
    lpips_metric: Any = None,
    paste_separation_lpips: float | None = None,
    clips: tuple[ProbeClip, ...] | None = None,
    progress: Any = print,
) -> CrossAppearanceResult:
    """Drive ``engine`` twice per clip and compare. Checkpoints after every clip."""
    from src.components.generation import REGISTRY as GENERATORS

    plan = plan_for(engine)
    probe_clips = clips if clips is not None else list_clips(probe_root)
    donors = donor_appearances(probe_clips, keyframe_index)
    metric = lpips_metric if lpips_metric is not None else build_lpips(device)
    built = generator if generator is not None else GENERATORS.build(engine)
    _require_sequence_path(plan, built)
    params = GenerationParams(width=CANVAS, height=CANVAS, steps=plan.steps)
    drive_mode = "clip" if plan.sequence else "frame"

    result = CrossAppearanceResult(
        engine=engine,
        drive_mode=drive_mode,
        seed=seed,
        device=device,
        offsets=list(offsets),
        keyframe_index=keyframe_index,
        paste_separation_lpips=paste_separation_lpips,
    )
    progress(
        f"[cross] {engine} mode={drive_mode} clips={len(probe_clips)} "
        f"offsets={offsets} device={device} seed={seed}"
    )
    for clip in probe_clips:
        started = time.perf_counter()
        donor_key, donor_rgb = donors[clip.key]
        try:
            samples = load_coding_sequence(clip, keyframe_index, offsets)
            wrong_samples = tuple(with_appearance(sample, donor_rgb) for sample in samples)
            own = _arm(
                plan, built, samples,
                seed=seed, device=device, params=params, lpips_metric=metric,
            )
            wrong = _arm(
                plan, built, wrong_samples,
                seed=seed, device=device, params=params, lpips_metric=metric,
            )
            pair = ClipPair(
                clip_key=clip.key,
                donor_key=donor_key,
                own=own,
                wrong=wrong,
                delta_lpips=(
                    wrong.object_lpips - own.object_lpips
                    if own.object_lpips is not None and wrong.object_lpips is not None
                    else None
                ),
                delta_psnr_db=(
                    own.object_psnr_db - wrong.object_psnr_db
                    if own.object_psnr_db is not None and wrong.object_psnr_db is not None
                    else None
                ),
            )
            progress(
                f"[cross] {engine} {clip.key} donor={donor_key} "
                f"own_lpips={_fmt(own.object_lpips)} wrong_lpips={_fmt(wrong.object_lpips)} "
                f"delta={_fmt(pair.delta_lpips)} ({time.perf_counter() - started:.1f}s)"
            )
        except Exception as exc:  # keep the other clips
            import traceback

            progress(f"[cross] {engine} FAIL {clip.key}: {exc}")
            pair = ClipPair(
                clip_key=clip.key,
                donor_key=donor_key,
                own=ArmScores(None, None, 0),
                wrong=ArmScores(None, None, 0),
                delta_lpips=None,
                delta_psnr_db=None,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        result.pairs.append(pair)
        result.verdict = summarise(result)
        _write_json(out_dir / f"cross-appearance-{engine}.json", asdict(result))
    result.verdict = summarise(result)
    _write_json(out_dir / f"cross-appearance-{engine}.json", asdict(result))
    progress(f"[cross] {engine}: {result.verdict.get('note', 'no verdict')}")
    return result


def summarise(result: CrossAppearanceResult) -> dict[str, Any]:
    """Paired comparison over clips, then the bound verdict.

    Paired because both arms run on the same clips: the per-clip difference
    removes the clip-to-clip variance, which is the dominant term here.
    """
    usable = [
        pair
        for pair in result.pairs
        if pair.error is None
        and pair.own.object_lpips is not None
        and pair.wrong.object_lpips is not None
    ]
    if len(usable) < 2:
        return {
            "readable": False,
            "n": len(usable),
            "note": (
                f"{len(usable)} usable clip(s): a paired comparison needs at least "
                "two, and a difference without a standard error is not a finding."
            ),
        }
    own_lpips = [float(pair.own.object_lpips) for pair in usable]  # type: ignore[arg-type]
    wrong_lpips = [float(pair.wrong.object_lpips) for pair in usable]  # type: ignore[arg-type]
    lpips_cmp = compare_paired(
        WRONG, wrong_lpips, OWN, own_lpips, higher_is_better=False
    )
    verdict = judge_cross_appearance(
        lpips_cmp.mean_difference,
        sigmas=lpips_cmp.sigmas,
        standard_error=lpips_cmp.standard_error,
        paste_separation=result.paste_separation_lpips,
        underpowered=lpips_cmp.verdict == "underpowered",
    )
    payload: dict[str, Any] = {
        "readable": True,
        "n": lpips_cmp.n,
        "status": verdict.status,
        "note": verdict.note,
        "lpips": {
            "own": sum(own_lpips) / len(own_lpips),
            "wrong": sum(wrong_lpips) / len(wrong_lpips),
            "delta": lpips_cmp.mean_difference,
            "standard_error": lpips_cmp.standard_error,
            "sigmas": lpips_cmp.sigmas,
            "comparison": lpips_cmp.describe(),
        },
        "paste_separation_lpips": result.paste_separation_lpips,
    }
    own_psnr = [
        float(pair.own.object_psnr_db)
        for pair in usable
        if pair.own.object_psnr_db is not None and pair.wrong.object_psnr_db is not None
    ]
    wrong_psnr = [
        float(pair.wrong.object_psnr_db)
        for pair in usable
        if pair.own.object_psnr_db is not None and pair.wrong.object_psnr_db is not None
    ]
    if len(own_psnr) >= 2:
        psnr_cmp = compare_paired(OWN, own_psnr, WRONG, wrong_psnr, higher_is_better=True)
        payload["psnr_db"] = {
            "own": sum(own_psnr) / len(own_psnr),
            "wrong": sum(wrong_psnr) / len(wrong_psnr),
            "delta": psnr_cmp.mean_difference,
            "standard_error": psnr_cmp.standard_error,
            "sigmas": psnr_cmp.sigmas,
            "comparison": psnr_cmp.describe(),
        }
    return payload


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def separation_from_summary(path: Path) -> float | None:
    """The measured paste separation from a roster ``summary.json``, if present."""
    if not path.is_file():
        return None
    payload: Mapping[str, Any] = json.loads(path.read_text())
    control = payload.get("control")
    if isinstance(control, Mapping):
        value = control.get("separation")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def main(argv: list[str] | None = None) -> int:
    """``python -m experiments.probe.cross_appearance --engine animate-anyone``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="append", dest="engines", required=True)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("outputs") / "bp12-clip-roster")
    parser.add_argument("--probe-root", type=Path, default=None)
    parser.add_argument("--keyframe", type=int, default=DEFAULT_KEYFRAME)
    parser.add_argument(
        "--offset", action="append", type=int, dest="offsets",
        help="Repeatable. Default: the contiguous clip-mode offsets 1-8.",
    )
    parser.add_argument(
        "--separation-from",
        type=Path,
        default=None,
        help=(
            "A roster summary.json whose measured paste separation sets the "
            "scale. Defaults to summary.json beside --out."
        ),
    )
    args = parser.parse_args(argv)
    offsets = tuple(args.offsets) if args.offsets else CLIP_MODE_OFFSETS
    source = args.separation_from or (args.out / "summary.json")
    separation = separation_from_summary(source)
    if separation is None:
        print(
            f"[cross] no paste separation found at {source}; the delta will be "
            "reported without the scale it should be read against."
        )
    else:
        print(f"[cross] paste separation {separation:.3f} LPIPS, from {source}")
    print("[cross] bounds were written in experiments/probe/bounds.py before this run")
    clips = list_clips(args.probe_root)
    metric = build_lpips(args.device)
    for engine in args.engines:
        run_cross_appearance(
            engine,
            device=args.device,
            seed=args.seed,
            out_dir=args.out,
            keyframe_index=args.keyframe,
            offsets=offsets,
            lpips_metric=metric,
            paste_separation_lpips=separation,
            clips=clips,
        )
        _release(args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
