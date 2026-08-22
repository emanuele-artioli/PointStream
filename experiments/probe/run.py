"""Drive one engine over the probe set. Checkpoint after every clip."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.probe.bounds import BoundVerdict, judge_frame_gap, judge_object_psnr
from experiments.probe.clips import ProbeClip, ProbeFrame, bundle_arrays, list_clips, load_frame
from experiments.probe.construct import stated_reason
from experiments.probe.engines import CANVAS, DEVICE, SEED, EnginePlan, PLANS, plan_for
from experiments.probe.score import ProbeScore, score_generation
from src.contracts.conditioning import ConditioningBundle, GenerationParams

HEADLINE_FRAME_INDEX = 24


@dataclass
class ClipResult:
    engine: str
    clip_key: str
    split: str
    frame_index: int
    object_psnr_db: float | None
    frame_psnr_db: float | None
    seed: int
    checkpoint_epoch: int | str | None
    peak_vram_bytes: int | None
    wall_s: float | None
    differs_from_input: bool | None
    n_object_pixels: int | None
    region_kind: str | None
    object_bound: str | None
    gap_bound: str | None
    error: str | None = None


@dataclass
class EngineResult:
    engine: str
    kind: str
    notes: str
    refused: bool
    refuse_reason: str | None
    seed: int
    checkpoint_epoch: int | str | None
    peak_vram_bytes: int | None
    clips: list[ClipResult] = field(default_factory=list)
    headline: dict[str, Any] = field(default_factory=dict)


def _device_is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _reset_peak(device: str) -> None:
    if not _device_is_cuda(device):
        return
    import torch

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(torch.device(device).index or 0)


def _peak_bytes(device: str) -> int:
    if not _device_is_cuda(device):
        return 0
    import torch

    return int(torch.cuda.max_memory_allocated(torch.device(device).index or 0))


def _release(device: str) -> None:
    import gc

    gc.collect()
    if not _device_is_cuda(device):
        return
    import torch

    torch.cuda.empty_cache()


def _epoch_of(generator: Any) -> int | str | None:
    epoch = getattr(generator, "loaded_epoch", None)
    if epoch is not None:
        return epoch
    checkpoint = getattr(generator, "loaded_checkpoint", None)
    if checkpoint:
        return str(checkpoint)
    last_run = getattr(generator, "last_run", None)
    if isinstance(last_run, dict) and last_run.get("checkpoint"):
        return str(last_run["checkpoint"])
    return None


def _bundle(frame: ProbeFrame) -> ConditioningBundle:
    payload = bundle_arrays(frame)
    return ConditioningBundle(
        appearance=payload["appearance"],
        pose=payload["pose"],
        mask=payload["mask"],
        canny=payload["canny"],
        motion_field=payload["motion_field"],
        frame_index=payload["frame_index"],
        object_id=payload["object_id"],
    )


def _score(frame: ProbeFrame, predicted: np.ndarray) -> ProbeScore:
    return score_generation(
        frame.appearance_rgb,
        predicted,
        object_mask=frame.object_mask,
        canvas_width=CANVAS,
        canvas_height=CANVAS,
    )


def _bounds(engine: str, score: ProbeScore) -> tuple[BoundVerdict, BoundVerdict]:
    return (
        judge_object_psnr(engine, score.object_psnr_db),
        judge_frame_gap(score.frame_psnr_db, score.object_psnr_db),
    )


def _clip_row(
    plan: EnginePlan,
    frame: ProbeFrame,
    score: ProbeScore,
    *,
    seed: int,
    epoch: int | str | None,
    peak: int | None,
    wall_s: float,
) -> ClipResult:
    object_bound, gap_bound = _bounds(plan.name, score)
    return ClipResult(
        engine=plan.name,
        clip_key=frame.key,
        split=frame.split,
        frame_index=frame.frame_index,
        object_psnr_db=score.object_psnr_db,
        frame_psnr_db=score.frame_psnr_db,
        seed=seed,
        checkpoint_epoch=epoch,
        peak_vram_bytes=peak,
        wall_s=wall_s,
        differs_from_input=score.differs_from_input,
        n_object_pixels=score.n_object_pixels,
        region_kind=score.region_kind,
        object_bound=object_bound.status,
        gap_bound=gap_bound.status,
    )


def _headline(plan: EnginePlan, rows: list[ClipResult]) -> dict[str, Any]:
    matched = [
        row
        for row in rows
        if row.error is None
        and row.frame_index == HEADLINE_FRAME_INDEX
        and row.object_psnr_db is not None
    ]
    if not matched:
        matched = [row for row in rows if row.error is None and row.object_psnr_db is not None]
    if not matched:
        return {"n": 0, "object_psnr_db": None, "frame_psnr_db": None}
    object_values = [float(row.object_psnr_db) for row in matched if row.object_psnr_db is not None]
    frame_values = [float(row.frame_psnr_db) for row in matched if row.frame_psnr_db is not None]
    mean_object = sum(object_values) / len(object_values)
    mean_frame = sum(frame_values) / len(frame_values)
    verdict = judge_object_psnr(plan.name, mean_object)
    gap = judge_frame_gap(mean_frame, mean_object)
    identity_fail = [row.clip_key for row in matched if row.differs_from_input is False]
    return {
        "n": len(matched),
        "frame_index": HEADLINE_FRAME_INDEX,
        "object_psnr_db": mean_object,
        "frame_psnr_db": mean_frame,
        "object_bound": verdict.status,
        "object_bound_note": verdict.note,
        "gap_bound": gap.status,
        "gap_db": mean_frame - mean_object,
        "identity_failures": identity_fail,
        "mean_wall_s": sum(float(row.wall_s or 0.0) for row in matched) / len(matched),
        "peak_vram_bytes": max((row.peak_vram_bytes or 0) for row in matched),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def drive_engine(
    plan: EnginePlan,
    clips: tuple[ProbeClip, ...],
    *,
    device: str,
    seed: int,
    out_dir: Path,
    generator: Any | None = None,
    progress: Any = print,
) -> EngineResult:
    """Run ``plan`` over ``clips``. ``generator`` is injected in tests."""
    from src.components.generation import REGISTRY as GENERATORS

    started = time.perf_counter()
    progress(f"[probe] {plan.name} kind={plan.kind} clips={len(clips)} device={device}")
    if plan.refuse_at == "construct":
        try:
            GENERATORS.build(plan.name)
            reason = f"{plan.name} was expected to refuse construction and did not"
            return EngineResult(
                engine=plan.name,
                kind=plan.kind,
                notes=plan.notes,
                refused=True,
                refuse_reason=reason,
                seed=seed,
                checkpoint_epoch=None,
                peak_vram_bytes=None,
                headline={"error": reason},
            )
        except Exception as exc:
            reason = stated_reason(exc, axis="generator", name=plan.name) or str(exc)
            progress(f"[probe] {plan.name} refused at construct: {reason[:180]}")
            result = EngineResult(
                engine=plan.name,
                kind=plan.kind,
                notes=plan.notes,
                refused=True,
                refuse_reason=reason,
                seed=seed,
                checkpoint_epoch=None,
                peak_vram_bytes=None,
                headline={"refused": True, "at": "construct", "reason": reason},
            )
            _write_json(out_dir / f"{plan.name}.json", asdict(result))
            return result

    built: Any = generator
    if built is None:
        try:
            built = GENERATORS.build(plan.name)
        except Exception as exc:
            construct_reason = stated_reason(exc, axis="generator", name=plan.name) or str(exc)
            result = EngineResult(
                engine=plan.name,
                kind=plan.kind,
                notes=plan.notes,
                refused=True,
                refuse_reason=construct_reason,
                seed=seed,
                checkpoint_epoch=None,
                peak_vram_bytes=None,
                headline={"refused": True, "at": "construct", "reason": construct_reason},
            )
            _write_json(out_dir / f"{plan.name}.json", asdict(result))
            return result

    rows: list[ClipResult] = []
    engine_peak = 0
    epoch: int | str | None = None
    params = GenerationParams(width=CANVAS, height=CANVAS, steps=plan.steps)

    for clip in clips:
        indices = plan.frame_indices or (HEADLINE_FRAME_INDEX,)
        for frame_index in indices:
            try:
                frame = load_frame(clip, frame_index)
                bundle = _bundle(frame)
                _reset_peak(device)
                t0 = time.perf_counter()
                predicted = built.generate(bundle, seed=seed, device=device, params=params)
                wall_s = time.perf_counter() - t0
                peak = _peak_bytes(device)
                engine_peak = max(engine_peak, peak)
                aa_run = getattr(built, "last_run", None)
                if isinstance(aa_run, dict) and aa_run.get("peak_vram_bytes"):
                    engine_peak = max(engine_peak, int(aa_run["peak_vram_bytes"]))
                    peak = max(peak, int(aa_run["peak_vram_bytes"]))
                epoch = _epoch_of(built)
                score = _score(frame, np.asarray(predicted))
                row = _clip_row(
                    plan, frame, score, seed=seed, epoch=epoch, peak=peak, wall_s=wall_s
                )
                if not score.differs_from_input:
                    row.error = (
                        f"{plan.name} output is identical to letterboxed appearance "
                        f"on {clip.key} frame {frame_index}"
                    )
                progress(
                    f"[probe] {plan.name} {clip.key} f={frame_index} "
                    f"object={score.object_psnr_db:.2f} frame={score.frame_psnr_db:.2f} "
                    f"bound={row.object_bound} differs={score.differs_from_input} "
                    f"{wall_s:.1f}s"
                )
            except Exception as exc:
                generate_reason = stated_reason(exc, axis="generator", name=plan.name)
                if plan.refuse_at == "generate" and generate_reason:
                    progress(f"[probe] {plan.name} refused at generate: {generate_reason[:180]}")
                    result = EngineResult(
                        engine=plan.name,
                        kind=plan.kind,
                        notes=plan.notes,
                        refused=True,
                        refuse_reason=generate_reason,
                        seed=seed,
                        checkpoint_epoch=epoch,
                        peak_vram_bytes=engine_peak or None,
                        headline={"refused": True, "at": "generate", "reason": generate_reason},
                    )
                    _write_json(out_dir / f"{plan.name}.json", asdict(result))
                    return result
                progress(f"[probe] {plan.name} FAIL {clip.key} f={frame_index}: {exc}")
                row = ClipResult(
                    engine=plan.name,
                    clip_key=clip.key,
                    split=clip.split,
                    frame_index=frame_index,
                    object_psnr_db=None,
                    frame_psnr_db=None,
                    seed=seed,
                    checkpoint_epoch=epoch,
                    peak_vram_bytes=None,
                    wall_s=None,
                    differs_from_input=None,
                    n_object_pixels=None,
                    region_kind=None,
                    object_bound=None,
                    gap_bound=None,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            rows.append(row)
            _write_json(
                out_dir / f"{plan.name}.json",
                asdict(
                    EngineResult(
                        engine=plan.name,
                        kind=plan.kind,
                        notes=plan.notes,
                        refused=False,
                        refuse_reason=None,
                        seed=seed,
                        checkpoint_epoch=epoch,
                        peak_vram_bytes=engine_peak or None,
                        clips=rows,
                        headline=_headline(plan, rows),
                    )
                ),
            )

    if built is not generator:
        del built
        _release(device)
    result = EngineResult(
        engine=plan.name,
        kind=plan.kind,
        notes=plan.notes,
        refused=False,
        refuse_reason=None,
        seed=seed,
        checkpoint_epoch=epoch,
        peak_vram_bytes=engine_peak or None,
        clips=rows,
        headline=_headline(plan, rows),
    )
    result.headline["engine_wall_s"] = time.perf_counter() - started
    _write_json(out_dir / f"{plan.name}.json", asdict(result))
    return result


def drive_all(
    *,
    device: str = DEVICE,
    seed: int = SEED,
    out_dir: Path,
    probe_root: Path | None = None,
    engines: tuple[str, ...] | None = None,
    progress: Any = print,
) -> dict[str, Any]:
    from experiments.probe.bounds import (
        FRAME_MINUS_OBJECT_SMALL_GAP_DB,
        IP_ADAPTER_KNOWN_FLOOR_DB,
        OBJECT_PSNR_ALARM_HIGH_DB,
        OBJECT_PSNR_ALARM_LOW_DB,
        OBJECT_PSNR_EXPECTED_HIGH_DB,
        OBJECT_PSNR_EXPECTED_LOW_DB,
    )

    clips = list_clips(probe_root)
    chosen = tuple(plan_for(name) for name in engines) if engines else PLANS
    summary: dict[str, Any] = {
        "citable": False,
        "seed": seed,
        "device": device,
        "canvas": CANVAS,
        "headline_frame_index": HEADLINE_FRAME_INDEX,
        "n_clips": len(clips),
        "bounds_written_before_generate": {
            "object_alarm_low_db": OBJECT_PSNR_ALARM_LOW_DB,
            "object_expected_low_db": OBJECT_PSNR_EXPECTED_LOW_DB,
            "object_expected_high_db": OBJECT_PSNR_EXPECTED_HIGH_DB,
            "object_alarm_high_db": OBJECT_PSNR_ALARM_HIGH_DB,
            "ip_adapter_known_floor_db": IP_ADAPTER_KNOWN_FLOOR_DB,
            "small_frame_gap_db": FRAME_MINUS_OBJECT_SMALL_GAP_DB,
        },
        "split_note": (
            "All 12 probe clips are from the 5 training-split videos. "
            "Animate-Anyone has also seen both held-out videos (PLAN.md §2.5); "
            "option 2: report AA as in-domain only. A pretrained engine carries "
            "the held-out arm when that arm is run."
        ),
        "engines": {},
    }
    _write_json(out_dir / "summary.json", summary)
    for plan in chosen:
        result = drive_engine(
            plan, clips, device=device, seed=seed, out_dir=out_dir, progress=progress
        )
        summary["engines"][plan.name] = {
            "refused": result.refused,
            "refuse_reason": result.refuse_reason,
            "checkpoint_epoch": result.checkpoint_epoch,
            "peak_vram_bytes": result.peak_vram_bytes,
            "headline": result.headline,
            "n_clip_rows": len(result.clips),
        }
        _write_json(out_dir / "summary.json", summary)
        progress(f"[probe] checkpointed {plan.name} -> {out_dir / 'summary.json'}")
    return summary
