"""Drive one engine over the probe set on the coding task.

Appearance from a keyframe, conditioning from frame N, score against frame N.
Static copy is a permanent arm. Checkpoint after every clip.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.probe.bounds import (
    STATIC_COPY_ALARM_HIGH_DB,
    STATIC_COPY_ALARM_LOW_DB,
    STATIC_COPY_EXPECTED_HIGH_DB,
    STATIC_COPY_EXPECTED_LOW_DB,
    appearance_use_label,
    judge_frame_gap,
    judge_static_copy_object_psnr,
    judge_vs_floor,
)
from experiments.probe.clips import (
    DEFAULT_KEYFRAME,
    DEFAULT_OFFSETS,
    HEADLINE_OFFSET,
    CodingSample,
    ProbeClip,
    ProbeFrame,
    bundle_arrays,
    bundle_coding,
    list_clips,
    load_coding_sample,
    load_frame,
)
from experiments.probe.construct import stated_reason
from experiments.probe.engines import (
    CANVAS,
    DEVICE,
    SEED,
    STATIC_COPY,
    STATIC_COPY_PLAN,
    EnginePlan,
    PLANS,
    plan_for,
)
from experiments.probe.score import ProbeScore, score_generation
from src.components.generation._numpy import prepare_letterboxed
from src.contracts.conditioning import ConditioningBundle, GenerationParams

RANKING_METRIC = "object_psnr_db"
NOT_RANKED = ("self_reconstruction_psnr",)


@dataclass
class ClipResult:
    engine: str
    clip_key: str
    split: str
    appearance_frame_index: int
    target_frame_index: int
    offset: int
    object_psnr_db: float | None
    frame_psnr_db: float | None
    self_reconstruction_psnr: float | None
    seed: int
    checkpoint_epoch: int | str | None
    peak_vram_bytes: int | None
    wall_s: float | None
    differs_from_input: bool | None
    differs_from_reference: bool | None
    n_object_pixels: int | None
    region_kind: str | None
    object_bound: str | None
    gap_bound: str | None
    vs_static_copy_db: float | None = None
    appearance_use: str | None = None
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, default=str))


def _coding_bundle(sample: CodingSample) -> ConditioningBundle:
    payload = bundle_coding(sample)
    return ConditioningBundle(
        appearance=payload["appearance"],
        pose=payload["pose"],
        mask=payload["mask"],
        canny=payload["canny"],
        motion_field=payload["motion_field"],
        frame_index=payload["frame_index"],
        object_id=payload["object_id"],
    )


def _self_bundle(frame: ProbeFrame) -> ConditioningBundle:
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


def _score_coding(sample: CodingSample, predicted: np.ndarray) -> ProbeScore:
    return score_generation(
        sample.reference_rgb,
        predicted,
        object_mask=sample.object_mask,
        canvas_width=CANVAS,
        canvas_height=CANVAS,
        appearance=sample.appearance_rgb,
    )


def predict_static_copy(appearance: np.ndarray, canvas_width: int, canvas_height: int) -> np.ndarray:
    """Paste the keyframe onto the generation canvas. No model."""
    prepared = prepare_letterboxed(appearance, None, canvas_width, canvas_height)
    return np.asarray(prepared["appearance"])


def _self_recon_offset(offsets: tuple[int, ...]) -> int | None:
    if not offsets:
        return None
    if HEADLINE_OFFSET in offsets:
        return HEADLINE_OFFSET
    return offsets[0]


def _apply_floor(row: ClipResult, floor_by_key: Mapping[tuple[str, int], float] | None) -> None:
    if row.object_psnr_db is None or row.engine == STATIC_COPY:
        if row.engine == STATIC_COPY and row.object_psnr_db is not None:
            row.appearance_use = "floor"
            row.object_bound = judge_static_copy_object_psnr(row.object_psnr_db).status
        return
    if floor_by_key is None:
        return
    floor = floor_by_key.get((row.clip_key, row.offset))
    if floor is None:
        return
    row.vs_static_copy_db = float(row.object_psnr_db) - float(floor)
    row.appearance_use = appearance_use_label(float(row.object_psnr_db), float(floor))
    row.object_bound = judge_vs_floor(float(row.object_psnr_db), float(floor)).status


def _clip_row(
    plan: EnginePlan,
    sample: CodingSample,
    score: ProbeScore,
    *,
    seed: int,
    epoch: int | str | None,
    peak: int | None,
    wall_s: float,
    self_reconstruction_psnr: float | None,
    floor_by_key: Mapping[tuple[str, int], float] | None,
) -> ClipResult:
    gap = judge_frame_gap(score.frame_psnr_db, score.object_psnr_db)
    row = ClipResult(
        engine=plan.name,
        clip_key=sample.key,
        split=sample.split,
        appearance_frame_index=sample.appearance_frame_index,
        target_frame_index=sample.target_frame_index,
        offset=sample.offset,
        object_psnr_db=score.object_psnr_db,
        frame_psnr_db=score.frame_psnr_db,
        self_reconstruction_psnr=self_reconstruction_psnr,
        seed=seed,
        checkpoint_epoch=epoch,
        peak_vram_bytes=peak,
        wall_s=wall_s,
        differs_from_input=score.differs_from_input,
        differs_from_reference=score.differs_from_reference,
        n_object_pixels=score.n_object_pixels,
        region_kind=score.region_kind,
        object_bound=None,
        gap_bound=gap.status,
    )
    _apply_floor(row, floor_by_key)
    return row


def _error_row(
    plan: EnginePlan,
    clip: ProbeClip,
    *,
    keyframe_index: int,
    offset: int,
    seed: int,
    epoch: int | str | None,
    error: str,
) -> ClipResult:
    return ClipResult(
        engine=plan.name,
        clip_key=clip.key,
        split=clip.split,
        appearance_frame_index=keyframe_index,
        target_frame_index=keyframe_index + offset,
        offset=offset,
        object_psnr_db=None,
        frame_psnr_db=None,
        self_reconstruction_psnr=None,
        seed=seed,
        checkpoint_epoch=epoch,
        peak_vram_bytes=None,
        wall_s=None,
        differs_from_input=None,
        differs_from_reference=None,
        n_object_pixels=None,
        region_kind=None,
        object_bound=None,
        gap_bound=None,
        error=error,
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def headline_rows(rows: list[ClipResult], offset: int) -> list[ClipResult]:
    matched = [
        row
        for row in rows
        if row.error is None and row.offset == offset and row.object_psnr_db is not None
    ]
    if matched:
        return matched
    return [row for row in rows if row.error is None and row.object_psnr_db is not None]


def _headline(
    plan: EnginePlan,
    rows: list[ClipResult],
    *,
    floor_headline: float | None,
    keyframe_index: int,
    headline_offset: int,
) -> dict[str, Any]:
    matched = headline_rows(rows, headline_offset)
    object_values = [float(row.object_psnr_db) for row in matched if row.object_psnr_db is not None]
    frame_values = [float(row.frame_psnr_db) for row in matched if row.frame_psnr_db is not None]
    self_values = [
        float(row.self_reconstruction_psnr)
        for row in matched
        if row.self_reconstruction_psnr is not None
    ]
    mean_object = _mean(object_values)
    mean_frame = _mean(frame_values)
    payload: dict[str, Any] = {
        "n": len(matched),
        "keyframe_index": keyframe_index,
        "offset": headline_offset,
        "object_psnr_db": mean_object,
        "frame_psnr_db": mean_frame,
        "self_reconstruction_psnr": _mean(self_values),
        "ranking_uses": RANKING_METRIC,
        "ranking_ignores": list(NOT_RANKED),
        "identity_failures": [row.clip_key for row in matched if row.differs_from_input is False],
        "mean_wall_s": _mean([float(row.wall_s or 0.0) for row in matched]),
        "peak_vram_bytes": max((row.peak_vram_bytes or 0) for row in matched) if matched else None,
        "by_offset": _by_offset(rows),
    }
    if mean_object is None:
        return payload
    if plan.name == STATIC_COPY:
        verdict = judge_static_copy_object_psnr(mean_object)
        payload["appearance_use"] = "floor"
        payload["object_bound"] = verdict.status
        payload["object_bound_note"] = verdict.note
    elif floor_headline is not None:
        payload["vs_static_copy_db"] = mean_object - floor_headline
        payload["appearance_use"] = appearance_use_label(mean_object, floor_headline)
        verdict = judge_vs_floor(mean_object, floor_headline)
        payload["object_bound"] = verdict.status
        payload["object_bound_note"] = verdict.note
    if mean_frame is not None:
        gap = judge_frame_gap(mean_frame, mean_object)
        payload["gap_bound"] = gap.status
        payload["gap_db"] = mean_frame - mean_object
    return payload


def _by_offset(rows: list[ClipResult]) -> dict[str, Any]:
    grouped: dict[int, list[ClipResult]] = {}
    for row in rows:
        if row.error is not None or row.object_psnr_db is None:
            continue
        grouped.setdefault(row.offset, []).append(row)
    out: dict[str, Any] = {}
    for offset, group in sorted(grouped.items()):
        object_values = [float(row.object_psnr_db) for row in group if row.object_psnr_db is not None]
        frame_values = [float(row.frame_psnr_db) for row in group if row.frame_psnr_db is not None]
        out[str(offset)] = {
            "n": len(group),
            "object_psnr_db": _mean(object_values),
            "frame_psnr_db": _mean(frame_values),
        }
    return out


def rank_engines(engine_summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Order engines by coding-task object PSNR. Never reads self-reconstruction."""
    scored: list[tuple[str, float]] = []
    for name, data in engine_summaries.items():
        if name == STATIC_COPY:
            continue
        headline = data.get("headline") if isinstance(data.get("headline"), dict) else data
        if not isinstance(headline, Mapping):
            continue
        value = headline.get(RANKING_METRIC)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            scored.append((name, float(value)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in scored]


def _floor_lookup(rows: list[ClipResult]) -> dict[tuple[str, int], float]:
    return {
        (row.clip_key, row.offset): float(row.object_psnr_db)
        for row in rows
        if row.error is None and row.object_psnr_db is not None
    }


def _measure_self_recon(
    *,
    clip: ProbeClip,
    target_index: int,
    canvas: int,
    generator: Any | None,
    seed: int,
    device: str,
    params: GenerationParams,
) -> float:
    frame = load_frame(clip, target_index)
    if generator is None:
        predicted = predict_static_copy(frame.appearance_rgb, canvas, canvas)
    else:
        predicted = generator.generate(_self_bundle(frame), seed=seed, device=device, params=params)
    score = score_generation(
        frame.appearance_rgb,
        np.asarray(predicted),
        object_mask=frame.object_mask,
        canvas_width=canvas,
        canvas_height=canvas,
        appearance=frame.appearance_rgb,
    )
    return score.object_psnr_db


def _refused(
    plan: EnginePlan,
    *,
    seed: int,
    reason: str,
    epoch: int | str | None = None,
    peak: int | None = None,
    at: str,
) -> EngineResult:
    return EngineResult(
        engine=plan.name,
        kind=plan.kind,
        notes=plan.notes,
        refused=True,
        refuse_reason=reason,
        seed=seed,
        checkpoint_epoch=epoch,
        peak_vram_bytes=peak,
        headline={"refused": True, "at": at, "reason": reason},
    )


def drive_engine(
    plan: EnginePlan,
    clips: tuple[ProbeClip, ...],
    *,
    device: str,
    seed: int,
    out_dir: Path,
    generator: Any | None = None,
    keyframe_index: int = DEFAULT_KEYFRAME,
    offsets: tuple[int, ...] | None = None,
    floor_by_key: Mapping[tuple[str, int], float] | None = None,
    floor_headline: float | None = None,
    self_recon: bool = True,
    progress: Any = print,
) -> EngineResult:
    """Run ``plan`` over ``clips``. ``generator`` is injected in tests."""
    from src.components.generation import REGISTRY as GENERATORS

    started = time.perf_counter()
    used_offsets = offsets if offsets is not None else plan.offsets
    progress(
        f"[probe] {plan.name} kind={plan.kind} clips={len(clips)} "
        f"keyframe={keyframe_index} offsets={used_offsets} device={device}"
    )
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
            result = _refused(plan, seed=seed, reason=reason, at="construct")
            _write_json(out_dir / f"{plan.name}.json", asdict(result))
            return result

    built: Any = generator
    if built is None and plan.name != STATIC_COPY:
        try:
            built = GENERATORS.build(plan.name)
        except Exception as exc:
            construct_reason = stated_reason(exc, axis="generator", name=plan.name) or str(exc)
            result = _refused(plan, seed=seed, reason=construct_reason, at="construct")
            _write_json(out_dir / f"{plan.name}.json", asdict(result))
            return result

    rows: list[ClipResult] = []
    engine_peak = 0
    epoch: int | str | None = None
    params = GenerationParams(width=CANVAS, height=CANVAS, steps=plan.steps)
    recon_at = _self_recon_offset(used_offsets) if self_recon else None

    for clip in clips:
        for offset in used_offsets:
            try:
                sample = load_coding_sample(clip, keyframe_index, offset)
                _reset_peak(device)
                t0 = time.perf_counter()
                if plan.name == STATIC_COPY or built is None:
                    predicted = predict_static_copy(sample.appearance_rgb, CANVAS, CANVAS)
                else:
                    predicted = built.generate(
                        _coding_bundle(sample), seed=seed, device=device, params=params
                    )
                wall_s = time.perf_counter() - t0
                peak = _peak_bytes(device)
                engine_peak = max(engine_peak, peak)
                aa_run = getattr(built, "last_run", None) if built is not None else None
                if isinstance(aa_run, dict) and aa_run.get("peak_vram_bytes"):
                    engine_peak = max(engine_peak, int(aa_run["peak_vram_bytes"]))
                    peak = max(peak, int(aa_run["peak_vram_bytes"]))
                epoch = _epoch_of(built) if built is not None else None
                score = _score_coding(sample, np.asarray(predicted))
                self_psnr: float | None = None
                if recon_at is not None and offset == recon_at:
                    self_psnr = _measure_self_recon(
                        clip=clip,
                        target_index=sample.target_frame_index,
                        canvas=CANVAS,
                        generator=None if plan.name == STATIC_COPY else built,
                        seed=seed,
                        device=device,
                        params=params,
                    )
                row = _clip_row(
                    plan,
                    sample,
                    score,
                    seed=seed,
                    epoch=epoch,
                    peak=peak,
                    wall_s=wall_s,
                    self_reconstruction_psnr=self_psnr,
                    floor_by_key=floor_by_key,
                )
                progress(
                    f"[probe] {plan.name} {clip.key} keyframe={keyframe_index} "
                    f"offset={offset} object={score.object_psnr_db:.2f} "
                    f"frame={score.frame_psnr_db:.2f} "
                    f"use={row.appearance_use} bound={row.object_bound} "
                    f"{wall_s:.1f}s"
                )
            except Exception as exc:
                generate_reason = stated_reason(exc, axis="generator", name=plan.name)
                if plan.refuse_at == "generate" and generate_reason:
                    progress(f"[probe] {plan.name} refused at generate: {generate_reason[:180]}")
                    result = _refused(
                        plan,
                        seed=seed,
                        reason=generate_reason,
                        epoch=epoch,
                        peak=engine_peak or None,
                        at="generate",
                    )
                    _write_json(out_dir / f"{plan.name}.json", asdict(result))
                    return result
                progress(f"[probe] {plan.name} FAIL {clip.key} offset={offset}: {exc}")
                row = _error_row(
                    plan,
                    clip,
                    keyframe_index=keyframe_index,
                    offset=offset,
                    seed=seed,
                    epoch=epoch,
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
                        headline=_headline(
                            plan,
                            rows,
                            floor_headline=floor_headline,
                            keyframe_index=keyframe_index,
                            headline_offset=HEADLINE_OFFSET,
                        ),
                    )
                ),
            )

    if built is not generator and built is not None:
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
        headline=_headline(
            plan,
            rows,
            floor_headline=floor_headline,
            keyframe_index=keyframe_index,
            headline_offset=HEADLINE_OFFSET,
        ),
    )
    result.headline["engine_wall_s"] = time.perf_counter() - started
    _write_json(out_dir / f"{plan.name}.json", asdict(result))
    return result


def _engine_summary(result: EngineResult) -> dict[str, Any]:
    return {
        "refused": result.refused,
        "refuse_reason": result.refuse_reason,
        "checkpoint_epoch": result.checkpoint_epoch,
        "peak_vram_bytes": result.peak_vram_bytes,
        "headline": result.headline,
        "n_clip_rows": len(result.clips),
    }


def drive_all(
    *,
    device: str = DEVICE,
    seed: int = SEED,
    out_dir: Path,
    probe_root: Path | None = None,
    engines: tuple[str, ...] | None = None,
    generators: Mapping[str, Any] | None = None,
    keyframe_index: int = DEFAULT_KEYFRAME,
    offsets: tuple[int, ...] | None = None,
    self_recon: bool = True,
    progress: Any = print,
) -> dict[str, Any]:
    used_offsets = offsets if offsets is not None else DEFAULT_OFFSETS
    clips = list_clips(probe_root)
    chosen = tuple(plan_for(name) for name in engines) if engines else PLANS
    bounds_record = {
        "static_copy_alarm_low_db": STATIC_COPY_ALARM_LOW_DB,
        "static_copy_expected_low_db": STATIC_COPY_EXPECTED_LOW_DB,
        "static_copy_expected_high_db": STATIC_COPY_EXPECTED_HIGH_DB,
        "static_copy_alarm_high_db": STATIC_COPY_ALARM_HIGH_DB,
        "anchored_on": "static-copy floor, not absolute engine dB",
        "task": "appearance from keyframe, score against later frame",
    }
    summary: dict[str, Any] = {
        "citable": False,
        "seed": seed,
        "device": device,
        "canvas": CANVAS,
        "keyframe_index": keyframe_index,
        "offsets": list(used_offsets),
        "headline_offset": HEADLINE_OFFSET,
        "ranking_uses": RANKING_METRIC,
        "ranking_ignores": list(NOT_RANKED),
        "n_clips": len(clips),
        "bounds_written_before_generate": bounds_record,
        "split_note": (
            "All 12 probe clips are from the 5 training-split videos. "
            "Animate-Anyone has also seen both held-out videos (PLAN.md §2.5); "
            "option 2: report AA as in-domain only. A pretrained engine carries "
            "the held-out arm when that arm is run."
        ),
        "static_copy": {},
        "engines": {},
        "rank": [],
    }
    _write_json(out_dir / "summary.json", summary)

    static_plan = EnginePlan(
        name=STATIC_COPY_PLAN.name,
        kind=STATIC_COPY_PLAN.kind,
        offsets=used_offsets,
        notes=STATIC_COPY_PLAN.notes,
    )
    static = drive_engine(
        static_plan,
        clips,
        device=device,
        seed=seed,
        out_dir=out_dir,
        generator=None,
        keyframe_index=keyframe_index,
        offsets=used_offsets,
        self_recon=self_recon,
        progress=progress,
    )
    floor_by_key = _floor_lookup(static.clips)
    floor_headline = static.headline.get("object_psnr_db")
    if isinstance(floor_headline, (int, float)):
        floor_headline_value: float | None = float(floor_headline)
    else:
        floor_headline_value = None
    summary["static_copy"] = _engine_summary(static)
    summary["engines"][STATIC_COPY] = _engine_summary(static)
    _write_json(out_dir / "summary.json", summary)

    for plan in chosen:
        if plan.name == STATIC_COPY:
            continue
        injected = None if generators is None else generators.get(plan.name)
        result = drive_engine(
            plan,
            clips,
            device=device,
            seed=seed,
            out_dir=out_dir,
            generator=injected,
            keyframe_index=keyframe_index,
            offsets=used_offsets,
            floor_by_key=floor_by_key,
            floor_headline=floor_headline_value,
            self_recon=self_recon,
            progress=progress,
        )
        summary["engines"][plan.name] = _engine_summary(result)
        summary["rank"] = rank_engines(summary["engines"])
        _write_json(out_dir / "summary.json", summary)
        progress(f"[probe] checkpointed {plan.name} -> {out_dir / 'summary.json'}")
    summary["rank"] = rank_engines(summary["engines"])
    _write_json(out_dir / "summary.json", summary)
    return summary
