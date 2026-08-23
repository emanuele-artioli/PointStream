"""Drive one engine over the probe set on the coding task.

Appearance from a keyframe, conditioning from frame N, score against frame N.
Checkpoint after every clip.

**Two things this harness does that the one it replaces did not.**

*Clip mode.* A plan marked ``sequence=True`` is driven once per clip through
``generate_sequence`` over a contiguous run of frames. Animate-Anyone carries a
motion module and was evaluated one frame at a time for three rounds
(``PLAN.md`` §2.7); a sequence plan whose backend has no ``generate_sequence``
now fails loudly rather than falling back to the single-frame path.

*Two baselines, always.* ``static-copy`` pastes this clip's own keyframe — the
floor. ``unrelated-image`` pastes another clip's keyframe — the null control.
Both run before any engine, and the run refuses to rank anything if the right
player and the wrong player do not separate on the metric. Every wrong
conclusion in this project was a pleasing number reported before its control.

Ranking is on **LPIPS**, lower better, with PSNR reported beside it: the usable
PSNR range on this task is ~11-21 dB against a ~2 dB per-clip sd, and the
subfield rejects PSNR for generatively reconstructed content (``PLAN.md`` §2.5).
"""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.probe.bounds import (
    STATIC_COPY_ALARM_HIGH_DB,
    STATIC_COPY_ALARM_LOW_DB,
    STATIC_COPY_EXPECTED_HIGH_DB,
    STATIC_COPY_EXPECTED_LOW_DB,
    STATIC_COPY_LPIPS_ALARM_HIGH,
    STATIC_COPY_LPIPS_ALARM_LOW,
    STATIC_COPY_LPIPS_EXPECTED_HIGH,
    STATIC_COPY_LPIPS_EXPECTED_LOW,
    UNRELATED_LPIPS_ALARM_LOW,
    UNRELATED_LPIPS_EXPECTED_HIGH,
    UNRELATED_LPIPS_EXPECTED_LOW,
    appearance_use_label,
    judge_engine_lpips,
    judge_frame_gap,
    judge_null_separation,
    judge_static_copy_clip,
    judge_static_copy_lpips,
    judge_static_copy_object_psnr,
    judge_unrelated_lpips,
    judge_vs_floor,
)
from experiments.probe.clips import (
    CLIP_MODE_OFFSETS,
    DEFAULT_KEYFRAME,
    CodingSample,
    ProbeClip,
    ProbeFrame,
    bundle_arrays,
    bundle_coding,
    list_clips,
    load_coding_sample,
    load_coding_sequence,
    load_frame,
    with_appearance,
)
from experiments.probe.construct import stated_reason
from experiments.probe.engines import (
    BASELINES,
    CANVAS,
    DEVICE,
    SEED,
    STATIC_COPY,
    STATIC_COPY_PLAN,
    UNRELATED_IMAGE,
    UNRELATED_IMAGE_PLAN,
    EnginePlan,
    PLANS,
    plan_for,
)
from experiments.probe.score import ProbeScore, score_generation
from src.components.generation._numpy import prepare_letterboxed
from src.contracts.conditioning import ConditioningBundle, GenerationParams

RANKING_METRIC = "object_lpips"
RANKING_LOWER_IS_BETTER = True
REPORTED_BESIDE = ("object_psnr_db", "frame_psnr_db", "frame_lpips")
NOT_RANKED = ("self_reconstruction_psnr",)

#: Offsets 1-8 are contiguous, which is what makes a clip a clip.
HEADLINE_OFFSET_CLIP_MODE = CLIP_MODE_OFFSETS[-1]

_LPIPS_CACHE: dict[str, Any] = {}


def build_lpips(device: str) -> Any | None:
    """The calibrated LPIPS backend, or None with a printed reason.

    A probe that cannot build its ranking metric says so and reports PSNR
    alone. It does not quietly rank on the fallback.
    """
    if device in _LPIPS_CACHE:
        return _LPIPS_CACHE[device]
    try:
        from src.components.metrics.lpips import LpipsMetric

        metric = LpipsMetric(device=device)
        # Touch it now: a lazy import failure at frame 300 of a run is worse
        # than the same failure before the first generation.
        probe = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        metric.score(probe, probe)
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"[probe] LPIPS unavailable on {device}: {type(exc).__name__}: {exc}")
        _LPIPS_CACHE[device] = None
        return None
    _LPIPS_CACHE[device] = metric
    return metric


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
    object_lpips: float | None = None
    frame_lpips: float | None = None
    lpips_box_padded: bool | None = None
    lpips_bound: str | None = None
    drive_mode: str = "frame"
    appearance_source: str | None = None
    vs_static_copy_db: float | None = None
    vs_static_copy_lpips: float | None = None
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
    drive_mode: str = "frame"
    clips: list[ClipResult] = field(default_factory=list)
    headline: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Anchors:
    """Both baselines, per ``(clip, offset)`` and as headline means.

    A number is reported with the range it sits in, never alone. These are
    what supply that range, measured in the same session on the same clips.
    """

    floor_psnr: Mapping[tuple[str, int], float]
    floor_lpips: Mapping[tuple[str, int], float]
    null_lpips: Mapping[tuple[str, int], float]
    floor_psnr_mean: float | None = None
    floor_lpips_mean: float | None = None
    null_lpips_mean: float | None = None

    @classmethod
    def empty(cls) -> Anchors:
        return cls(floor_psnr={}, floor_lpips={}, null_lpips={})


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


#: How a wrong appearance is chosen. The two answers measure different things.
#:
#: ``different-video`` is the default and the stronger null: a different court,
#: kit and lighting as well as a different player.
#:
#: ``same-video`` is the *tighter* control. It holds the broadcast fixed and
#: changes only the track, so a delta that survives it is not the model copying
#: scene colour. Read it as a lower bound and nothing more: a tennis video has
#: two players, so a same-video donor is sometimes the same person, and a small
#: delta here is ambiguous between "copies the scene" and "drew the right man".
DONOR_MODES = ("different-video", "same-video")


def donor_for(
    clips: Sequence[ProbeClip], index: int, *, mode: str = "different-video"
) -> ProbeClip:
    """The clip whose keyframe stands in for a wrong appearance."""
    if mode not in DONOR_MODES:
        raise ValueError(f"donor mode must be one of {DONOR_MODES}; got {mode!r}")
    if len(clips) < 2:
        raise ValueError(
            "the null control needs a second clip to borrow an appearance from. "
            "With one clip it would paste the right player, which is not a control."
        )
    this = clips[index]
    n = len(clips)
    want_same = mode == "same-video"
    for step in range(1, n):
        candidate = clips[(index + step) % n]
        if (candidate.video == this.video) == want_same:
            return candidate
    return clips[(index + 1) % n]


def donor_appearances(
    clips: Sequence[ProbeClip],
    keyframe_index: int,
    *,
    mode: str = "different-video",
) -> dict[str, tuple[str, np.ndarray]]:
    """``clip.key -> (donor key, that donor's keyframe RGB)``. Deterministic."""
    out: dict[str, tuple[str, np.ndarray]] = {}
    for index, clip in enumerate(clips):
        donor = donor_for(clips, index, mode=mode)
        out[clip.key] = (donor.key, load_frame(donor, keyframe_index).appearance_rgb)
    return out


def _score_coding(
    sample: CodingSample, predicted: np.ndarray, *, lpips_metric: Any = None
) -> ProbeScore:
    return score_generation(
        sample.reference_rgb,
        predicted,
        object_mask=sample.object_mask,
        canvas_width=CANVAS,
        canvas_height=CANVAS,
        appearance=sample.appearance_rgb,
        lpips_metric=lpips_metric,
    )


def predict_static_copy(appearance: np.ndarray, canvas_width: int, canvas_height: int) -> np.ndarray:
    """Paste the keyframe onto the generation canvas. No model."""
    prepared = prepare_letterboxed(appearance, None, canvas_width, canvas_height)
    return np.asarray(prepared["appearance"])


def _self_recon_offset(offsets: tuple[int, ...], headline_offset: int) -> int | None:
    if not offsets:
        return None
    if headline_offset in offsets:
        return headline_offset
    return offsets[0]


def _apply_anchors(row: ClipResult, anchors: Anchors) -> None:
    """Attach the two baselines to one engine row. Baselines anchor themselves."""
    key = (row.clip_key, row.offset)
    if row.engine == STATIC_COPY:
        row.appearance_use = "floor"
        if row.object_psnr_db is not None:
            row.object_bound = judge_static_copy_clip(row.object_psnr_db).status
        return
    if row.engine == UNRELATED_IMAGE:
        row.appearance_use = "null-control"
        return
    floor_psnr = anchors.floor_psnr.get(key)
    if row.object_psnr_db is not None and floor_psnr is not None:
        row.vs_static_copy_db = float(row.object_psnr_db) - float(floor_psnr)
        row.appearance_use = appearance_use_label(float(row.object_psnr_db), float(floor_psnr))
        row.object_bound = judge_vs_floor(float(row.object_psnr_db), float(floor_psnr)).status
    floor_lpips = anchors.floor_lpips.get(key)
    null_lpips = anchors.null_lpips.get(key)
    if row.object_lpips is not None and floor_lpips is not None:
        row.vs_static_copy_lpips = float(row.object_lpips) - float(floor_lpips)
        if null_lpips is not None:
            row.lpips_bound = judge_engine_lpips(
                float(row.object_lpips), float(floor_lpips), float(null_lpips)
            ).status


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
    anchors: Anchors,
    drive_mode: str,
    appearance_source: str | None,
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
        object_lpips=score.object_lpips,
        frame_lpips=score.frame_lpips,
        lpips_box_padded=score.lpips_box_padded,
        drive_mode=drive_mode,
        appearance_source=appearance_source,
    )
    _apply_anchors(row, anchors)
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
    drive_mode: str = "frame",
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
        drive_mode=drive_mode,
        error=error,
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _stderr(values: list[float]) -> float | None:
    """Standard error of the mean. A comparison without one is not a finding."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def scoring_rows(rows: list[ClipResult]) -> list[ClipResult]:
    """Every successful row. Clip mode ranks over the whole clip, not one offset."""
    return [row for row in rows if row.error is None and row.object_psnr_db is not None]


def _headline(
    plan: EnginePlan,
    rows: list[ClipResult],
    *,
    anchors: Anchors,
    keyframe_index: int,
    offsets: tuple[int, ...],
    drive_mode: str,
) -> dict[str, Any]:
    matched = scoring_rows(rows)
    object_values = [float(row.object_psnr_db) for row in matched if row.object_psnr_db is not None]
    frame_values = [float(row.frame_psnr_db) for row in matched if row.frame_psnr_db is not None]
    lpips_values = [float(row.object_lpips) for row in matched if row.object_lpips is not None]
    frame_lpips_values = [float(row.frame_lpips) for row in matched if row.frame_lpips is not None]
    self_values = [
        float(row.self_reconstruction_psnr)
        for row in matched
        if row.self_reconstruction_psnr is not None
    ]
    mean_object = _mean(object_values)
    mean_frame = _mean(frame_values)
    mean_lpips = _mean(lpips_values)
    payload: dict[str, Any] = {
        "n": len(matched),
        "n_clips": len({row.clip_key for row in matched}),
        "keyframe_index": keyframe_index,
        "offsets": list(offsets),
        "drive_mode": drive_mode,
        "object_lpips": mean_lpips,
        "object_lpips_stderr": _stderr(lpips_values),
        "frame_lpips": _mean(frame_lpips_values),
        "object_psnr_db": mean_object,
        "object_psnr_db_stderr": _stderr(object_values),
        "frame_psnr_db": mean_frame,
        "self_reconstruction_psnr": _mean(self_values),
        "ranking_uses": RANKING_METRIC,
        "ranking_lower_is_better": RANKING_LOWER_IS_BETTER,
        "reported_beside": list(REPORTED_BESIDE),
        "ranking_ignores": list(NOT_RANKED),
        "lpips_boxes_padded": sum(1 for row in matched if row.lpips_box_padded),
        "mean_wall_s": _mean([float(row.wall_s or 0.0) for row in matched]),
        "peak_vram_bytes": max((row.peak_vram_bytes or 0) for row in matched) if matched else None,
        "by_offset": _by_offset(rows),
    }
    if plan.name not in BASELINES:
        payload["identity_failures"] = [
            row.clip_key for row in matched if row.differs_from_input is False
        ]
    if mean_object is None:
        return payload
    if plan.name == STATIC_COPY:
        payload["appearance_use"] = "floor"
        verdict = judge_static_copy_object_psnr(mean_object)
        payload["object_bound"] = verdict.status
        payload["object_bound_note"] = verdict.note
        if mean_lpips is not None:
            lpips_verdict = judge_static_copy_lpips(mean_lpips)
            payload["lpips_bound"] = lpips_verdict.status
            payload["lpips_bound_note"] = lpips_verdict.note
    elif plan.name == UNRELATED_IMAGE:
        payload["appearance_use"] = "null-control"
        if mean_lpips is not None:
            lpips_verdict = judge_unrelated_lpips(mean_lpips)
            payload["lpips_bound"] = lpips_verdict.status
            payload["lpips_bound_note"] = lpips_verdict.note
    else:
        if anchors.floor_psnr_mean is not None:
            payload["vs_static_copy_db"] = mean_object - anchors.floor_psnr_mean
            payload["appearance_use"] = appearance_use_label(
                mean_object, anchors.floor_psnr_mean
            )
            verdict = judge_vs_floor(mean_object, anchors.floor_psnr_mean)
            payload["object_bound"] = verdict.status
            payload["object_bound_note"] = verdict.note
        if (
            mean_lpips is not None
            and anchors.floor_lpips_mean is not None
            and anchors.null_lpips_mean is not None
        ):
            payload["vs_static_copy_lpips"] = mean_lpips - anchors.floor_lpips_mean
            payload["anchors"] = {
                "static_copy_lpips": anchors.floor_lpips_mean,
                "unrelated_image_lpips": anchors.null_lpips_mean,
            }
            lpips_verdict = judge_engine_lpips(
                mean_lpips, anchors.floor_lpips_mean, anchors.null_lpips_mean
            )
            payload["lpips_bound"] = lpips_verdict.status
            payload["lpips_bound_note"] = lpips_verdict.note
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
        lpips_values = [float(row.object_lpips) for row in group if row.object_lpips is not None]
        out[str(offset)] = {
            "n": len(group),
            "object_lpips": _mean(lpips_values),
            "object_lpips_stderr": _stderr(lpips_values),
            "object_psnr_db": _mean(object_values),
            "frame_psnr_db": _mean(frame_values),
        }
    return out


def rank_engines(engine_summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Order engines by coding-task LPIPS, lower first. Baselines are not ranked.

    Falls back to nothing: an engine with no LPIPS is left out of the ranking
    rather than ranked on PSNR, because those two orders are not the same order
    and mixing them is how a table starts lying.
    """
    scored: list[tuple[str, float]] = []
    for name, data in engine_summaries.items():
        if name in BASELINES:
            continue
        headline = data.get("headline") if isinstance(data.get("headline"), dict) else data
        if not isinstance(headline, Mapping):
            continue
        value = headline.get(RANKING_METRIC)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            scored.append((name, float(value)))
    scored.sort(key=lambda item: item[1], reverse=not RANKING_LOWER_IS_BETTER)
    return [name for name, _ in scored]


def _lookup(rows: list[ClipResult], attribute: str) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.error is not None:
            continue
        value = getattr(row, attribute)
        if value is None:
            continue
        out[(row.clip_key, row.offset)] = float(value)
    return out


def _measure_self_recon(
    *,
    clip: ProbeClip,
    target_index: int,
    canvas: int,
    generator: Any | None,
    seed: int,
    device: str,
    params: GenerationParams,
    lpips_metric: Any = None,
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
        lpips_metric=lpips_metric,
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
        drive_mode="clip" if plan.sequence else "frame",
        headline={"refused": True, "at": at, "reason": reason},
    )


def _require_sequence_path(plan: EnginePlan, built: Any) -> None:
    """A temporal plan must reach the temporal path, or the run stops.

    Falling back to ``generate`` here is exactly the fault that produced three
    rounds of void Animate-Anyone verdicts. A silent fallback is worse than a
    crash because it produces a number.
    """
    if not plan.sequence or plan.name in BASELINES:
        return
    if built is None or not hasattr(built, "generate_sequence"):
        raise RuntimeError(
            f"{plan.name} is declared temporal (sequence=True) but its backend has "
            "no generate_sequence. Driving it frame-by-frame is what voided every "
            "Animate-Anyone number before 2026-08-23; refusing rather than falling back."
        )


def _generate_clip(
    plan: EnginePlan,
    built: Any,
    samples: Sequence[CodingSample],
    *,
    seed: int,
    device: str,
    params: GenerationParams,
) -> tuple[list[np.ndarray], float]:
    """One ``generate_sequence`` call for the whole clip. Returns frames and wall time."""
    bundles = [_coding_bundle(sample) for sample in samples]
    started = time.perf_counter()
    produced = built.generate_sequence(bundles, seed=seed, device=device, params=params)
    wall_s = time.perf_counter() - started
    frames = [np.asarray(frame) for frame in produced]
    if len(frames) != len(samples):
        raise RuntimeError(
            f"{plan.name} returned {len(frames)} frames for {len(samples)} bundles. "
            "A clip-mode arm that drops or pads frames is not scoring what it "
            "was asked to score."
        )
    return frames, wall_s


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
    anchors: Anchors | None = None,
    self_recon: bool = False,
    lpips_metric: Any = None,
    donors: Mapping[str, tuple[str, np.ndarray]] | None = None,
    progress: Any = print,
) -> EngineResult:
    """Run ``plan`` over ``clips``. ``generator`` is injected in tests."""
    from src.components.generation import REGISTRY as GENERATORS

    started = time.perf_counter()
    used_offsets = offsets if offsets is not None else plan.offsets
    used_anchors = anchors if anchors is not None else Anchors.empty()
    drive_mode = "clip" if plan.sequence else "frame"
    headline_offset = _self_recon_offset(used_offsets, HEADLINE_OFFSET_CLIP_MODE)
    progress(
        f"[probe] {plan.name} kind={plan.kind} mode={drive_mode} clips={len(clips)} "
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
                drive_mode=drive_mode,
                headline={"error": reason},
            )
        except Exception as exc:
            reason = stated_reason(exc, axis="generator", name=plan.name) or str(exc)
            progress(f"[probe] {plan.name} refused at construct: {reason[:180]}")
            result = _refused(plan, seed=seed, reason=reason, at="construct")
            _write_json(out_dir / f"{plan.name}.json", asdict(result))
            return result

    built: Any = generator
    if built is None and plan.name not in BASELINES:
        try:
            built = GENERATORS.build(plan.name)
        except Exception as exc:
            construct_reason = stated_reason(exc, axis="generator", name=plan.name) or str(exc)
            result = _refused(plan, seed=seed, reason=construct_reason, at="construct")
            _write_json(out_dir / f"{plan.name}.json", asdict(result))
            return result
    if plan.name not in BASELINES:
        _require_sequence_path(plan, built)

    rows: list[ClipResult] = []
    engine_peak = 0
    epoch: int | str | None = None
    params = GenerationParams(width=CANVAS, height=CANVAS, steps=plan.steps)
    recon_at = headline_offset if (self_recon and not plan.sequence) else None

    def checkpoint() -> None:
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
                    drive_mode=drive_mode,
                    clips=rows,
                    headline=_headline(
                        plan,
                        rows,
                        anchors=used_anchors,
                        keyframe_index=keyframe_index,
                        offsets=used_offsets,
                        drive_mode=drive_mode,
                    ),
                )
            ),
        )

    for clip in clips:
        donor_key: str | None = None
        donor_rgb: np.ndarray | None = None
        if plan.name == UNRELATED_IMAGE:
            if donors is None or clip.key not in donors:
                raise RuntimeError(
                    "the unrelated-image null control needs a donor keyframe per clip; "
                    "none was supplied. A control that silently pastes the right "
                    "player is not a control."
                )
            donor_key, donor_rgb = donors[clip.key]

        if plan.sequence:
            try:
                samples = load_coding_sequence(clip, keyframe_index, used_offsets)
                _reset_peak(device)
                frames, clip_wall = _generate_clip(
                    plan, built, samples, seed=seed, device=device, params=params
                )
                peak = _peak_bytes(device)
                last_run = getattr(built, "last_run", None)
                if isinstance(last_run, dict) and last_run.get("peak_vram_bytes"):
                    peak = max(peak, int(last_run["peak_vram_bytes"]))
                engine_peak = max(engine_peak, peak)
                epoch = _epoch_of(built)
                per_frame = clip_wall / max(len(samples), 1)
                for sample, predicted in zip(samples, frames):
                    score = _score_coding(sample, predicted, lpips_metric=lpips_metric)
                    row = _clip_row(
                        plan,
                        sample,
                        score,
                        seed=seed,
                        epoch=epoch,
                        peak=peak,
                        wall_s=per_frame,
                        self_reconstruction_psnr=None,
                        anchors=used_anchors,
                        drive_mode=drive_mode,
                        appearance_source="own-keyframe",
                    )
                    rows.append(row)
                    progress(
                        f"[probe] {plan.name} {clip.key} offset={sample.offset} "
                        f"lpips={_fmt(row.object_lpips)} psnr={score.object_psnr_db:.2f} "
                        f"bound={row.lpips_bound}"
                    )
                progress(
                    f"[probe] {plan.name} {clip.key} clip of {len(samples)} frames "
                    f"in {clip_wall:.1f}s ({per_frame:.1f}s/frame)"
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
                progress(f"[probe] {plan.name} FAIL {clip.key} clip mode: {exc}")
                for offset in used_offsets:
                    rows.append(
                        _error_row(
                            plan,
                            clip,
                            keyframe_index=keyframe_index,
                            offset=offset,
                            seed=seed,
                            epoch=epoch,
                            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                            drive_mode=drive_mode,
                        )
                    )
            checkpoint()
            continue

        for offset in used_offsets:
            try:
                sample = load_coding_sample(clip, keyframe_index, offset)
                appearance_source = "own-keyframe"
                if donor_rgb is not None:
                    sample = with_appearance(sample, donor_rgb)
                    appearance_source = f"donor:{donor_key}"
                _reset_peak(device)
                t0 = time.perf_counter()
                if plan.name in BASELINES or built is None:
                    predicted = predict_static_copy(sample.appearance_rgb, CANVAS, CANVAS)
                else:
                    predicted = built.generate(
                        _coding_bundle(sample), seed=seed, device=device, params=params
                    )
                wall_s = time.perf_counter() - t0
                peak = _peak_bytes(device)
                engine_peak = max(engine_peak, peak)
                last_run = getattr(built, "last_run", None) if built is not None else None
                if isinstance(last_run, dict) and last_run.get("peak_vram_bytes"):
                    engine_peak = max(engine_peak, int(last_run["peak_vram_bytes"]))
                    peak = max(peak, int(last_run["peak_vram_bytes"]))
                epoch = _epoch_of(built) if built is not None else None
                score = _score_coding(sample, np.asarray(predicted), lpips_metric=lpips_metric)
                self_psnr: float | None = None
                if recon_at is not None and offset == recon_at:
                    self_psnr = _measure_self_recon(
                        clip=clip,
                        target_index=sample.target_frame_index,
                        canvas=CANVAS,
                        generator=None if plan.name in BASELINES else built,
                        seed=seed,
                        device=device,
                        params=params,
                        lpips_metric=lpips_metric,
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
                    anchors=used_anchors,
                    drive_mode=drive_mode,
                    appearance_source=appearance_source,
                )
                progress(
                    f"[probe] {plan.name} {clip.key} offset={offset} "
                    f"lpips={_fmt(row.object_lpips)} psnr={score.object_psnr_db:.2f} "
                    f"frame={score.frame_psnr_db:.2f} use={row.appearance_use} "
                    f"bound={row.lpips_bound or row.object_bound} {wall_s:.1f}s"
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
                    drive_mode=drive_mode,
                )
            rows.append(row)
            checkpoint()

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
        drive_mode=drive_mode,
        clips=rows,
        headline=_headline(
            plan,
            rows,
            anchors=used_anchors,
            keyframe_index=keyframe_index,
            offsets=used_offsets,
            drive_mode=drive_mode,
        ),
    )
    result.headline["engine_wall_s"] = time.perf_counter() - started
    _write_json(out_dir / f"{plan.name}.json", asdict(result))
    return result


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _engine_summary(result: EngineResult) -> dict[str, Any]:
    return {
        "refused": result.refused,
        "refuse_reason": result.refuse_reason,
        "checkpoint_epoch": result.checkpoint_epoch,
        "drive_mode": result.drive_mode,
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
    self_recon: bool = False,
    lpips_metric: Any = None,
    progress: Any = print,
) -> dict[str, Any]:
    used_offsets = offsets if offsets is not None else CLIP_MODE_OFFSETS
    clips = list_clips(probe_root)
    # ``engines=()`` means the baselines and nothing else. Treating an empty
    # tuple as "unset" would silently drive the whole roster instead.
    chosen = PLANS if engines is None else tuple(plan_for(name) for name in engines)
    metric = lpips_metric if lpips_metric is not None else build_lpips(device)
    donors = donor_appearances(clips, keyframe_index)
    bounds_record = {
        "static_copy_psnr_expected_db": [
            STATIC_COPY_EXPECTED_LOW_DB,
            STATIC_COPY_EXPECTED_HIGH_DB,
        ],
        "static_copy_psnr_alarm_db": [STATIC_COPY_ALARM_LOW_DB, STATIC_COPY_ALARM_HIGH_DB],
        "static_copy_lpips_expected": [
            STATIC_COPY_LPIPS_EXPECTED_LOW,
            STATIC_COPY_LPIPS_EXPECTED_HIGH,
        ],
        "static_copy_lpips_alarm": [STATIC_COPY_LPIPS_ALARM_LOW, STATIC_COPY_LPIPS_ALARM_HIGH],
        "unrelated_lpips_expected": [
            UNRELATED_LPIPS_EXPECTED_LOW,
            UNRELATED_LPIPS_EXPECTED_HIGH,
        ],
        "unrelated_lpips_alarm_low": UNRELATED_LPIPS_ALARM_LOW,
        "published_lpips_anchors": {
            "identical": 0.0,
            "mild_noise": 0.250,
            "heavy_blur": 0.430,
            "unrelated_image": 0.645,
        },
        "anchored_on": "static-copy floor and unrelated-image null, measured in this run",
        "task": "appearance from keyframe, score against later frame",
    }
    summary: dict[str, Any] = {
        "citable": False,
        "seed": seed,
        "device": device,
        "canvas": CANVAS,
        "keyframe_index": keyframe_index,
        "offsets": list(used_offsets),
        "ranking_uses": RANKING_METRIC,
        "ranking_lower_is_better": RANKING_LOWER_IS_BETTER,
        "reported_beside": list(REPORTED_BESIDE),
        "ranking_ignores": list(NOT_RANKED),
        "lpips_available": metric is not None,
        "n_clips": len(clips),
        "donors": {key: donor for key, (donor, _) in donors.items()},
        "bounds_written_before_generate": bounds_record,
        "split_note": (
            "All 12 probe clips are from the 5 training-split videos. "
            "Animate-Anyone has also seen both held-out videos (PLAN.md §2.8); "
            "option 2: report AA as in-domain only. A pretrained engine carries "
            "the held-out arm when that arm is run."
        ),
        "static_copy": {},
        "null_control": {},
        "engines": {},
        "rank": [],
    }
    _write_json(out_dir / "summary.json", summary)

    baseline_results: dict[str, EngineResult] = {}
    for baseline_plan in (STATIC_COPY_PLAN, UNRELATED_IMAGE_PLAN):
        result = drive_engine(
            baseline_plan,
            clips,
            device=device,
            seed=seed,
            out_dir=out_dir,
            generator=None,
            keyframe_index=keyframe_index,
            offsets=used_offsets,
            self_recon=self_recon,
            lpips_metric=metric,
            donors=donors,
            progress=progress,
        )
        baseline_results[baseline_plan.name] = result
        summary["engines"][baseline_plan.name] = _engine_summary(result)
        _write_json(out_dir / "summary.json", summary)

    static = baseline_results[STATIC_COPY]
    null = baseline_results[UNRELATED_IMAGE]
    anchors = Anchors(
        floor_psnr=_lookup(static.clips, "object_psnr_db"),
        floor_lpips=_lookup(static.clips, "object_lpips"),
        null_lpips=_lookup(null.clips, "object_lpips"),
        floor_psnr_mean=_as_float(static.headline.get("object_psnr_db")),
        floor_lpips_mean=_as_float(static.headline.get("object_lpips")),
        null_lpips_mean=_as_float(null.headline.get("object_lpips")),
    )
    summary["static_copy"] = _engine_summary(static)
    summary["null_control"] = _engine_summary(null)
    summary["control"] = _control_record(anchors)
    progress(f"[probe] control: {summary['control']['note']}")
    _write_json(out_dir / "summary.json", summary)

    for plan in chosen:
        if plan.name in BASELINES:
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
            anchors=anchors,
            self_recon=self_recon,
            lpips_metric=metric,
            donors=donors,
            progress=progress,
        )
        summary["engines"][plan.name] = _engine_summary(result)
        summary["rank"] = _ranking(summary)
        _write_json(out_dir / "summary.json", summary)
        progress(f"[probe] checkpointed {plan.name} -> {out_dir / 'summary.json'}")
    summary["rank"] = _ranking(summary)
    _write_json(out_dir / "summary.json", summary)
    return summary


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _control_record(anchors: Anchors) -> dict[str, Any]:
    """Did the instrument separate the right player from the wrong one?"""
    floor = anchors.floor_lpips_mean
    null = anchors.null_lpips_mean
    if floor is None or null is None:
        return {
            "readable": False,
            "note": (
                "no LPIPS on one of the baselines, so the null control did not "
                "run. Nothing in this run may be ranked on LPIPS."
            ),
            "static_copy_lpips": floor,
            "unrelated_image_lpips": null,
        }
    verdict = judge_null_separation(floor, null)
    return {
        "readable": verdict.status == "ok",
        "status": verdict.status,
        "note": verdict.note,
        "static_copy_lpips": floor,
        "unrelated_image_lpips": null,
        "separation": null - floor,
    }


def _ranking(summary: Mapping[str, Any]) -> list[str]:
    """Rank only when the control says the instrument resolved identity."""
    control = summary.get("control")
    if isinstance(control, Mapping) and not control.get("readable", False):
        return []
    return rank_engines(summary["engines"])
