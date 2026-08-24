"""FG/BG headroom on real 4K point scenes, across the codec ladder.

Bounds go to disk before any encode. Paste-back must pass before the first
encode. 4K is slow: a scene window, not a match, checkpointed after every codec.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

from src.components.metrics.bd_rate import RDCurve
from src.components.metrics.comparison import compare_paired
from experiments.headroom.ladder import (
    AV1_BG_QPS,
    DEFAULT_QPS,
    encode_luma_curve,
    encoders_available,
    resolved_tools,
)
from experiments.headroom.measure import (
    _saving,
    bg_headroom_intercoded,
    codec_ranking_alarm,
    common_quality_interval,
    declared_bounds,
    duplicate_rate_ratio,
    saving_on_interval,
)
from experiments.headroom.real import (
    MIN_CLIPS,
    MIN_MATCHES,
    PasteBackError,
    SceneClip,
    iter_point_scenes_spread,
    load_cached_clip,
    load_scene_clip,
)
from experiments.headroom.remove import player_fraction, prepare_fills

CODECS = ("avc", "hevc", "av1", "vvc")
N_FRAMES = 48
FPS = 24.0
BP20_DIR = Path("outputs/bp20-headroom")
BP20_ALCARAZ_AVC_PLATE = 0.2602328735909394
BP20_KEYS = frozenset({"alcaraz_highlights/scene_000", "federer_djokovic/scene_001"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, RDCurve):
        return {
            "rates": [float(rate) for rate in value.rates],
            "qualities": [float(quality) for quality in value.qualities],
            "label": value.label,
        }
    if isinstance(value, dict):
        skip = {"frames", "masks", "plate_bgr", "homographies"}
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if key not in skip
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value) if isinstance(value, np.floating) else int(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _first_plate_saving(by_codec: Any, codec_name: str) -> float | None:
    """The first clip's plate saving for one codec, or None if it has no rows.

    Spelled out rather than chained through ``.get(...).get(...)`` because the
    codec-ranking alarm reads this, and an alarm that silently sees ``None``
    where it should see a number would stop firing without anyone noticing.
    """
    clips = by_codec.get(codec_name) or {}
    for row in clips.values():
        arm = row.get("plate_vs_original") or {}
        saving = arm.get("saving")
        if isinstance(saving, (int, float)):
            return float(saving)
        return None
    return None


def _write(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    print(f"{_now()} wrote {path}", flush=True)


def _mean_se(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "se": None}
    mean = float(statistics.fmean(values))
    se = float(statistics.stdev(values) / (n**0.5)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "se": se}


def _curve_from_payload(payload: Any) -> RDCurve | None:
    if isinstance(payload, RDCurve):
        return payload
    if not isinstance(payload, dict):
        return None
    rates = payload.get("rates")
    qualities = payload.get("qualities")
    if not rates or not qualities:
        return None
    return RDCurve(
        rates=tuple(float(r) for r in rates),
        qualities=tuple(float(q) for q in qualities),
        label=str(payload.get("label") or ""),
    )


def _clip_record(clip: SceneClip) -> dict[str, Any]:
    return {
        "video": clip.video,
        "scene": clip.scene,
        "video_path": str(clip.video_path),
        "t_start": clip.t_start,
        "t_end": clip.t_end,
        "cluster": clip.cluster,
        "convention": clip.convention,
        "window_start": clip.window_start,
        "n_frames": clip.n_frames,
        "frame_hw": [int(clip.frames.shape[1]), int(clip.frames.shape[2])],
        "player_area": clip.player_area,
        "paste_back": {
            "winner": clip.paste_back.get("winner"),
            "winner_mae": clip.paste_back.get("winner_mae"),
            "mean_mae": clip.paste_back.get("mean_mae"),
            "window_mae": clip.paste_back.get("window_mae"),
            "threshold": clip.paste_back.get("threshold"),
            "n_extract_24": clip.paste_back.get("n_extract_24"),
        },
        "ffmpeg": clip.ffmpeg,
    }


def _resume_state(path: Path, fresh: dict[str, Any]) -> dict[str, Any]:
    """Keep finished codec/clip arms. A fresh ``fg: {}`` would re-encode 4K."""
    if not path.is_file():
        return fresh
    previous = json.loads(path.read_text())
    for key in (
        "fg",
        "bg_intercoded",
        "nulls",
        "tools",
        "dropped",
        "clips",
        "plate_nan_bp20",
        "summary",
        "common_interval",
    ):
        if previous.get(key):
            fresh[key] = previous[key]
    if previous.get("alarms"):
        fresh["alarms"] = list(previous["alarms"])
    if previous.get("started_at"):
        fresh["started_at"] = previous["started_at"]
    if previous.get("input") and previous["input"] != "not yet loaded":
        fresh["input"] = previous["input"]
    return fresh


def _reuse_dirs_for(
    *,
    out_dir: Path,
    reuse_from: Path | None,
    clip: SceneClip,
    codec_name: str,
    arm: str,
    label: str,
) -> tuple[Path, ...]:
    """Original-arm bitstreams may be reused when clip/codec/QP identity match.

    Plate/flat/median/bg change if the plate filler changed, so those are
    encoded fresh. BP20 laid arms out as ``encode/{video}/{codec}/{arm}``
    (one scene per video); BP21 uses ``encode/{video}/{scene}/{codec}/{arm}``.
    """
    if label != "original" or reuse_from is None:
        return ()
    found: list[Path] = []
    new_layout = (
        out_dir / "encode" / clip.video / clip.scene / codec_name / arm
    )
    if new_layout.is_dir():
        found.append(new_layout)
    if reuse_from.is_dir():
        old = reuse_from / "encode" / clip.video / codec_name / arm
        if old.is_dir():
            found.append(old)
        new_in_old = reuse_from / "encode" / clip.video / clip.scene / codec_name / arm
        if new_in_old.is_dir():
            found.append(new_in_old)
    return tuple(found)


def _encode(
    frames: np.ndarray,
    *,
    work_dir: Path,
    qps: tuple[int, ...],
    codec_name: str,
    masks: np.ndarray | None = None,
    label: str = "",
    reuse_dirs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    print(
        f"{_now()} encode {codec_name} {label} qps={qps} "
        f"shape={tuple(frames.shape)} -> {work_dir}",
        flush=True,
    )
    return encode_luma_curve(
        frames,
        work_dir=work_dir,
        qps=qps,
        codec_name=codec_name,
        masks=masks,
        label=label,
        fps=FPS,
        reuse_dirs=reuse_dirs,
    )


def _input_line(clips: list[SceneClip]) -> str:
    return " | ".join(
        f"{clip.video_path} {clip.scene} [{clip.window_start}:"
        f"{clip.window_start + clip.n_frames}] "
        f"{clip.frames.shape[2]}x{clip.frames.shape[1]} {FPS:g}fps {clip.convention}"
        for clip in clips
    )


def _load_or_select_clips(
    *,
    out_dir: Path,
    report: dict[str, Any],
    report_path: Path,
    n_frames: int,
    n_clips: int,
    min_matches: int,
    bounds: dict[str, Any],
) -> list[SceneClip]:
    cached: list[SceneClip] = []
    for record in report.get("clips") or []:
        dest = out_dir / "clips" / record["video"] / record["scene"]
        clip = load_cached_clip(record, dest)
        if clip is None:
            cached = []
            break
        cached.append(clip)
    if len(cached) >= n_clips:
        n_ok_matches = len({clip.video for clip in cached})
        if n_ok_matches >= min_matches:
            print(
                f"{_now()} resume {len(cached)} cached clips "
                f"from {n_ok_matches} matches",
                flush=True,
            )
            return cached[:n_clips] if len(cached) > n_clips else cached

    scenes = iter_point_scenes_spread()
    extra, dropped = _append_survivors(
        list(scenes),
        have=cached,
        n=n_clips,
        min_matches=min_matches,
        work_dir=out_dir / "clips",
        n_frames=n_frames,
    )
    report["dropped"] = list(report.get("dropped") or []) + dropped
    clips = cached + extra
    report["clips"] = [_clip_record(clip) for clip in clips]
    area_band = bounds["player_area_band"]
    for clip in clips:
        if not (area_band[0] <= clip.player_area <= area_band[1]):
            report["alarms"].append(
                f"{clip.video}/{clip.scene} player_area {clip.player_area:.4f} "
                f"outside {area_band}"
            )
    report["input"] = _input_line(clips)
    _write(report, report_path)
    return clips


def _append_survivors(
    scenes: list[dict[str, Any]],
    *,
    have: list[SceneClip],
    n: int,
    min_matches: int,
    work_dir: Path,
    n_frames: int,
) -> tuple[list[SceneClip], list[dict[str, Any]]]:
    extra: list[SceneClip] = []
    dropped: list[dict[str, Any]] = []
    have_keys = {f"{c.video}/{c.scene}" for c in have}

    for scene in scenes:
        survivors = have + extra
        if len(survivors) >= n and len({c.video for c in survivors}) >= min_matches:
            break
        key = f"{scene['video']}/{scene['scene']}"
        if key in have_keys:
            continue
        dest = work_dir / scene["video"] / scene["scene"]
        try:
            clip = load_scene_clip(scene, dest, n_frames=n_frames)
        except PasteBackError as exc:
            dropped.append(
                {"video": scene.get("video"), "scene": scene.get("scene"), "reason": str(exc)}
            )
            print(f"DROP {key}: {exc}", flush=True)
            continue
        extra.append(clip)
        have_keys.add(key)
    survivors = have + extra
    if len(survivors) < n or len({c.video for c in survivors}) < min_matches:
        raise PasteBackError(
            f"need ≥{n} paste-back survivors from ≥{min_matches} matches; "
            f"got {len(survivors)} from {len({c.video for c in survivors})}. "
            f"dropped={dropped}"
        )
    return extra, dropped


def _plate_nan_check(
    *,
    clips: list[SceneClip],
    out_dir: Path,
    reuse_from: Path | None,
    report: dict[str, Any],
    report_path: Path,
    bounds: dict[str, Any],
) -> None:
    """Re-encode one BP20 plate arm with the new filler; publish a correction if big."""
    if report.get("plate_nan_bp20"):
        return
    target = next(
        (c for c in clips if f"{c.video}/{c.scene}" == "alcaraz_highlights/scene_000"),
        None,
    )
    if target is None or not encoders_available("avc"):
        report["plate_nan_bp20"] = {"skipped": True, "reason": "BP20 clip or AVC missing"}
        _write(report, report_path)
        return
    fills = prepare_fills(target.frames, target.masks)
    work = out_dir / "plate_nan_check" / target.video / target.scene / "avc"
    original = _encode(
        target.frames,
        work_dir=work / "original",
        qps=(32, 40, 48),
        codec_name="avc",
        masks=target.masks,
        label="original",
        reuse_dirs=_reuse_dirs_for(
            out_dir=out_dir,
            reuse_from=reuse_from,
            clip=target,
            codec_name="avc",
            arm="original",
            label="original",
        ),
    )
    plate_arm = _encode(
        fills.plate,
        work_dir=work / "plate",
        qps=(32, 40, 48),
        codec_name="avc",
        masks=target.masks,
        label="plate",
    )
    compared = _saving(original["curve"], plate_arm["curve"])
    new_saving = compared.get("saving")
    delta = None if new_saving is None else float(new_saving) - BP20_ALCARAZ_AVC_PLATE
    row = {
        "clip": "alcaraz_highlights/scene_000",
        "codec": "avc",
        "bp20_plate_saving": BP20_ALCARAZ_AVC_PLATE,
        "bp21_plate_saving": new_saving,
        "delta": delta,
        "qps": [32, 40, 48],
        "expect_abs_le": bounds["plate_nan_fg_delta_abs_max"],
        "alarm_if_abs_gt": bounds["plate_nan_fg_delta_alarm"],
        "compared": compared,
        "tool": original.get("tool"),
    }
    if delta is not None and abs(delta) > bounds["plate_nan_fg_delta_alarm"]:
        report["alarms"].append(
            f"plate-NaN fill moved BP20 AVC FG saving by {delta:.4f} "
            f"(>{bounds['plate_nan_fg_delta_alarm']}); this is a published-number correction"
        )
        row["correction_to_publish"] = True
    elif delta is not None and abs(delta) > bounds["plate_nan_fg_delta_abs_max"]:
        report["alarms"].append(
            f"plate-NaN fill moved BP20 AVC FG saving by {delta:.4f}; "
            f"above expect ≤{bounds['plate_nan_fg_delta_abs_max']} but below alarm"
        )
        row["correction_to_publish"] = False
    else:
        row["correction_to_publish"] = False
    report["plate_nan_bp20"] = row
    _write(report, report_path)


def _summarize(report: dict[str, Any], clips: list[SceneClip], bounds: dict[str, Any]) -> None:
    keys = [f"{c.video}/{c.scene}" for c in clips]
    areas = [float(c.player_area) for c in clips]
    summary: dict[str, Any] = {
        "n_clips": len(clips),
        "n_matches": len({c.video for c in clips}),
        "player_area": _mean_se(areas),
        "player_area_per_clip": {
            f"{c.video}/{c.scene}": c.player_area for c in clips
        },
        "fg_plate": {},
        "fg_plate_common_interval": {},
        "fg_flat": {},
        "fg_median": {},
        "bg_saving": {},
        "concentration": {},
        "paired": {},
        "vvc_gap": {},
    }
    for codec_name in report.get("codecs") or CODECS:
        plate_vals: list[float] = []
        plate_common: list[float] = []
        flat_vals: list[float] = []
        median_vals: list[float] = []
        bg_vals: list[float] = []
        conc_vals: list[float] = []
        per_clip: dict[str, Any] = {}
        for key in keys:
            fg_row = ((report.get("fg") or {}).get(codec_name) or {}).get(key) or {}
            bg_row = ((report.get("bg_intercoded") or {}).get(codec_name) or {}).get(key) or {}
            plate = (fg_row.get("plate_vs_original") or {}).get("saving")
            common = (fg_row.get("plate_vs_original_common_interval") or {}).get("saving")
            flat = (fg_row.get("flat_vs_original") or {}).get("saving")
            median = (fg_row.get("median_vs_original") or {}).get("saving")
            bg_save = (bg_row.get("bd_vs_conventional") or {}).get("saving")
            area = fg_row.get("player_area")
            cell = {
                "plate": plate,
                "plate_common_interval": common,
                "flat": flat,
                "median": median,
                "bg": bg_save,
                "player_area": area,
                "bg_overlap_fraction": (bg_row.get("bd_vs_conventional") or {}).get(
                    "overlap_fraction"
                ),
            }
            per_clip[key] = cell
            if isinstance(plate, (int, float)):
                plate_vals.append(float(plate))
                if isinstance(area, (int, float)) and float(area) > 0:
                    conc_vals.append(float(plate) / float(area))
            if isinstance(common, (int, float)):
                plate_common.append(float(common))
            if isinstance(flat, (int, float)):
                flat_vals.append(float(flat))
            if isinstance(median, (int, float)):
                median_vals.append(float(median))
            if isinstance(bg_save, (int, float)):
                bg_vals.append(float(bg_save))
        summary["fg_plate"][codec_name] = {**_mean_se(plate_vals), "per_clip": per_clip}
        summary["fg_plate_common_interval"][codec_name] = _mean_se(plate_common)
        summary["fg_flat"][codec_name] = _mean_se(flat_vals)
        summary["fg_median"][codec_name] = _mean_se(median_vals)
        summary["bg_saving"][codec_name] = _mean_se(bg_vals)
        summary["concentration"][codec_name] = _mean_se(conc_vals)
        mean = summary["fg_plate"][codec_name]["mean"]
        band = bounds.get(f"fg_saving_band_{codec_name}")
        if mean is not None and band and not (band[0] <= mean <= band[1]):
            report["alarms"].append(
                f"{codec_name} mean FG plate {mean:.4f} (n={len(plate_vals)}) "
                f"outside {band}"
            )

    def _vals(codec_name: str, field: str = "plate") -> list[float]:
        rows = ((report.get("fg") or {}).get(codec_name) or {})
        out: list[float] = []
        for key in keys:
            saving = ((rows.get(key) or {}).get(f"{field}_vs_original") or {}).get("saving")
            if isinstance(saving, (int, float)):
                out.append(float(saving))
            else:
                out.append(float("nan"))
        return out

    avc = [v for v in _vals("avc") if v == v]
    vvc = [v for v in _vals("vvc") if v == v]
    if len(avc) >= 2 and len(avc) == len(vvc):
        paired = compare_paired("avc", avc, "vvc", vvc, higher_is_better=True)
        summary["paired"]["avc_minus_vvc"] = {
            "n": paired.n,
            "mean_difference": paired.mean_difference,
            "standard_error": paired.standard_error,
            "verdict": paired.verdict,
            "winner": paired.winner,
            "describe": paired.describe(),
        }
        gap_band = bounds.get("vvc_gap_avc_minus_vvc") or [0.04, 0.10]
        gap = paired.mean_difference
        survived = gap_band[0] <= gap <= gap_band[1]
        summary["vvc_gap"] = {
            "avc_minus_vvc": gap,
            "n": paired.n,
            "se": paired.standard_error,
            "expect_survive": bounds.get("vvc_gap_expect_survive"),
            "expect_band": gap_band,
            "survived": survived,
            "sentence": (
                "codec: the AVC−VVC FG gap survived a common QP set and a "
                "common PSNR interval."
                if survived
                else "confound: the AVC−VVC FG gap did not survive a common "
                "QP set / common PSNR interval."
            ),
        }
    area_mean = summary["player_area"]["mean"]
    if area_mean is not None:
        band = bounds["player_area_band"]
        if not (band[0] <= area_mean <= band[1]):
            report["alarms"].append(f"mean player_area {area_mean:.4f} outside {band}")
    report["summary"] = summary


def _fill_common_interval(report: dict[str, Any], clips: list[SceneClip]) -> None:
    """Slice every codec's original/plate curves to the clip's common PSNR range."""
    codecs = [name for name in (report.get("codecs") or CODECS) if name in (report.get("fg") or {})]
    per_clip: dict[str, Any] = {}
    for clip in clips:
        key = f"{clip.video}/{clip.scene}"
        curves: list[RDCurve] = []
        pairs: dict[str, tuple[RDCurve, RDCurve]] = {}
        for codec_name in codecs:
            row = ((report.get("fg") or {}).get(codec_name) or {}).get(key) or {}
            original = _curve_from_payload(row.get("original_curve"))
            plate = _curve_from_payload(row.get("plate_curve"))
            if original is None or plate is None:
                continue
            curves.extend([original, plate])
            pairs[codec_name] = (original, plate)
        if len(curves) < 2:
            continue
        try:
            interval = common_quality_interval(*curves)
        except ValueError:
            continue
        per_clip[key] = {"interval": list(interval), "n_curves": len(curves)}
        for codec_name, (original, plate) in pairs.items():
            sliced = saving_on_interval(original, plate, interval)
            report["fg"][codec_name][key]["plate_vs_original_common_interval"] = sliced
    report["common_interval"] = per_clip


def run(
    *,
    out_dir: Path,
    n_frames: int = N_FRAMES,
    codecs: tuple[str, ...] = CODECS,
    qps: tuple[int, ...] = DEFAULT_QPS,
    av1_bg_qps: tuple[int, ...] = AV1_BG_QPS,
    n_clips: int = MIN_CLIPS,
    min_matches: int = MIN_MATCHES,
    paste_back_only: bool = False,
    reuse_from: Path | None = BP20_DIR,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    bounds = declared_bounds()
    report: dict[str, Any] = {
        "input": "not yet loaded",
        "bounds_written_before_measurement": bounds,
        "started_at": _now(),
        "n_frames": n_frames,
        "fps": FPS,
        "qps": list(qps),
        "av1_bg_qps": list(av1_bg_qps),
        "n_clips": n_clips,
        "min_matches": min_matches,
        "codecs": list(codecs),
        "clips": [],
        "dropped": [],
        "fg": {},
        "bg_intercoded": {},
        "nulls": {},
        "alarms": [],
        "tools": {},
        "not_a_pointstream_result": True,
        "instrument_range_psnr_dB": [20.0, 50.0],
        "instrument_range_rate": "payload bytes",
        "vvc_gap_prediction_before_run": {
            "expect_gap_to_survive": True,
            "expect_avc_minus_vvc": [0.04, 0.10],
            "reason": (
                "QP 47 vs 48 is a small rate step; a 0.077 gap that repeated "
                "on both BP20 clips is large for that. Common QP 32/40/46 and "
                "a common PSNR interval should not erase a gap that size."
            ),
        },
    }
    report = _resume_state(report_path, report)
    _write(report, report_path)

    clips = _load_or_select_clips(
        out_dir=out_dir,
        report=report,
        report_path=report_path,
        n_frames=n_frames,
        n_clips=n_clips,
        min_matches=min_matches,
        bounds=bounds,
    )
    report["input"] = _input_line(clips)
    _write(report, report_path)
    if paste_back_only:
        report["finished_at"] = _now()
        report["paste_back_only"] = True
        _write(report, report_path)
        return report

    _plate_nan_check(
        clips=clips,
        out_dir=out_dir,
        reuse_from=reuse_from,
        report=report,
        report_path=report_path,
        bounds=bounds,
    )

    for codec_name in codecs:
        if not encoders_available(codec_name):
            report["alarms"].append(f"{codec_name} encoder not available on this host")
            report["tools"][codec_name] = {"available": False}
            _write(report, report_path)
            continue
        report["tools"][codec_name] = resolved_tools(codec_name)
        report["fg"].setdefault(codec_name, {})
        report["bg_intercoded"].setdefault(codec_name, {})

        for clip in clips:
            key = f"{clip.video}/{clip.scene}"
            if key in report["fg"][codec_name]:
                print(f"{_now()} skip existing {codec_name} {key}", flush=True)
                continue
            work = out_dir / "encode" / clip.video / clip.scene / codec_name
            print(
                f"{_now()} prepare fills {codec_name} {key} "
                f"shape={tuple(clip.frames.shape)}",
                flush=True,
            )
            fills = prepare_fills(clip.frames, clip.masks)

            def encode_fn(
                frames: np.ndarray,
                *,
                work_dir: Path,
                qps: tuple[int, ...],
                masks: np.ndarray | None = None,
                label: str = "",
                _codec: str = codec_name,
                _clip: SceneClip = clip,
            ) -> dict[str, Any]:
                return _encode(
                    frames,
                    work_dir=work_dir,
                    qps=qps,
                    codec_name=_codec,
                    masks=masks,
                    label=label,
                    reuse_dirs=_reuse_dirs_for(
                        out_dir=out_dir,
                        reuse_from=reuse_from,
                        clip=_clip,
                        codec_name=_codec,
                        arm=label,
                        label=label,
                    ),
                )

            original = encode_fn(
                clip.frames,
                work_dir=work / "original",
                qps=qps,
                masks=clip.masks,
                label="original",
            )
            for note in original.get("notes") or ():
                if note not in report["alarms"]:
                    report["alarms"].append(note)
            plate_arm = encode_fn(
                fills.plate,
                work_dir=work / "plate",
                qps=qps,
                masks=clip.masks,
                label="plate",
            )
            flat_arm = encode_fn(
                fills.flat,
                work_dir=work / "flat",
                qps=qps,
                masks=clip.masks,
                label="flat",
            )
            median_arm = encode_fn(
                fills.median,
                work_dir=work / "median",
                qps=qps,
                masks=clip.masks,
                label="median",
            )
            fg = {
                "player_area": player_fraction(clip.masks),
                "plate_vs_original": _saving(original["curve"], plate_arm["curve"]),
                "flat_vs_original": _saving(original["curve"], flat_arm["curve"]),
                "median_vs_original": _saving(original["curve"], median_arm["curve"]),
                "original_curve": original["curve"],
                "plate_curve": plate_arm["curve"],
                "original_fg_psnr": original.get("fg_psnr"),
                "original_bg_psnr": original.get("bg_psnr"),
                "qps_used": original.get("qps"),
                "tool": original.get("tool"),
            }
            for arm_name in ("plate", "flat", "median"):
                saving = fg[f"{arm_name}_vs_original"].get("saving")
                if saving is None:
                    continue
                if saving < 0:
                    report["alarms"].append(
                        f"{codec_name} {key} {arm_name} saving {saving:.4f} < 0"
                    )
                if saving > 0.40:
                    report["alarms"].append(
                        f"{codec_name} {key} {arm_name} saving {saving:.4f} > 0.40 "
                        "(possible mask encode)"
                    )
                band = bounds.get(f"fg_saving_band_{codec_name}")
                if band and arm_name == "plate" and not (band[0] <= saving <= band[1]):
                    report["alarms"].append(
                        f"{codec_name} {key} plate saving {saving:.4f} outside {band}"
                    )
            report["fg"][codec_name][key] = fg
            bg_qps = av1_bg_qps if codec_name == "av1" else qps
            if codec_name == "av1" and tuple(bg_qps) != tuple(qps):
                plate_for_bg = encode_fn(
                    fills.plate,
                    work_dir=work / "plate_av1_bg",
                    qps=bg_qps,
                    masks=clip.masks,
                    label="plate",
                )
                conventional_for_bg = plate_for_bg
            else:
                conventional_for_bg = plate_arm
            bg = bg_headroom_intercoded(
                conventional_curve=conventional_for_bg,
                plate_bgr=fills.plate_bgr,
                homographies=fills.homographies,
                work_dir=work / "bg_still",
                encode_curve=encode_fn,
                qps=bg_qps,
            )
            report["bg_intercoded"][codec_name][key] = bg
            report["alarms"].extend(bg.get("alarms") or [])
            _write(report, report_path)

        plate_savings = {
            name: _first_plate_saving(report["fg"], name)
            for name in ("avc", "hevc", "av1")
        }
        ranking = codec_ranking_alarm(plate_savings)
        if ranking:
            report["alarms"].append(ranking)
        _write(report, report_path)

    _fill_common_interval(report, clips)

    null_clip = next((c for c in clips if f"{c.video}/{c.scene}" not in BP20_KEYS), clips[0])
    if encoders_available("avc") and not report.get("nulls"):

        def avc_fn(
            frames: np.ndarray,
            *,
            work_dir: Path,
            qps: tuple[int, ...],
            masks: np.ndarray | None = None,
            label: str = "",
        ) -> dict[str, Any]:
            return _encode(
                frames,
                work_dir=work_dir,
                qps=qps,
                codec_name="avc",
                masks=masks,
                label=label,
            )

        empty = np.zeros(null_clip.masks.shape, dtype=bool)
        empty_fills = prepare_fills(null_clip.frames, empty)
        empty_orig = avc_fn(
            null_clip.frames,
            work_dir=out_dir / "null" / "empty_original",
            qps=qps[:2],
            masks=empty,
            label="empty_original",
        )
        empty_plate = avc_fn(
            empty_fills.plate,
            work_dir=out_dir / "null" / "empty_plate",
            qps=qps[:2],
            masks=empty,
            label="empty_plate",
        )
        empty_saving = _saving(empty_orig["curve"], empty_plate["curve"]).get("saving")
        dup = duplicate_rate_ratio(
            null_clip.frames,
            work_dir=out_dir / "null" / "dup",
            encode_curve=avc_fn,
        )
        report["nulls"] = {
            "empty_mask_plate_saving": empty_saving,
            "duplicate_rate_ratio": dup,
            "clip": f"{null_clip.video}/{null_clip.scene}",
            "codec": "avc",
            "note": "nulls on a clip that is not one of the two BP20 scenes"
            if f"{null_clip.video}/{null_clip.scene}" not in BP20_KEYS
            else "only BP20 clips survived; nulls ran on the first of those",
        }
        if (
            empty_saving is not None
            and abs(empty_saving) > bounds["empty_mask_saving_abs_max"]
        ):
            report["alarms"].append(
                f"empty-mask saving {empty_saving:.4f} exceeds "
                f"{bounds['empty_mask_saving_abs_max']}"
            )
        lo, hi = bounds["duplicate_rate_ratio_band"]
        if not (lo <= dup <= hi):
            report["alarms"].append(f"duplicate rate ratio {dup:.4f} outside [{lo}, {hi}]")
        _write(report, report_path)

    _summarize(report, clips, bounds)
    report["finished_at"] = _now()
    _write(report, report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/bp21-headroom"))
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument("--n-clips", type=int, default=MIN_CLIPS)
    parser.add_argument("--min-matches", type=int, default=MIN_MATCHES)
    parser.add_argument("--codecs", nargs="+", default=list(CODECS))
    parser.add_argument("--qps", nargs="+", type=int, default=list(DEFAULT_QPS))
    parser.add_argument("--av1-bg-qps", nargs="+", type=int, default=list(AV1_BG_QPS))
    parser.add_argument("--reuse-from", type=Path, default=BP20_DIR)
    parser.add_argument("--paste-back-only", action="store_true")
    args = parser.parse_args(argv)
    reuse = args.reuse_from if args.reuse_from is not None else None
    try:
        report = run(
            out_dir=args.out,
            n_frames=args.n_frames,
            codecs=tuple(args.codecs),
            qps=tuple(args.qps),
            av1_bg_qps=tuple(args.av1_bg_qps),
            n_clips=args.n_clips,
            min_matches=args.min_matches,
            paste_back_only=args.paste_back_only,
            reuse_from=reuse,
        )
    except PasteBackError as exc:
        print(f"PASTE-BACK FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(_jsonable(report), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
