"""FG/BG headroom on real 4K point scenes, across the codec ladder.

Bounds go to disk before any encode. Paste-back must pass before the first
encode. 4K is slow: a scene window, not a match, checkpointed after every codec.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.components.metrics.bd_rate import RDCurve
from experiments.headroom.ladder import (
    DEFAULT_QPS,
    encode_luma_curve,
    encoders_available,
    resolved_tools,
)
from experiments.headroom.measure import (
    _saving,
    bg_headroom_intercoded,
    codec_ranking_alarm,
    declared_bounds,
    duplicate_rate_ratio,
)
from experiments.headroom.real import (
    PasteBackError,
    SceneClip,
    choose_two_point_scenes,
    load_scene_clip,
)
from experiments.headroom.remove import player_fraction, prepare_fills

CODECS = ("avc", "hevc", "av1", "vvc")
N_FRAMES = 48
FPS = 24.0


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


def _encode(
    frames: np.ndarray,
    *,
    work_dir: Path,
    qps: tuple[int, ...],
    codec_name: str,
    masks: np.ndarray | None = None,
    label: str = "",
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
    for key in ("fg", "bg_intercoded", "nulls", "tools"):
        if previous.get(key):
            fresh[key] = previous[key]
    if previous.get("alarms"):
        fresh["alarms"] = list(previous["alarms"])
    if previous.get("started_at"):
        fresh["started_at"] = previous["started_at"]
    return fresh


def run(
    *,
    out_dir: Path,
    n_frames: int = N_FRAMES,
    codecs: tuple[str, ...] = CODECS,
    qps: tuple[int, ...] = DEFAULT_QPS,
    paste_back_only: bool = False,
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
        "codecs": list(codecs),
        "clips": [],
        "fg": {},
        "bg_intercoded": {},
        "nulls": {},
        "alarms": [],
        "tools": {},
        "not_a_pointstream_result": True,
        "instrument_range_psnr_dB": [20.0, 50.0],
    }
    report = _resume_state(report_path, report)
    _write(report, report_path)

    scenes = choose_two_point_scenes()
    clips: list[SceneClip] = []
    for scene in scenes:
        print(f"{_now()} paste-back + load {scene['video']}/{scene['scene']}", flush=True)
        clip = load_scene_clip(scene, out_dir / "clips" / scene["video"], n_frames=n_frames)
        clips.append(clip)
        report["clips"].append(_clip_record(clip))
        report["input"] = (
            f"{clip.video_path} {clip.scene} frames {clip.window_start}:"
            f"{clip.window_start + clip.n_frames} "
            f"{clip.frames.shape[2]}x{clip.frames.shape[1]} {FPS:g}fps extract "
            f"convention={clip.convention}"
        )
        area_band = bounds["player_area_band"]
        if not (area_band[0] <= clip.player_area <= area_band[1]):
            report["alarms"].append(
                f"{clip.video} player_area {clip.player_area:.4f} outside {area_band}"
            )
        _write(report, report_path)

    report["input"] = " | ".join(
        f"{clip.video_path} {clip.scene} [{clip.window_start}:"
        f"{clip.window_start + clip.n_frames}] "
        f"{clip.frames.shape[2]}x{clip.frames.shape[1]} {FPS:g}fps {clip.convention}"
        for clip in clips
    )
    _write(report, report_path)
    if paste_back_only:
        report["finished_at"] = _now()
        report["paste_back_only"] = True
        _write(report, report_path)
        return report

    for codec_name in codecs:
        if not encoders_available(codec_name):
            report["alarms"].append(f"{codec_name} encoder not available on this host")
            report["tools"][codec_name] = {"available": False}
            _write(report, report_path)
            continue
        report["tools"][codec_name] = resolved_tools(codec_name)
        report["fg"].setdefault(codec_name, {})
        report["bg_intercoded"].setdefault(codec_name, {})

        def encode_fn(
            frames: np.ndarray,
            *,
            work_dir: Path,
            qps: tuple[int, ...],
            masks: np.ndarray | None = None,
            label: str = "",
            _codec: str = codec_name,
        ) -> dict[str, Any]:
            return _encode(
                frames,
                work_dir=work_dir,
                qps=qps,
                codec_name=_codec,
                masks=masks,
                label=label,
            )

        for clip in clips:
            key = f"{clip.video}/{clip.scene}"
            if key in report["fg"][codec_name]:
                print(f"{_now()} skip existing {codec_name} {key}", flush=True)
                continue
            work = out_dir / "encode" / clip.video / codec_name
            print(
                f"{_now()} prepare fills {codec_name} {key} "
                f"shape={tuple(clip.frames.shape)}",
                flush=True,
            )
            fills = prepare_fills(clip.frames, clip.masks)
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
                "original_fg_psnr": original.get("fg_psnr"),
                "original_bg_psnr": original.get("bg_psnr"),
                "tool": original.get("tool"),
            }
            arms: dict[str, Any] = fg
            for arm_name in ("plate", "flat", "median"):
                saving = arms[f"{arm_name}_vs_original"].get("saving")
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
            bg = bg_headroom_intercoded(
                conventional_curve=plate_arm,
                plate_bgr=fills.plate_bgr,
                homographies=fills.homographies,
                work_dir=work / "bg_still",
                encode_curve=encode_fn,
                qps=qps,
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

    first = clips[0]
    if encoders_available("avc"):

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

        empty = np.zeros(first.masks.shape, dtype=bool)
        empty_fills = prepare_fills(first.frames, empty)
        empty_orig = avc_fn(
            first.frames,
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
            first.frames,
            work_dir=out_dir / "null" / "dup",
            encode_curve=avc_fn,
        )
        report["nulls"] = {
            "empty_mask_plate_saving": empty_saving,
            "duplicate_rate_ratio": dup,
            "clip": f"{first.video}/{first.scene}",
            "codec": "avc",
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

    report["finished_at"] = _now()
    _write(report, report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/bp20-headroom"))
    parser.add_argument("--n-frames", type=int, default=N_FRAMES)
    parser.add_argument("--codecs", nargs="+", default=list(CODECS))
    parser.add_argument("--paste-back-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(
            out_dir=args.out,
            n_frames=args.n_frames,
            codecs=tuple(args.codecs),
            paste_back_only=args.paste_back_only,
        )
    except PasteBackError as exc:
        print(f"PASTE-BACK FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(_jsonable(report), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
