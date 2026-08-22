"""Coding-task probe for appearance conditioning (BP8).

Appearance from track-local frame 0, pose from frame N, score against frame N.
Pairing is by position in the sorted ``frame_*.png`` lists, never by rebuilding
a filename (copy of ``src/shared/tennis_dataset.py`` 95–110).

A static-copy arm (paste the keyframe, no model) is the floor. An engine at or
below that floor is reported as not using appearance.

This script does not edit ``experiments/probe/**``.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.components.generation._numpy import as_chw, as_hwc
from src.components.generation.pose import fit_to_canvas, letterbox_image
from src.components.metrics.evaluator import triage
from src.components.metrics.region import Region
from src.contracts.conditioning import ConditioningBundle, GenerationParams

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = REPO_ROOT / "assets" / "probe_set"
CANVAS = 512
KEYFRAME = 0
TARGET = 24
SEED = 42
DEVICE = "cuda:0"
STEPS = 20
STATIC_COPY_FLOOR_DB = 11.82


@dataclass(frozen=True)
class CodingClip:
    key: str
    path: Path
    n_frames: int
    split: str


def _sorted_frames(directory: Path) -> list[Path]:
    frames = sorted(directory.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png under {directory}")
    return frames


def _load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb, rgba[:, :, 3] > 0


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def letterbox_rgb(image: np.ndarray, canvas: int = CANVAS) -> tuple[np.ndarray, Any]:
    array = as_hwc(image)
    if array.ndim == 2:
        height, width = array.shape
    else:
        height, width = array.shape[:2]
    box = fit_to_canvas(width, height, canvas, canvas)
    return letterbox_image(array, box), box


def stretch_rgb(image: np.ndarray, canvas: int = CANVAS) -> np.ndarray:
    import cv2

    array = as_hwc(image)
    interp = cv2.INTER_NEAREST if array.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(array, (canvas, canvas), interpolation=interp)


def score_canvases(
    reference_hwc: np.ndarray,
    predicted_hwc: np.ndarray,
    object_mask_hw: np.ndarray,
) -> dict[str, float | int | str | bool]:
    region = Region.object(mask=np.asarray(object_mask_hw, dtype=bool), name="player")
    record = triage(reference_hwc[None, ...], predicted_hwc[None, ...], regions=[region])
    object_scores = record.for_role("object")
    frame_scores = record.for_role("whole-frame")
    if not object_scores or not frame_scores:
        raise RuntimeError("triage did not return object and whole-frame PSNR")
    return {
        "object_psnr_db": float(object_scores[0].value),
        "frame_psnr_db": float(frame_scores[0].value),
        "n_object_pixels": int(object_scores[0].n_pixels),
        "n_frame_pixels": int(frame_scores[0].n_pixels),
        "region_kind": "mask",
        "differs_from_reference": bool(not np.array_equal(predicted_hwc, reference_hwc)),
    }


def list_clips(probe_root: Path) -> tuple[CodingClip, ...]:
    manifest = json.loads((probe_root / "manifest.json").read_text())
    clips: list[CodingClip] = []
    training = set(manifest.get("training_videos", ()))
    held = set(manifest.get("held_out_videos", ()))
    for record in manifest["probe_clips"]:
        video = str(record["video"])
        if video in held:
            split = "held-out"
        elif video in training:
            split = "train"
        else:
            split = "unknown"
        rel = Path(str(record["path"]))
        track_dir = probe_root / rel
        clips.append(
            CodingClip(
                key=str(record["key"]),
                path=track_dir,
                n_frames=int(record["num_frames"]),
                split=split,
            )
        )
    return tuple(clips)


def load_pair(clip: CodingClip, *, keyframe: int, target: int) -> dict[str, Any]:
    """Pair colour and skeleton by position in the sorted frame lists.

    Copied from ``src/shared/tennis_dataset.py`` 95–110: glob ``frame_*.png``,
    sort, zip by index. Do not reconstruct a filename.
    """
    color_frames = _sorted_frames(clip.path)
    skel_dir = clip.path.parent / f"{clip.path.name}_skeleton"
    skel_frames = _sorted_frames(skel_dir)
    min_len = min(len(color_frames), len(skel_frames))
    if target >= min_len or keyframe >= min_len:
        raise IndexError(
            f"{clip.key} has {min_len} paired frames; need keyframe={keyframe} "
            f"and target={target}"
        )
    appearance_rgb, _ = _load_rgba(color_frames[keyframe])
    target_rgb, target_mask = _load_rgba(color_frames[target])
    pose_rgb = _load_rgb(skel_frames[target])
    return {
        "key": clip.key,
        "split": clip.split,
        "appearance_rgb": appearance_rgb,
        "target_rgb": target_rgb,
        "target_mask": np.asarray(target_mask, dtype=bool),
        "pose_rgb": pose_rgb,
        "appearance_path": str(color_frames[keyframe]),
        "target_path": str(color_frames[target]),
        "pose_path": str(skel_frames[target]),
        "appearance_hw": list(appearance_rgb.shape[:2]),
        "target_hw": list(target_rgb.shape[:2]),
        "pose_hw": list(pose_rgb.shape[:2]),
        "paired_by": "position_in_sorted_frame_lists",
        "filename_rebuild_used": False,
    }


def prepare_inputs(pair: dict[str, Any], *, fit: str) -> dict[str, np.ndarray]:
    if fit == "letterbox":
        appearance, _ = letterbox_rgb(pair["appearance_rgb"])
        pose, _ = letterbox_rgb(pair["pose_rgb"])
        target, target_box = letterbox_rgb(pair["target_rgb"])
        mask = letterbox_image(
            np.asarray(pair["target_mask"], dtype=np.uint8) * 255, target_box
        )
        return {
            "appearance": appearance,
            "pose": pose,
            "target": target,
            "mask": mask > 0,
        }
    if fit == "stretch":
        mask_u8 = np.asarray(pair["target_mask"], dtype=np.uint8) * 255
        return {
            "appearance": stretch_rgb(pair["appearance_rgb"]),
            "pose": stretch_rgb(pair["pose_rgb"]),
            "target": stretch_rgb(pair["target_rgb"]),
            "mask": stretch_rgb(mask_u8) > 0,
        }
    raise ValueError(f"unknown fit {fit!r}")


def write_bounds(path: Path) -> dict[str, Any]:
    """Write plausible bands *before* any generate() call."""
    payload = {
        "written_before_generate": True,
        "task": (
            "appearance from frame 0, pose from frame 24, score against frame 24, "
            "12 probe clips, seed 42, 20 DDIM steps, object-scoped PSNR on "
            "letterboxed crop alpha"
        ),
        "static_copy": {
            "expect_db": STATIC_COPY_FLOOR_DB,
            "worst_db": 10.0,
            "best_db": 14.0,
            "alarm_low_db": 9.0,
            "alarm_high_db": 16.0,
            "basis": (
                "PLAN.md §2.3 measured 11.82 dB object-scoped for paste-the-keyframe "
                "on this same coding task. Reproducing that floor should land near "
                "11.82. Outside 9–16 is a scoring/pairing bug, not a new finding."
            ),
        },
        "animate_anyone": {
            "worst_db": 9.0,
            "best_db": 28.0,
            "alarm_low_db": 9.0,
            "alarm_high_db": 35.0,
            "must_beat_floor_by_db": 1.0,
            "success_min_db": STATIC_COPY_FLOOR_DB + 1.0,
            "basis": (
                "AA is reference-conditioned (ReferenceNet). A working appearance "
                "path must beat the 11.82 dB static-copy floor by ~1 dB; at or "
                "below the floor it is not using appearance. Best case ~high "
                "teens/20s if identity holds. Alarm below ~9 dB is the 3-step "
                "melt band. Alarm above 35 dB is scoring the source against itself. "
                "Any number is in-domain: AA fine-tune includes both held-out videos "
                "(PLAN.md §2.4)."
            ),
        },
        "ip_adapter": {
            "worst_db": 10.0,
            "best_db": 28.0,
            "alarm_low_db": 9.0,
            "alarm_high_db": 35.0,
            "must_beat_floor_by_db": 1.0,
            "success_min_db": STATIC_COPY_FLOOR_DB + 1.0,
            "basis": (
                "Real IP-Adapter (h94/IP-Adapter) on stock SD-1.5 + stock OpenPose. "
                "The tennis ip-adapter-controlnet directory is a mislabelled seg "
                "ControlNet and is not this arm. A working appearance path must beat "
                "11.82 by ~1 dB. The known txt2img floor (~11 dB) is what the "
                "mislabeled checkpoint posted; beating the static-copy floor is the "
                "test that appearance entered."
            ),
        },
        "in_domain": True,
        "in_domain_reason": (
            "Animate-Anyone fine-tune set contains alcaraz_highlights and "
            "djokovic_zverev (PLAN.md §2.4). Probe clips are the 5 training-split "
            "videos, which AA also saw."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def drive(
    *,
    probe_root: Path,
    out_dir: Path,
    device: str,
    seed: int,
    steps: int,
    fit: str,
    engine: str,
) -> dict[str, Any]:
    clips = list_clips(probe_root)
    bounds_path = out_dir / "bounds.json"
    if not bounds_path.is_file():
        write_bounds(bounds_path)

    rows: list[dict[str, Any]] = []
    generator: Any = None
    if engine == "animate-anyone":
        from src.components.generation.animate_anyone import AnimateAnyoneGenerator

        generator = AnimateAnyoneGenerator(width=CANVAS, height=CANVAS, steps=steps)
    elif engine in {"ip-adapter-controlnet", "pose-ref-controlnet"}:
        from src.components.generation import REGISTRY as GENERATORS

        generator = GENERATORS.build(engine)

    started = time.perf_counter()
    last_progress = started
    for index, clip in enumerate(clips):
        pair = load_pair(clip, keyframe=KEYFRAME, target=TARGET)
        prepared = prepare_inputs(pair, fit=fit)
        static_score = score_canvases(
            prepared["target"], prepared["appearance"], prepared["mask"]
        )
        row: dict[str, Any] = {
            "engine": engine,
            "fit": fit,
            "clip_key": pair["key"],
            "split": pair["split"],
            "appearance_hw": pair["appearance_hw"],
            "target_hw": pair["target_hw"],
            "pose_hw": pair["pose_hw"],
            "sizes_differ": pair["appearance_hw"] != pair["target_hw"],
            "paired_by": pair["paired_by"],
            "static_copy": static_score,
        }
        if engine == "static-copy":
            row["object_psnr_db"] = static_score["object_psnr_db"]
            row["frame_psnr_db"] = static_score["frame_psnr_db"]
        elif engine in {"animate-anyone", "ip-adapter-controlnet", "pose-ref-controlnet"}:
            assert generator is not None
            # Independently letterboxed (or stretched) 512 canvases. Passing the
            # raw crops would let ControlNet's shared-box prepare resize the
            # later pose onto the keyframe canvas — the coding-task fault.
            bundle = ConditioningBundle(
                appearance=as_chw(prepared["appearance"]),
                pose=as_chw(prepared["pose"]),
                frame_index=TARGET,
                object_id=pair["key"],
            )
            t0 = time.perf_counter()
            predicted = generator.generate(
                bundle,
                seed=seed,
                device=device,
                params=GenerationParams(width=CANVAS, height=CANVAS, steps=steps),
            )
            wall_s = time.perf_counter() - t0
            pred_hwc = as_hwc(predicted)[..., :3]
            gen_score = score_canvases(prepared["target"], pred_hwc, prepared["mask"])
            last_run = dict(getattr(generator, "last_run", None) or {})
            if not last_run:
                last_run = {
                    "loaded_checkpoint": getattr(generator, "loaded_checkpoint", None),
                    "loaded_epoch": getattr(generator, "loaded_epoch", None),
                    "variant": getattr(generator, "variant", None),
                    "steps": steps,
                    "ip_adapter_scale": getattr(generator, "ip_adapter_scale", None),
                }
            row.update(
                {
                    "object_psnr_db": gen_score["object_psnr_db"],
                    "frame_psnr_db": gen_score["frame_psnr_db"],
                    "generation": gen_score,
                    "wall_s": wall_s,
                    "last_run": last_run,
                }
            )
            if index == 0:
                dump = out_dir / "first_clip"
                dump.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prepared["appearance"]).save(dump / "reference.png")
                Image.fromarray(prepared["pose"]).save(dump / "pose.png")
                Image.fromarray(prepared["target"]).save(dump / "target.png")
                Image.fromarray(pred_hwc).save(dump / "generated.png")
                Image.fromarray(prepared["appearance"]).save(dump / "static_copy.png")
        else:
            raise ValueError(f"unknown engine {engine!r}")

        verdict = "not using appearance"
        object_db = float(row["object_psnr_db"])
        if engine == "static-copy":
            verdict = "floor"
        elif object_db >= STATIC_COPY_FLOOR_DB + 1.0:
            verdict = "uses appearance"
        elif object_db > STATIC_COPY_FLOOR_DB:
            verdict = "below 1 dB margin — not a result"
        row["verdict"] = verdict
        rows.append(row)
        now = time.perf_counter()
        print(
            f"[bp8] {engine} {fit} {pair['key']} "
            f"object={object_db:.2f} static={float(static_score['object_psnr_db']):.2f} "
            f"verdict={verdict} sizes_differ={row['sizes_differ']}",
            flush=True,
        )
        if now - last_progress >= 600:
            print(f"[bp8] still running, clip {index + 1}/{len(clips)}", flush=True)
            last_progress = now
        _write_json(
            out_dir / f"{engine}-{fit}.json",
            {"engine": engine, "fit": fit, "clips": rows, "partial": True},
        )

    object_values = [float(row["object_psnr_db"]) for row in rows]
    static_values = [float(row["static_copy"]["object_psnr_db"]) for row in rows]
    mean_object = _mean(object_values)
    mean_static = _mean(static_values)
    summary = {
        "citable": False,
        "in_domain": True,
        "engine": engine,
        "fit": fit,
        "seed": seed,
        "device": device,
        "steps": steps if engine == "animate-anyone" else None,
        "canvas": CANVAS,
        "keyframe_index": KEYFRAME,
        "target_index": TARGET,
        "n_clips": len(rows),
        "mean_object_psnr_db": mean_object,
        "mean_static_copy_object_psnr_db": mean_static,
        "static_copy_floor_db": STATIC_COPY_FLOOR_DB,
        "delta_vs_static_db": (
            None if mean_object is None or mean_static is None else mean_object - mean_static
        ),
        "verdict": (
            "uses appearance"
            if mean_object is not None and mean_object >= STATIC_COPY_FLOOR_DB + 1.0
            else "not using appearance"
            if engine != "static-copy"
            else "floor"
        ),
        "engine_wall_s": time.perf_counter() - started,
        "clips": rows,
    }
    if generator is not None:
        last = getattr(generator, "last_run", None) or {}
        summary["checkpoint"] = last.get("checkpoint") or getattr(
            generator, "loaded_checkpoint", None
        )
        summary["loaded_epoch"] = getattr(generator, "loaded_epoch", None)
        summary["scheduler"] = last.get("scheduler")
        summary["reference_feed_first_clip"] = rows[0].get("last_run", {}).get("reference_feed")
        summary["reference_unet_first_clip"] = rows[0].get("last_run", {}).get("reference_unet")
    _write_json(out_dir / f"{engine}-{fit}.json", summary)
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "bp8-coding-task")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--fit", choices=("letterbox", "stretch"), default="letterbox")
    parser.add_argument(
        "--engine",
        choices=(
            "static-copy",
            "animate-anyone",
            "ip-adapter-controlnet",
            "pose-ref-controlnet",
        ),
        default="animate-anyone",
    )
    parser.add_argument("--bounds-only", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = write_bounds(out_dir / "bounds.json")
    print(f"[bp8] bounds written to {out_dir / 'bounds.json'} before generate", flush=True)
    if args.bounds_only:
        print(json.dumps(bounds, indent=2))
        return
    try:
        summary = drive(
            probe_root=args.probe_root,
            out_dir=out_dir,
            device=args.device,
            seed=args.seed,
            steps=args.steps,
            fit=args.fit,
            engine=args.engine,
        )
    except Exception:
        log_path = out_dir / "failed.log"
        log_path.write_text(traceback.format_exc())
        print(f"[bp8] FAILED; traceback at {log_path}", flush=True)
        raise
    print(json.dumps({k: v for k, v in summary.items() if k != "clips"}, indent=2, default=str))


if __name__ == "__main__":
    main()
