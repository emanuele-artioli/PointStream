"""Successive-halving driver for the G2 training campaign (report 10, Phase 5.4).

**This script trains for real when invoked without --dry-run.** It is the
"launch variants under successive halving" half of the protocol; the probe
set (`scripts/select_probe_set.py`) and checkpoint scoring through the
runner are its building blocks. Per this workstream's
explicit scope gate: this driver is meant to be validated with tiny budgets
(a couple of epochs, a couple of minutes) — a human decides separately when
to point it at the real multi-day campaign with realistic epoch budgets.

Successive-halving rule (exact)
--------------------------------
1. Rung 0: every alive variant trains for `--initial-epochs` (from scratch).
2. After training, each variant's checkpoint is scored on the probe set
   (`evaluate_checkpoint` — retired with src.decoder in BP22; rewire to
   the runner before the next real campaign), producing an
   aggregate PSNR/SSIM/VMAF/FVD(+LPIPS) record.
3. Rank variants by a composite score: for each metric, min-max normalize
   across the variants *present this rung* (lower-is-better metrics — FVD,
   the uncalibrated LPIPS-like distance — are flipped so higher-normalized
   always means better), then average the normalized metrics available for
   that variant. A metric is only used for ranking if at least 2 variants
   reported a non-None value for it that rung.
4. Keep the top `ceil(n_alive / 2)` variants (classic successive halving);
   with only 3 initial candidates this means: rung 0 keeps 2 of 3, rung 1
   keeps 1 of 2, rung 2 is a no-op (1 of 1) and the loop stops.
5. Survivors' budget **doubles** (rung 1 trains to 2x rung 0's cumulative
   epochs, rung 2 to 4x, ...); pix2pix/spade4tennis resume from their own
   checkpoint to an absolute cumulative-epoch target, ControlNet resumes
   from its saved directory for `rung_epochs = target - previous` more
   epochs (see `build_train_command`'s docstring for why the two families
   differ here).
6. **By default this driver runs exactly one rung per invocation and stops**,
   printing the ranking and survivors, so a human reviews the probe-log
   curve before the next rung's (bigger) budget is authorized — re-run with
   the same `--campaign-dir` to continue. `--auto-continue` opts into
   running all remaining rungs unattended; this flag is for a
   human-approved real campaign, not for validating the harness.

Animate-Anyone note (report 10 Phase 5.4(b) asks to "note if a training
script exists"): it does — `animate_anyone.scripts.train_stage_1`/
`train_stage_2` (installed from the vendored `moore-animateanyone` package),
driven by YAML configs under `assets/animate-anyone/configs/` rather than
argparse data-root/epochs flags, and its dataset loader expects a separate
`extract_meta_info.py` pre-pass to build `meta_paths` before training can
start. Wiring it as a fourth `Variant` kind needs a YAML-templating step
(patching `train_bs`/`max_train_steps`/`data.meta_paths` per rung) that is
out of scope for this pass — **not wired into `default_variants()`**; a
future session should add an `AnimateAnyoneVariant` following the same
`build_train_command` contract once that templating exists.

Usage
-----
    conda run -n pointstream python scripts/train_campaign.py \\
        --campaign-dir outputs/campaign/g2_smoke \\
        --manifest assets/probe_set/manifest.json \\
        --data-root assets/probe_set/training_view \\
        --eval-dataset-root assets/dataset \\
        --initial-epochs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import sqlite3  # noqa: F401

import cv2
import numpy as np
import torch

from src.components.generation.base import as_runner_ref
from src.components.generation._numpy import as_chw
from src.contracts.conditioning import ConditioningBundle
from src.contracts.config import (
    EvaluationConfig,
    GeneratorConfig,
    PointstreamConfig,
    ResidualConfig,
)
from src.contracts.lattice import (
    STAGE_APPEARANCE,
    STAGE_DETECTION,
    STAGE_GENERATION,
    STAGE_METRICS,
    STAGE_POSE,
    STAGE_RESIDUAL,
    StageLattice,
)
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.runner import lattice_config_from, run


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def append_jsonl_log(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=False) + "\n")


def compute_checkpoint_sha256(ckpt_path: Path) -> str:
    """Compute immutable SHA256 content hash of checkpoint file or directory."""
    if ckpt_path.is_file():
        h = hashlib.sha256()
        with ckpt_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    elif ckpt_path.is_dir():
        h = hashlib.sha256()
        for p in sorted(ckpt_path.rglob("*")):
            if p.is_file():
                h.update(p.relative_to(ckpt_path).as_posix().encode("utf-8"))
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
        return h.hexdigest()
    return "unknown"


def build_eval_generator_ref(
    arch: str,
    checkpoint_path: Path,
    condition_type: str | None = None,
    arch_kwargs: dict[str, Any] | None = None,
    device: str | None = None,
    seed: int | None = None,
) -> Any:
    """Build a runner-compatible GeneratorRef for the given checkpoint."""
    from src.contracts.capabilities import CAP_TEMPORAL_SEQUENCE

    extra = dict(arch_kwargs or {})
    if device is not None and "device" not in extra:
        extra["device"] = device

    backend: Any
    caps: frozenset[str] = frozenset()
    reqs: frozenset[str] = frozenset({"pose", "appearance"})

    if arch == "pix2pix":
        from src.components.generation.pix2pix import Pix2PixGenerator
        backend = Pix2PixGenerator(checkpoint=str(checkpoint_path))
    elif arch == "spade4tennis":
        from src.components.generation.spade import Spade4TennisGenerator
        backend = Spade4TennisGenerator(checkpoint=str(checkpoint_path))
    elif arch in ("controlnet", "pose-controlnet"):
        from src.components.generation.controlnet import ControlNetGenerator
        backend = ControlNetGenerator(
            variant=condition_type or "pose",
            checkpoint=str(checkpoint_path),
            **extra,
        )
    elif arch in ("animate-anyone", "animate_anyone"):
        from src.components.generation.animate_anyone import AnimateAnyoneGenerator
        backend = AnimateAnyoneGenerator(checkpoint=str(checkpoint_path), **extra)
        caps = frozenset({CAP_TEMPORAL_SEQUENCE})
    else:
        raise ValueError(f"Unsupported generator architecture: {arch}")

    return as_runner_ref(backend, name=arch, capabilities=caps, requires=reqs)


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    arch: str,
    manifest: dict[str, Any],
    dataset_root: Path | str,
    img_size: int = 512,
    device: str = "cuda:0",
    metrics: tuple[str, ...] = ("psnr", "ssim", "vmaf"),
    include_lpips: bool = False,
    fps: float = 24.0,
    condition_type: str | None = None,
    arch_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
    residual_settings: dict[str, Any] | None = None,
    fitted_weights_path: Path | str | None = None,
    max_clips: int | None = None,
    held_out_only: bool = True,
    eval_mode: str = "whole_codec",
    allow_synthetic: bool = False,
    runner_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Score checkpoint through the current runner (src.runner.run).

    Evaluates full reconstructed clips, total transmitted bytes, and encode/decode
    runtime against held-out development scenes from the manifest.
    """
    ckpt_file = Path(checkpoint_path).resolve()
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}")

    ckpt_sha = compute_checkpoint_sha256(ckpt_file)
    checkpoint_identity = f"{ckpt_file.name}:{ckpt_sha}"

    # Apply seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Count video-specific fitted weights if provided
    fitted_weight_bytes = 0
    if fitted_weights_path is not None:
        fpath = Path(fitted_weights_path).resolve()
        if fpath.exists():
            fitted_weight_bytes = fpath.stat().st_size

    # Build generator ref with device and seed
    eval_arch_kwargs = dict(arch_kwargs or {})
    ref = build_eval_generator_ref(
        arch=arch,
        checkpoint_path=ckpt_file,
        condition_type=condition_type,
        arch_kwargs=eval_arch_kwargs,
        device=device,
        seed=seed,
    )

    # Select held-out development scenes
    probe_clips = manifest.get("probe_clips", [])
    if held_out_only:
        held_out_set = set(manifest.get("held_out_videos", []))
        clips = [c for c in probe_clips if c.get("video") in held_out_set]
        # Never fall back from an empty held-out selection to all probes!
    else:
        clips = probe_clips

    if max_clips is not None and max_clips > 0:
        clips = clips[:max_clips]

    if not clips:
        # Empty clips must fail closed - cannot succeed or promote!
        return {
            "per_clip": [],
            "source_grouping": {},
            "aggregate": {
                "residual_bytes": float("inf"),
                "total_bytes": float("inf"),
                "video_fitted_weight_bytes": fitted_weight_bytes,
                "psnr_mean": None,
                "ssim_mean": None,
                "vmaf_mean": None,
                "temporal_error": None,
                "encoder_seconds": 0.0,
                "client_seconds": 0.0,
                "checkpoint_identity": checkpoint_identity,
                "success": False,
                "eval_failed": True,
                "error": "empty_clip_selection",
            },
        }

    runner = runner_fn if runner_fn is not None else run
    dataset_path = Path(dataset_root)
    res_settings = residual_settings or {}
    res_codec = res_settings.get("codec", "avc")
    if res_codec in ("libx264", "x264"):
        res_codec = "avc"

    cfg = PointstreamConfig(
        lattice=lattice_config_from(
            StageLattice.of(
                STAGE_DETECTION,
                STAGE_APPEARANCE,
                STAGE_POSE,
                STAGE_GENERATION,
                STAGE_RESIDUAL,
                STAGE_METRICS,
            )
        ),
        generator=GeneratorConfig(backend=arch),
        residual=ResidualConfig(
            codec=res_codec,
            rate=res_settings.get("crf", 28),
            preset=str(res_settings.get("preset", "medium")),
            pix_fmt=str(res_settings.get("pix_fmt", "yuv420p")),
            block_size=int(res_settings.get("block_size", 8)),
            block_threshold=float(res_settings.get("block_threshold", 0.0)),
        ),
        evaluation=EvaluationConfig(metrics=tuple(metrics)),
    )

    per_clip_metrics: list[dict[str, Any]] = []

    for clip in clips:
        video = clip.get("video", "")
        scene = clip.get("scene", "")
        track = clip.get("track", "")
        clip_key = clip.get("key", f"{video}/{scene}/{track}")

        # Injected test fixture support
        is_synthetic = clip.get("is_synthetic_fixture") or clip.get("synthetic") or (allow_synthetic and not (dataset_path / video).exists())
        if is_synthetic:
            frame_ids = sorted(clip.get("frame_ids", [0, 1]))
            n_frames = max(2, len(frame_ids))
            source_frames_list = [np.full((img_size, img_size, 3), 100 + i * 2, dtype=np.uint8) for i in range(n_frames)]
            pose_frames_list = [np.full((img_size, img_size, 3), 50, dtype=np.uint8) for _ in range(n_frames)]
            ref_frame: np.ndarray = source_frames_list[0].copy()
            source_clip = np.stack(source_frames_list, axis=0)
            t, h, w, _ = source_clip.shape
            mask = np.ones((h, w), dtype=bool)
            bbox = (0, 0, w, h)
            obj_requests = []
            for idx in range(t):
                bundle = ConditioningBundle(
                    appearance=as_chw(ref_frame),
                    pose=as_chw(pose_frames_list[idx]),
                    bbox=bbox,
                    frame_index=idx,
                    object_id=track or "player",
                )
                req = ObjectRequest(
                    object_id=track or "player",
                    appearance=ref_frame,
                    bbox=bbox,
                    mask=mask,
                    frame_index=idx,
                    conditioning=bundle,
                )
                obj_requests.append(req)
        else:
            # Real dataset loading: resolve candidate track directories
            track_dir = None
            skel_dir = None
            if clip.get("path") and (dataset_path / clip["path"]).is_dir():
                track_dir = dataset_path / clip["path"]
                skel_dir = track_dir.with_name(f"{track_dir.name}_skeleton")
            elif (dataset_path / video / "segmentations" / scene / track).is_dir():
                track_dir = dataset_path / video / "segmentations" / scene / track
                skel_dir = dataset_path / video / "segmentations" / scene / f"{track}_skeleton"
            elif clip.get("source_track") and Path(clip["source_track"]).is_dir():
                track_dir = Path(clip["source_track"])
                skel_dir = track_dir.with_name(f"{track_dir.name}_skeleton")
            else:
                raise FileNotFoundError(
                    f"Clip {clip_key}: neither {dataset_path / video / 'segmentations' / scene / track} "
                    f"nor {clip.get('path')} could be found"
                )

            if not track_dir.is_dir():
                raise FileNotFoundError(f"Missing crop directory: {track_dir} for clip {clip_key}")
            if not skel_dir.is_dir():
                raise FileNotFoundError(f"Missing skeleton directory: {skel_dir} for clip {clip_key}")

            frame_ids = list(sorted(clip.get("frame_ids", [])))
            global_frame_ids = list(clip.get("global_frame_ids", []))
            if not frame_ids and global_frame_ids:
                frame_ids = list(range(len(global_frame_ids)))
            elif not global_frame_ids and frame_ids:
                global_frame_ids = list(frame_ids)

            from experiments.probe_set.materialize import sorted_frame_files, window_positions
            crop_files_all = sorted_frame_files(track_dir)
            skel_files_all = sorted_frame_files(skel_dir)

            if not crop_files_all:
                raise FileNotFoundError(f"Crop directory {track_dir} contains no frame_*.png files")
            if not skel_files_all:
                raise FileNotFoundError(f"Skeleton directory {skel_dir} contains no frame_*.png files")

            positions: list[int] = []
            if global_frame_ids:
                try:
                    positions = window_positions(crop_files_all, tuple(global_frame_ids))
                except Exception:
                    try:
                        positions = window_positions(crop_files_all, tuple(frame_ids))
                    except Exception as exc:
                        raise FileNotFoundError(
                            f"Clip {clip_key}: frame resolution failed for global {global_frame_ids[:3]} and local {frame_ids[:3]}"
                        ) from exc
            else:
                try:
                    positions = window_positions(crop_files_all, tuple(frame_ids))
                except Exception as exc:
                    raise FileNotFoundError(f"Clip {clip_key}: frame resolution failed for {frame_ids[:3]}") from exc

            crop_paths: list[Path] = []
            skel_paths: list[Path] = []
            for local_idx, pos in enumerate(positions):
                if pos >= len(crop_files_all):
                    raise FileNotFoundError(f"Clip {clip_key}: position {pos} out of range in {track_dir}")
                crop_paths.append(crop_files_all[pos])

                if len(skel_files_all) == len(crop_files_all):
                    skel_paths.append(skel_files_all[pos])
                elif pos < len(skel_files_all):
                    skel_paths.append(skel_files_all[pos])
                elif local_idx < len(skel_files_all):
                    skel_paths.append(skel_files_all[local_idx])
                else:
                    raise FileNotFoundError(f"Clip {clip_key}: cannot pair skeleton for frame at position {pos}")

            crops_rgb: list[np.ndarray] = []
            poses_rgb: list[np.ndarray] = []
            crops_rgba: list[np.ndarray] = []

            for cp, sp in zip(crop_paths, skel_paths):
                if not cp.is_file():
                    raise FileNotFoundError(f"Missing crop file: {cp}")
                if not sp.is_file():
                    raise FileNotFoundError(f"Missing skeleton file: {sp}")

                c_raw = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
                if c_raw is None:
                    raise FileNotFoundError(f"Failed to read crop image {cp}")
                p_raw = cv2.imread(str(sp))
                if p_raw is None:
                    raise FileNotFoundError(f"Failed to read pose image {sp}")

                if c_raw.ndim == 3 and c_raw.shape[2] == 4:
                    crops_rgba.append(c_raw)
                    c_rgb = cv2.cvtColor(c_raw[:, :, :3], cv2.COLOR_BGR2RGB)
                elif c_raw.ndim == 3:
                    c_rgb = cv2.cvtColor(c_raw, cv2.COLOR_BGR2RGB)
                    crops_rgba.append(np.dstack([c_raw, np.full(c_raw.shape[:2], 255, dtype=np.uint8)]))
                else:
                    raise ValueError(f"Invalid crop image format at {cp}")

                p_rgb = cv2.cvtColor(p_raw, cv2.COLOR_BGR2RGB)
                crops_rgb.append(c_rgb)
                poses_rgb.append(p_rgb)

            ref_crop: np.ndarray = crops_rgb[0]

            from src.contracts import paths as ps_paths
            bp46_dir = ps_paths.outputs() / "bp46-long-scenes" / "clips" / video / scene / "extract_24"
            bp21_dir = ps_paths.outputs() / "bp21-headroom" / "clips" / video / scene / "extract_24"
            extract_dir = bp46_dir if bp46_dir.is_dir() else (bp21_dir if bp21_dir.is_dir() else None)

            meta_path = track_dir.parent / f"{track}_metadata.json"
            metadata: list[dict[str, Any]] = []
            if meta_path.is_file():
                try:
                    metadata = json.loads(meta_path.read_text())
                except Exception:
                    pass

            meta_by_fid: dict[int, dict[str, Any]] = {}
            if metadata and isinstance(metadata, list):
                for entry in metadata:
                    if isinstance(entry, dict) and "frame_id" in entry:
                        meta_by_fid[entry["frame_id"]] = entry

            use_full_frame = (eval_mode == "whole_codec") and (extract_dir is not None) and bool(meta_by_fid) and bool(global_frame_ids)

            if use_full_frame and extract_dir is not None:
                full_frames_list = []
                obj_requests = []
                t_count = len(crop_paths)

                sample_full = cv2.imread(str(extract_dir / f"frame_{global_frame_ids[0]:06d}.png"))
                if sample_full is None:
                    raise FileNotFoundError(f"Missing extracted full frame for {global_frame_ids[0]}")
                full_h, full_w = sample_full.shape[:2]

                for idx in range(t_count):
                    gid = global_frame_ids[idx]
                    fpath = extract_dir / f"frame_{gid:06d}.png"
                    if not fpath.is_file():
                        raise FileNotFoundError(f"Missing full frame: {fpath}")
                    fbgr = cv2.imread(str(fpath))
                    if fbgr is None:
                        raise FileNotFoundError(f"Failed to read full frame: {fpath}")
                    frgb = cv2.cvtColor(fbgr, cv2.COLOR_BGR2RGB)
                    full_frames_list.append(frgb)

                    meta_entry = meta_by_fid.get(gid) or (metadata[positions[idx]] if positions[idx] < len(metadata) else None)
                    if meta_entry is None or "bbox" not in meta_entry:
                        raise ValueError(f"Missing bbox in metadata for frame {gid}")

                    bbox_raw = meta_entry["bbox"]
                    x1, y1, x2, y2 = (int(v) for v in bbox_raw)

                    mask_full = np.zeros((full_h, full_w), dtype=bool)
                    crop_a = crops_rgba[idx][:, :, 3] >= 128
                    ch, cw = crop_a.shape[:2]
                    clip_y2 = min(full_h, y1 + ch)
                    clip_x2 = min(full_w, x1 + cw)
                    mask_full[y1:clip_y2, x1:clip_x2] = crop_a[:(clip_y2 - y1), :(clip_x2 - x1)]

                    bundle = ConditioningBundle(
                        appearance=as_chw(ref_crop),
                        pose=as_chw(poses_rgb[idx]),
                        bbox=(x1, y1, x2, y2),
                        frame_index=idx,
                        object_id=track,
                    )
                    req = ObjectRequest(
                        object_id=track,
                        appearance=ref_crop,
                        bbox=(x1, y1, x2, y2),
                        mask=mask_full,
                        frame_index=idx,
                        conditioning=bundle,
                    )
                    obj_requests.append(req)

                source_clip = np.stack(full_frames_list, axis=0)
            else:
                crops_resized = [cv2.resize(c, (img_size, img_size)) for c in crops_rgb]
                poses_resized = [cv2.resize(p, (img_size, img_size)) for p in poses_rgb]
                ref_resized: np.ndarray = crops_resized[0]
                source_clip = np.stack(crops_resized, axis=0)
                t, h, w, _ = source_clip.shape

                obj_requests = []
                for idx in range(t):
                    crop_a = cv2.resize(crops_rgba[idx][:, :, 3], (img_size, img_size)) >= 128
                    bundle = ConditioningBundle(
                        appearance=as_chw(ref_resized),
                        pose=as_chw(poses_resized[idx]),
                        bbox=(0, 0, w, h),
                        frame_index=idx,
                        object_id=track,
                    )
                    req = ObjectRequest(
                        object_id=track,
                        appearance=ref_resized,
                        bbox=(0, 0, w, h),
                        mask=crop_a,
                        frame_index=idx,
                        conditioning=bundle,
                    )
                    obj_requests.append(req)

        run_res = runner(
            cfg,
            [source_clip],
            generator=ref,
            objects=(tuple(obj_requests),),
            context_ids=(clip_key,),
        )

        delivered = run_res.delivered_frames
        psnr_val = None
        try:
            psnr_val = float(run_res.delivered_quality.whole_frame("psnr"))
        except KeyError:
            pass

        ssim_val = None
        try:
            ssim_val = float(run_res.delivered_quality.whole_frame("ssim"))
        except KeyError:
            pass

        vmaf_val = None
        try:
            vmaf_val = float(run_res.delivered_quality.whole_frame("vmaf"))
        except KeyError:
            pass

        t_len = len(source_clip)
        if t_len > 1:
            src_diff = np.mean(np.abs(source_clip[1:].astype(float) - source_clip[:-1].astype(float)))
            del_diff = np.mean(np.abs(delivered[1:].astype(float) - delivered[:-1].astype(float)))
            temporal_error = float(abs(src_diff - del_diff))
        else:
            temporal_error = 0.0

        clip_record = {
            "clip_key": clip_key,
            "video": video,
            "scene": scene,
            "track": track,
            "residual_bytes": int(run_res.sizes.residual),
            "transport_total": int(run_res.sizes.transport_total),
            "psnr": psnr_val,
            "ssim": ssim_val,
            "vmaf": vmaf_val,
            "temporal_error": temporal_error,
            "encoder_seconds": float(run_res.encoder_seconds),
            "client_seconds": float(run_res.client_seconds),
        }
        per_clip_metrics.append(clip_record)

    total_residual_bytes = sum(c["residual_bytes"] for c in per_clip_metrics)
    total_transport_bytes = sum(c["transport_total"] for c in per_clip_metrics) + fitted_weight_bytes

    psnr_vals = [c["psnr"] for c in per_clip_metrics if c["psnr"] is not None]
    psnr_mean = float(np.mean(psnr_vals)) if psnr_vals else None

    ssim_vals = [c["ssim"] for c in per_clip_metrics if c["ssim"] is not None]
    ssim_mean = float(np.mean(ssim_vals)) if ssim_vals else None

    vmaf_vals = [c["vmaf"] for c in per_clip_metrics if c["vmaf"] is not None]
    vmaf_mean = float(np.mean(vmaf_vals)) if vmaf_vals else None

    temporal_errors = [c["temporal_error"] for c in per_clip_metrics]
    temporal_error_mean = float(np.mean(temporal_errors)) if temporal_errors else 0.0

    total_encoder_seconds = sum(c["encoder_seconds"] for c in per_clip_metrics)
    total_client_seconds = sum(c["client_seconds"] for c in per_clip_metrics)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for c in per_clip_metrics:
        vid = c.get("video", "unknown")
        by_source.setdefault(vid, []).append(c)

    source_grouping: dict[str, dict[str, Any]] = {}
    for vid, clips_in_vid in by_source.items():
        source_grouping[vid] = {
            "num_clips": len(clips_in_vid),
            "residual_bytes": sum(c["residual_bytes"] for c in clips_in_vid),
            "total_bytes": sum(c["transport_total"] for c in clips_in_vid),
            "psnr_mean": float(np.mean([c["psnr"] for c in clips_in_vid if c["psnr"] is not None])) if any(c["psnr"] is not None for c in clips_in_vid) else None,
            "ssim_mean": float(np.mean([c["ssim"] for c in clips_in_vid if c["ssim"] is not None])) if any(c["ssim"] is not None for c in clips_in_vid) else None,
            "vmaf_mean": float(np.mean([c["vmaf"] for c in clips_in_vid if c["vmaf"] is not None])) if any(c["vmaf"] is not None for c in clips_in_vid) else None,
            "temporal_error": float(np.mean([c["temporal_error"] for c in clips_in_vid])),
        }

    aggregate = {
        "residual_bytes": total_residual_bytes,
        "total_bytes": total_transport_bytes,
        "video_fitted_weight_bytes": fitted_weight_bytes,
        "psnr_mean": psnr_mean,
        "ssim_mean": ssim_mean,
        "vmaf_mean": vmaf_mean,
        "temporal_error": temporal_error_mean,
        "lpips_vgg_uncalibrated": None,
        "encoder_seconds": total_encoder_seconds,
        "client_seconds": total_client_seconds,
        "checkpoint_identity": checkpoint_identity,
        "device": device,
        "seed": seed,
        "per_clip_count": len(per_clip_metrics),
        "success": True,
    }

    return {"per_clip": per_clip_metrics, "source_grouping": source_grouping, "aggregate": aggregate}


LOWER_IS_BETTER = {"temporal_error", "lpips_vgg_uncalibrated"}
HIGHER_IS_BETTER = {"psnr_mean", "ssim_mean", "vmaf_mean"}
RANKED_METRICS = tuple(sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER))

#: The project's ranking currency. Under the Residual Guarantee a component is
#: justified only by shrinking the residual payload by more than the metadata it
#: adds, so the winner is the checkpoint whose output is cheapest to *correct* --
#: not the one that scores best perceptually. The two genuinely disagree: an
#: aligned but blurry prediction leaves a smooth residual that codes cheaply,
#: while a sharp but misaligned one leaves expensive double edges.
PRIMARY_METRIC = "residual_bytes"


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    arch: str  # eval_checkpoint.py --arch
    kind: str  # "pix2pix" | "spade4tennis" | "controlnet"
    condition_type: str | None = None  # controlnet only
    model_size: str = "lite"  # spade4tennis only


def default_variants() -> list[Variant]:
    """The 3 wrapped candidates (report 10 Phase 5.4(b)); Animate-Anyone
    deliberately not included yet — see module docstring."""
    return [
        Variant(name="pix2pix", arch="pix2pix", kind="pix2pix"),
        Variant(name="spade4tennis_lite", arch="spade4tennis", kind="spade4tennis", model_size="lite"),
        Variant(name="controlnet_pose", arch="controlnet", kind="controlnet", condition_type="pose"),
    ]


def checkpoint_path_for_eval(variant: Variant, ckpt_dir: Path) -> Path:
    if variant.kind == "pix2pix":
        return ckpt_dir / "pix2pix_generator.pt"
    if variant.kind == "spade4tennis":
        return ckpt_dir / f"spade4tennis_{variant.model_size}_generator.pt"
    if variant.kind == "controlnet":
        return ckpt_dir  # diffusers ControlNetModel.from_pretrained expects a directory
    raise ValueError(f"unknown variant kind: {variant.kind}")


# ---------------------------------------------------------------------------
# Command building (pure — no subprocess execution, so this is unit-testable)
# ---------------------------------------------------------------------------


def build_train_command(
    variant: Variant,
    data_root: Path,
    ckpt_dir: Path,
    cumulative_epochs: int,
    rung_epochs: int,
    resume: bool,
    python_bin: str | None = None,
    batch_size: str | None = None,
) -> list[str]:
    """Builds the training-script invocation for one variant's rung.

    pix2pix/spade4tennis express "how far to train" as an *absolute* epoch
    target plus --resume (their own scripts track how many epochs a
    checkpoint has already seen and pick up from there). ControlNet's script
    has no such numeric resume convention — each invocation trains
    `rung_epochs` *more* epochs on top of whatever `--controlnet-model-id`
    points at (the previous rung's saved directory), or starts
    `--from-scratch` on rung 0.

    `batch_size` (str, since pix2pix/spade4tennis accept "auto") is passed
    through as-is when given; left at each script's own default otherwise.
    Needed for tiny smoke-test datasets — pix2pix/spade4tennis's "auto"
    default (64/32) plus drop_last=True silently empties the dataloader
    when the dataset has fewer images than that.
    """
    py = python_bin or sys.executable

    if variant.kind == "pix2pix":
        cmd = [
            py, "scripts/train_pix2pix.py",
            "--data-root", str(data_root),
            "--epochs", str(cumulative_epochs),
            "--out-weights", str(ckpt_dir / "pix2pix_generator.pt"),
            "--checkpoint-path", str(ckpt_dir / "pix2pix_checkpoint.pt"),
            "--sample-dir", str(ckpt_dir / "samples"),
        ]
        if batch_size is not None:
            cmd.extend(["--batch-size", batch_size])
        if resume:
            cmd.append("--resume")
        return cmd

    if variant.kind == "spade4tennis":
        cmd = [
            py, "scripts/train_spade4tennis.py",
            "--model-size", variant.model_size,
            "--data-root", str(data_root),
            "--epochs", str(cumulative_epochs),
            "--out-weights", str(ckpt_dir / f"spade4tennis_{variant.model_size}_generator.pt"),
            "--checkpoint-path", str(ckpt_dir / f"spade4tennis_{variant.model_size}_checkpoint.pt"),
            "--sample-dir", str(ckpt_dir / "samples"),
        ]
        if batch_size is not None:
            cmd.extend(["--batch-size", batch_size])
        if resume:
            cmd.append("--resume")
        return cmd

    if variant.kind == "controlnet":
        cmd = [
            py, "scripts/train_controlnet.py",
            "--data-root", str(data_root),
            "--condition-type", variant.condition_type or "pose",
            "--epochs", str(rung_epochs),
            "--output-dir", str(ckpt_dir),
        ]
        if resume:
            cmd.extend(["--controlnet-model-id", str(ckpt_dir)])
        else:
            cmd.append("--from-scratch")
        return cmd

    raise ValueError(f"unknown variant kind: {variant.kind}")


# ---------------------------------------------------------------------------
# Non-overlap safety net
# ---------------------------------------------------------------------------


def verify_data_root_excludes_probe_set(data_root: Path, manifest: dict) -> list[str]:
    """Returns the list of probe-set keys that ARE reachable under data_root
    (i.e. leaked into training) — empty means the exclusion held."""
    violations = []
    for key in manifest["excluded_training_keys"]:
        video, scene, track = key.split("/")
        if (data_root / video / "segmentations" / scene / track).exists():
            violations.append(key)
    return violations


# ---------------------------------------------------------------------------
# Ranking / successive halving
# ---------------------------------------------------------------------------


def is_valid_eval(agg: dict[str, Any] | None) -> bool:
    """Return True if an evaluation result is valid, finite, and well-accounted."""
    if not isinstance(agg, dict):
        return False
    if agg.get("success") is False:
        return False
    if agg.get("eval_failed") is True:
        return False
    if agg.get("status") in ("failed", "error"):
        return False
    if agg.get("per_clip_count") is not None and agg["per_clip_count"] <= 0:
        return False
    if agg.get("total_bytes") is not None and agg.get("residual_bytes") is not None:
        if agg["total_bytes"] < agg["residual_bytes"]:
            return False
    if "checkpoint_identity" in agg:
        ckpt_id = agg["checkpoint_identity"]
        if not ckpt_id or not isinstance(ckpt_id, str) or not ckpt_id.strip():
            return False
    # Check numeric values for NaN or Inf
    for val in agg.values():
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return False
    return True


def rank_variants(aggregate_by_variant: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, float]]:
    """Rank candidates, primarily by residual bytes.

    `PRIMARY_METRIC` decides whenever every scored variant has it: fewest bytes
    wins, because that is the quantity the Residual Guarantee is denominated in.
    The perceptual composite is retained only as a fallback (for runs scored with
    `--no-residual-bytes`) and as a diagnostic that explains *why* a variant won
    -- it must never silently outvote the payload.

    Failed evaluations (success=False, eval_failed=True, non-finite or missing
    metrics) are strictly ordered after all successful variants and cannot win.

    Returns (ranked_names, composite_scores). The composite is always reported so
    a bytes-win with poor perceptual scores is visible rather than hidden.
    """
    valid_variants = [v for v in aggregate_by_variant if is_valid_eval(aggregate_by_variant[v])]
    invalid_variants = [v for v in aggregate_by_variant if v not in valid_variants]

    normalized: dict[str, dict[str, float]] = {v: {} for v in aggregate_by_variant}

    for key in RANKED_METRICS:
        values = {
            v: aggregate_by_variant[v][key]
            for v in valid_variants
            if aggregate_by_variant[v].get(key) is not None
        }
        if len(values) < 2:
            continue
        lo, hi = min(values.values()), max(values.values())
        span = hi - lo
        for v, val in values.items():
            norm = 0.5 if span == 0 else (val - lo) / span
            if key in LOWER_IS_BETTER:
                norm = 1.0 - norm
            normalized[v][key] = norm

    composite = {
        v: (sum(scores.values()) / len(scores) if scores else (0.0 if v in valid_variants else -1.0))
        for v, scores in normalized.items()
    }

    payloads = {
        v: aggregate_by_variant[v][PRIMARY_METRIC]
        for v in valid_variants
        if aggregate_by_variant[v].get(PRIMARY_METRIC) is not None
    }
    if payloads and len(payloads) == len(valid_variants):
        # Fewest bytes wins; composite breaks exact ties only.
        valid_ranked = sorted(valid_variants, key=lambda v: (payloads[v], -composite[v], v))
    else:
        valid_ranked = sorted(valid_variants, key=lambda v: (-composite[v], v))

    invalid_ranked = sorted(invalid_variants, key=lambda v: (composite.get(v, -1.0), v))
    ranked = valid_ranked + invalid_ranked
    return ranked, composite


def promote_survivors(
    ranked: list[str],
    aggregate_by_variant: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Select survivors for the next rung (top half).

    If aggregate_by_variant is provided, unsuccessful evaluations are never
    promoted as survivors.
    """
    if aggregate_by_variant is not None:
        valid_ranked = [v for v in ranked if is_valid_eval(aggregate_by_variant.get(v))]
        if not valid_ranked:
            return []
        keep = math.ceil(len(valid_ranked) / 2)
        return valid_ranked[:keep]

    if len(ranked) <= 1:
        return ranked
    keep = math.ceil(len(ranked) / 2)
    return ranked[:keep]


# ---------------------------------------------------------------------------
# Campaign state (JSON on disk, so a rung can be re-run/resumed by invocation)
# ---------------------------------------------------------------------------


def init_state(variants: list[Variant]) -> dict[str, Any]:
    return {
        "rung": 0,
        "alive": [v.name for v in variants],
        "cumulative_epochs": {v.name: 0 for v in variants},
        "variants": {v.name: asdict(v) for v in variants},
        "history": [],
    }


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def variants_from_state(state: dict[str, Any]) -> dict[str, Variant]:
    return {name: Variant(**fields) for name, fields in state["variants"].items()}


# ---------------------------------------------------------------------------
# Rung execution (subprocess for training; in-process for scoring)
# ---------------------------------------------------------------------------


def halved_batch_size(batch_size: str | None) -> str | None:
    """Halve a numeric --train-batch-size for a retry; leave non-numeric values (e.g. "auto") unchanged."""
    if batch_size is None or not batch_size.isdigit():
        return batch_size
    return str(max(1, int(batch_size) // 2))


def run_training_subprocess(cmd: list[str], log_path: Path, timeout: float | None, repo_root: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, cwd=str(repo_root), stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode


def eval_variant(
    variant: Variant,
    ckpt_dir: Path,
    manifest: dict,
    eval_dataset_root: Path,
    step: int,
    rung: int,
    log_path: Path,
    device: str,
    img_size: int,
    metrics: tuple[str, ...],
    include_lpips: bool,
    eval_steps: int,
    residual_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint_path_for_eval(variant, ckpt_dir)
    condition_type = variant.condition_type if variant.kind == "controlnet" else None

    arch_kwargs = {"num_inference_steps": eval_steps} if variant.kind == "controlnet" else {}

    result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        arch=variant.arch,
        manifest=manifest,
        dataset_root=eval_dataset_root,
        img_size=img_size,
        device=device,
        metrics=metrics,
        include_lpips=include_lpips,
        fps=24.0,
        condition_type=condition_type,
        arch_kwargs=arch_kwargs,
        residual_settings=residual_settings,
    )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rung": rung,
        "step": step,
        "variant": variant.name,
        "arch": variant.arch,
        "eval_steps": eval_steps,
        "checkpoint": str(checkpoint),
        **result["aggregate"],
    }
    append_jsonl_log(log_path, record)
    return result["aggregate"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Probe-excluded training view (--data-root for the wrapped training scripts)")
    parser.add_argument("--eval-dataset-root", type=Path, default=Path("assets/dataset"), help="Full (unfiltered) dataset root, for reading probe-clip ground truth")
    parser.add_argument("--initial-epochs", type=int, default=2)
    parser.add_argument("--max-rungs", type=int, default=None)
    parser.add_argument("--eval-img-size", type=int, default=512, help="Eval output resolution (square)")
    parser.add_argument("--eval-steps", type=int, default=10, help="Inference steps for diffusion variants during campaign eval")
    parser.add_argument("--eval-metrics", type=str, default="psnr,ssim,vmaf", help="Comma separated metrics")
    # Residual bytes is the ranking currency (PRIMARY_METRIC). Disabling it
    # falls the ranking back to the perceptual composite, which is a weaker
    # basis for an architecture decision -- see rank_variants.
    parser.add_argument("--no-residual-bytes", action="store_true",
                        help="Skip residual-byte pricing and rank on the perceptual composite instead")
    parser.add_argument("--residual-codec", type=str, default="libx264")
    parser.add_argument("--residual-crf", type=int, default=28)
    parser.add_argument("--residual-preset", type=str, default="medium")
    parser.add_argument("--residual-pix-fmt", type=str, default="yuv444p")
    parser.add_argument("--residual-block-size", type=int, default=8)
    parser.add_argument("--residual-block-threshold", type=float, default=0.0)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-timeout-sec", type=float, default=None)
    parser.add_argument("--train-batch-size", type=str, default=None, help="Passed through to pix2pix/spade4tennis --batch-size (e.g. for tiny smoke datasets)")
    parser.add_argument("--auto-continue", action="store_true", help="Run all remaining rungs unattended (NOT for harness validation — see module docstring)")
    parser.add_argument(
        "--prune-on-train-failure",
        action="store_true",
        help="Rank/prune a rung even when a variant's training subprocess failed (old, dangerous "
        "behavior — an infra failure such as CUDA OOM would be scored as a quality loss). Default "
        "off: a training failure that survives one halved-batch-size retry aborts the rung without "
        "ranking or mutating state, so a human can fix the infra issue and resume.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print training commands without executing them or scoring")
    parser.add_argument(
        "--restore-spade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore spade4tennis_lite to alive candidates if loading an existing campaign state "
        "where it was prematurely pruned (default True).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated subset of default_variants() names to run (default: all 3). "
        "Useful for a cheap smoke test (e.g. 'pix2pix,spade4tennis_lite' to skip the heavier ControlNet path).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    violations = verify_data_root_excludes_probe_set(args.data_root, manifest)
    if violations:
        parser.error(
            f"--data-root {args.data_root} leaks {len(violations)} probe-set track(s) into training "
            f"(e.g. {violations[:3]}) — refusing to start. Use scripts/select_probe_set.py's "
            f"--materialize-training-view to build a clean --data-root."
        )

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = tuple(m.strip().lower() for m in args.eval_metrics.split(",") if m.strip())
    residual_settings = None if args.no_residual_bytes else {
        "codec": args.residual_codec,
        "crf": args.residual_crf,
        "preset": args.residual_preset,
        "pix_fmt": args.residual_pix_fmt,
        "block_size": args.residual_block_size,
        "block_threshold": args.residual_block_threshold,
    }

    state_path = args.campaign_dir / "campaign_state.json"
    log_path = args.campaign_dir / "probe_log.jsonl"

    if state_path.exists():
        state = load_state(state_path)
        if args.restore_spade:
            if "spade4tennis_lite" not in state.get("alive", []):
                all_variants = default_variants()
                spade_variant = next((v for v in all_variants if v.name == "spade4tennis_lite"), None)
                if spade_variant is not None:
                    if "spade4tennis_lite" not in state.setdefault("variants", {}):
                        state["variants"]["spade4tennis_lite"] = asdict(spade_variant)
                    state.setdefault("cumulative_epochs", {}).setdefault("spade4tennis_lite", 0)
                    state["alive"].append("spade4tennis_lite")
                    print("Restored spade4tennis_lite to alive candidates in campaign state.")
                    save_state(state_path, state)
    else:
        variants = default_variants()
        if args.variants:
            requested = [name.strip() for name in args.variants.split(",") if name.strip()]
            by_name = {v.name: v for v in variants}
            unknown = [name for name in requested if name not in by_name]
            if unknown:
                parser.error(f"unknown --variants entries {unknown}; choices are {sorted(by_name)}")
            variants = [by_name[name] for name in requested]
        state = init_state(variants)
        save_state(state_path, state)

    variant_by_name = variants_from_state(state)
    repo_root = Path(__file__).resolve().parents[1]

    while True:
        if not state["alive"]:
            print("Campaign halted: no alive candidates remaining.")
            break
        if len(state["alive"]) == 1 and (args.max_rungs is None or state["rung"] >= args.max_rungs):
            print(f"Campaign converged: survivor = {state['alive']}")
            break
        if args.max_rungs is not None and state["rung"] >= args.max_rungs:
            print(f"Reached --max-rungs={args.max_rungs}; stopping with alive={state['alive']}")
            break

        rung = state["rung"]
        # All alive variants share the same cumulative target by construction (see docstring step 5).
        prev_cumulative = max(state["cumulative_epochs"].get(name, 0) for name in state["alive"])
        target_epochs = args.initial_epochs if (rung == 0 or prev_cumulative == 0) else prev_cumulative * 2

        aggregate_by_variant: dict[str, dict[str, Any]] = {}
        rung_train_failed = False
        for name in state["alive"]:
            variant = variant_by_name[name]
            ckpt_dir = args.campaign_dir / "checkpoints" / name
            prev = state["cumulative_epochs"][name]
            resume = prev > 0
            rung_epochs = target_epochs - prev

            if rung_epochs <= 0:
                print(f"Rung {rung}: {name} already has {prev} cumulative epoch(s) (target {target_epochs}); skipping training.")
                returncode = 0
            else:
                batch_size = args.train_batch_size
                cmd = build_train_command(
                    variant, args.data_root, ckpt_dir, target_epochs, rung_epochs, resume,
                    batch_size=batch_size,
                )

                if args.dry_run:
                    print(f"[dry-run] rung {rung} {name}: {' '.join(cmd)}")
                    continue

                print(f"Rung {rung}: training {name} to {target_epochs} cumulative epoch(s)...")
                log_path_variant = args.campaign_dir / "logs" / f"{name}_rung{rung}.log"
                returncode = run_training_subprocess(cmd, log_path_variant, args.train_timeout_sec, repo_root)

            if returncode != 0:
                retry_batch_size = halved_batch_size(batch_size)
                print(
                    f"WARNING: {name} training subprocess exited {returncode}; retrying once with "
                    f"--train-batch-size {retry_batch_size} (was {batch_size}) in case this was a "
                    "shared-GPU OOM, not a real quality failure."
                )
                retry_cmd = build_train_command(
                    variant, args.data_root, ckpt_dir, target_epochs, rung_epochs, resume,
                    batch_size=retry_batch_size,
                )
                returncode = run_training_subprocess(
                    retry_cmd, args.campaign_dir / "logs" / f"{name}_rung{rung}_retry.log",
                    args.train_timeout_sec, repo_root,
                )

            if returncode != 0:
                print(f"ERROR: {name} training failed twice this rung (exit {returncode}).")
                if args.prune_on_train_failure:
                    print(f"--prune-on-train-failure set: skipping {name}'s eval and continuing this rung.")
                    rung_train_failed = True
                    continue
                print(
                    f"Aborting rung {rung} without ranking or pruning — a training failure is not a "
                    f"quality result. Fix the failure (see {log_path_variant.with_name(log_path_variant.stem + '_retry.log')}) "
                    "and resume by re-running this command; state is unchanged."
                )
                return 1

            aggregate_by_variant[name] = eval_variant(
                variant, ckpt_dir, manifest, args.eval_dataset_root, target_epochs, rung, log_path,
                device, args.eval_img_size, metrics, not args.skip_lpips, args.eval_steps,
                residual_settings=residual_settings,
            )
            state["cumulative_epochs"][name] = target_epochs

        if args.dry_run:
            print("[dry-run] stopping before eval/ranking")
            break

        if not aggregate_by_variant:
            print("No variant produced a scorable checkpoint this rung; stopping.")
            break

        if rung_train_failed:
            print(
                "NOTE: this rung's ranking excludes at least one variant that failed training twice "
                "(--prune-on-train-failure was set, so it will be pruned as if it lost on quality)."
            )

        ranked, composite = rank_variants(aggregate_by_variant)
        survivors = promote_survivors(ranked, aggregate_by_variant)
        pruned = [v for v in state["alive"] if v not in survivors]

        state["history"].append({
            "rung": rung,
            "target_epochs": target_epochs,
            "aggregate_by_variant": aggregate_by_variant,
            "composite_scores": composite,
            "ranked": ranked,
            "survivors": survivors,
            "pruned": pruned,
        })
        state["alive"] = survivors
        state["rung"] = rung + 1
        save_state(state_path, state)

        print(f"Rung {rung} complete. Ranked (best first): {ranked}")
        print(f"Composite scores: {composite}")
        if pruned:
            print(f"Pruned: {pruned}")
        print(f"Survivors advancing to rung {rung + 1} (target {target_epochs * 2} cumulative epochs): {survivors}")

        if not args.auto_continue:
            print(
                "Stopping after one rung (default — a human should review the probe-log curve "
                f"at {log_path} before authorizing the next rung). Re-run with the same "
                "--campaign-dir to continue, or pass --auto-continue to run unattended."
            )
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
