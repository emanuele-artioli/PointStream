"""Diagnostic Matrix: Generation (OFF/ON) x Residual (OFF/ON) with controls.

Evaluates the ablation lattice plus a shuffled-conditioning generation null:
1. [Gen OFF, Res OFF]: Pasted-reference control without residual correction.
2. [Gen OFF, Res ON]:  Pasted-reference control with residual correction.
3. [Gen ON, Res OFF]:  Generative synthesis without residual correction.
4. [Gen ON, Res ON]:   Generative synthesis with residual correction.
5. [Gen ON, shuffled]: Same generator/checkpoint, permuted/foreign pose.

The saved JSON records identity (revision, source hashes, config, checkpoint
SHA), byte parts, timings, per-corner failures, and whether generation actually
changed delivered pixels. Reuse of a prior corner requires that whole identity.
"""
from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from experiments.long_scenes.loader import LongSceneClip, load_long_scene_clip
from experiments.tier.diagnostic_report import (
    REQUIRED_REPORT_KEYS,
    actual_timing,
    assess_generator_comparison,
    build_run_identity,
    extract_byte_subledger,
    extract_size_parts,
    failure_record,
    generation_effect,
    git_revision,
    inference_parameters,
    per_frame_sha256,
    resolved_configuration,
    reusable_corners,
    sha256_path,
    source_manifest,
    wire_reconciliation,
    wrap_generator_with_counter,
)
from experiments.tier.low_rate_measure import score_headlines
from src.contracts import paths as ps_paths
from src.contracts.config import PointstreamConfig
from src.contracts.lattice import (
    STAGE_APPEARANCE,
    STAGE_BACKGROUND,
    STAGE_CODEC,
    STAGE_DETECTION,
    STAGE_GENERATION,
    STAGE_METRICS,
    STAGE_MOTION,
    STAGE_POSE,
    STAGE_RESIDUAL,
    STAGE_SEGMENTATION,
    STAGE_SELECTION,
    STAGE_TEMPORAL,
    STAGE_TRACKING,
    STAGE_TRANSPORT,
    StageLattice,
)
from src.runner.config_io import load_tier


RunFn = Callable[..., Any]
ScoreFn = Callable[[np.ndarray, np.ndarray], dict[str, Any]]
GeneratorFactory = Callable[..., Any]

DEFAULT_CHECKPOINT_BY_ARCH: dict[str, Path] = {
    "pix2pix": Path("weights") / "pix2pix_generator.pt",
    "spade4tennis": Path("weights") / "spade4tennis_lite_generator.pt",
}


def default_checkpoint_for(arch: str) -> Path:
    weights_dir = ps_paths.assets() / "weights"
    relative = DEFAULT_CHECKPOINT_BY_ARCH.get(arch)
    if relative is not None:
        return weights_dir / relative.name
    if arch == "animate-anyone":
        return Path("/home/itec/emanuele/Models/AnimateAnyone/profiles/finetuned_tennis")
    raise ValueError(f"Unknown generator arch: {arch}")


def _build_lattice(*, gen_on: bool, res_on: bool) -> StageLattice:
    stages = [
        STAGE_BACKGROUND,
        STAGE_DETECTION,
        STAGE_SELECTION,
        STAGE_TRACKING,
        STAGE_APPEARANCE,
        STAGE_MOTION,
        STAGE_TEMPORAL,
        STAGE_SEGMENTATION,
        STAGE_CODEC,
        STAGE_TRANSPORT,
        STAGE_METRICS,
    ]
    if gen_on:
        stages.extend([STAGE_POSE, STAGE_GENERATION])
    if res_on:
        stages.append(STAGE_RESIDUAL)
    return StageLattice.of(*stages)


def resolve_clip_start_frame(clip: Any, n_frames: int | None = None) -> int:
    """Resolve the verified start_frame offset from clip or manifest.

    Never infer or guess global frame IDs.
    """
    if hasattr(clip, "start_frame") and clip.start_frame is not None and clip.start_frame > 0:
        return int(clip.start_frame)

    video = getattr(clip, "video", None)
    scene = getattr(clip, "scene", None)
    if not video or not scene:
        if hasattr(clip, "start_frame") and clip.start_frame is not None:
            return int(clip.start_frame)
        raise ValueError("Cannot resolve start_frame: clip is missing video/scene")

    from experiments.long_scenes.loader import get_long_scene_manifest

    try:
        manifest = get_long_scene_manifest()
        for s in manifest.get("scenes", []):
            if s.get("video") == video and s.get("scene") == scene:
                target_count = n_frames or getattr(clip, "n_frames", None) or len(getattr(clip, "frames", []))
                intervals = s.get("intervals", {})
                if str(target_count) in intervals:
                    return int(intervals[str(target_count)].get("start_frame", 0))
                for span in ("48", "96", "192", "384"):
                    if span in intervals and "start_frame" in intervals[span]:
                        return int(intervals[span]["start_frame"])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve verified start_frame for {video}/{scene} from manifest: {exc}"
        ) from exc

    if hasattr(clip, "start_frame") and clip.start_frame is not None:
        return int(clip.start_frame)

    raise ValueError(
        f"Cannot resolve verified source coordinates (start_frame) for {video}/{scene}"
    )


def _augment_objects_with_pose(clip: Any, *, shuffle: bool, seed: int) -> tuple[Any, ...]:
    from src.contracts.conditioning import ConditioningBundle
    import cv2

    objects = tuple(clip.objects)
    if not objects:
        return objects

    start_frame = resolve_clip_start_frame(clip, n_frames=len(clip.frames))
    dataset_scene_dir = ps_paths.assets() / "dataset" / clip.video / "segmentations" / clip.scene
    poses: list[np.ndarray] = []
    augmented: list[Any] = []
    for obj in objects:
        obj_id = obj.object_id
        abs_frame = start_frame + obj.frame_index
        skel_dir = dataset_scene_dir / f"{obj_id}_skeleton"
        if not skel_dir.is_dir():
            raise FileNotFoundError(
                f"Missing required pose conditioning skeleton directory for {clip.video}/{clip.scene} object {obj_id} at {skel_dir}"
            )
        pose_path = skel_dir / f"frame_{abs_frame:06d}.png"
        if not pose_path.is_file():
            raise FileNotFoundError(
                f"Missing required pose conditioning skeleton for {clip.video}/{clip.scene} object {obj_id} frame {abs_frame} at {pose_path}"
            )
        skel_bgr = cv2.imread(str(pose_path))
        if skel_bgr is None:
            raise ValueError(f"Failed to load skeleton image at {pose_path}")
        skel_img = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2RGB)
        height, width = obj.appearance.shape[:2]
        if skel_img.shape[:2] != (height, width):
            raise ValueError(
                f"Misaligned pose conditioning: skeleton shape {skel_img.shape[:2]} != appearance shape {(height, width)} for object {obj_id} frame {abs_frame}"
            )
        poses.append(skel_img)
        bundle = ConditioningBundle(
            appearance=np.transpose(obj.appearance, (2, 0, 1)),
            pose=np.transpose(skel_img, (2, 0, 1)),
            mask=obj.mask,
            bbox=obj.bbox,
            frame_index=obj.frame_index,
            object_id=obj.object_id,
        )
        augmented.append(replace(obj, conditioning=bundle))

    if not shuffle:
        return tuple(augmented)

    shuffled_poses = _permute_or_foreign_poses(poses, seed=seed)
    out: list[Any] = []
    for obj, pose_img in zip(augmented, shuffled_poses, strict=True):
        bundle = obj.conditioning
        assert bundle is not None
        out.append(
            replace(
                obj,
                conditioning=replace(bundle, pose=np.transpose(pose_img, (2, 0, 1))),
            )
        )
    return tuple(out)


def _permute_or_foreign_poses(poses: list[np.ndarray], *, seed: int) -> list[np.ndarray]:
    if not poses:
        return poses
    if len(poses) == 1:
        pose = np.ascontiguousarray(poses[0])
        rolled = np.roll(pose, shift=max(1, pose.shape[1] // 3), axis=1)
        if np.array_equal(rolled, pose):
            rolled = np.flip(pose, axis=1).copy()
        return [rolled]
    rng = np.random.default_rng(seed)
    order = np.arange(len(poses))
    perm = rng.permutation(len(poses))
    if np.array_equal(perm, order):
        perm = np.roll(order, 1)
    return [poses[int(index)] for index in perm]


def run_diagnostic_corner(
    corner_name: str,
    clip: Any,
    base_config: PointstreamConfig,
    *,
    gen_on: bool,
    res_on: bool,
    generator_arch: str = "pix2pix",
    generator_checkpoint: Path | None = None,
    residual_qp: int = 38,
    shuffle_conditioning: bool = False,
    seed: int | None = None,
    device: str = "cpu",
    run_fn: RunFn | None = None,
    score_fn: ScoreFn | None = None,
    generator_factory: GeneratorFactory | None = None,
) -> dict[str, Any]:
    print("\n=======================================================")
    print(f"Running Diagnostic Corner: {corner_name}")
    gen_label = "OFF (Pasted-Reference)"
    if gen_on:
        gen_label = "ON (" + generator_arch + ")"
        if shuffle_conditioning:
            gen_label += ", shuffled conditioning"
    print(
        f"Generation: {gen_label} | Residual: "
        f"{'ON (QP ' + str(residual_qp) + ')' if res_on else 'OFF'}"
    )
    print("=======================================================")

    source_frames = np.asarray(clip.frames)
    base_hashes = per_frame_sha256(source_frames)
    run_seed = int(base_config.run.seed if seed is None else seed)
    cfg = replace(base_config, run=replace(base_config.run, seed=run_seed))

    from src.runner.routing import lattice_config_from

    lattice = _build_lattice(gen_on=gen_on, res_on=res_on)
    cfg = replace(cfg, lattice=lattice_config_from(lattice))

    checkpoint_path = generator_checkpoint
    checkpoint_sha: str | None = None
    generator_ref: Any = None
    invocation_counter = [0]
    objects_for_run = clip.objects

    if gen_on:
        cfg = replace(cfg, generator=replace(cfg.generator, backend=generator_arch))
        if checkpoint_path is None:
            checkpoint_path = default_checkpoint_for(generator_arch)
        checkpoint_path = Path(checkpoint_path)
        cfg = replace(
            cfg,
            generator=replace(cfg.generator, checkpoint=str(checkpoint_path)),
        )
        checkpoint_sha = sha256_path(checkpoint_path)
        objects_for_run = _augment_objects_with_pose(
            clip, shuffle=shuffle_conditioning, seed=run_seed
        )

    if res_on:
        res_cfg = replace(
            cfg.residual,
            codec="avc",
            rate=residual_qp,
            background_downscale=1,
            block_threshold=0.0,
        )
        cfg = replace(cfg, residual=res_cfg)

    execute: RunFn
    if run_fn is None:
        from src.runner.run import run as imported_run

        execute = imported_run
    else:
        execute = run_fn
    score: ScoreFn = score_fn if score_fn is not None else score_headlines

    report: dict[str, Any] = {
        "corner": corner_name,
        "generation_on": gen_on,
        "residual_on": res_on,
        "shuffled_conditioning": bool(shuffle_conditioning),
        "generator_arch": generator_arch if gen_on else "none (pasted_reference_control)",
        "residual_qp": residual_qp if res_on else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": checkpoint_sha,
        "seed": run_seed,
        "device": device,
        "inference_parameters": inference_parameters(cfg),
        "resolved_configuration": resolved_configuration(cfg),
        "base_frame_hashes": base_hashes,
        "delivered_frame_hashes": [],
        "coded_bytes": None,
        "parts": {key: 0 for key in ("residual", "panorama", "actor_reference", "metadata", "transport_total")},
        "byte_subledger": None,
        "wire_reconciliation": {
            "wire_request_present": False,
            "transport_total": 0,
            "wire_bytes": None,
            "matched": None,
            "verdict": "not_run",
        },
        "scores": {},
        "timing": {
            "wall_seconds": None,
            "encoder_seconds": None,
            "client_seconds": None,
            "evaluation_seconds": None,
        },
        "delivered_shape": None,
        "model_invocation_count": 0,
        "failure": None,
        "control": (
            "shuffled_conditioning"
            if shuffle_conditioning
            else ("pasted_reference" if not gen_on else "generator")
        ),
    }

    if gen_on and checkpoint_sha is None:
        report["failure"] = {
            "type": "FileNotFoundError",
            "message": f"generator checkpoint missing: {checkpoint_path}",
        }
        report["generator_comparison_valid"] = False
        print(f"  FAIL: checkpoint missing at {checkpoint_path}")
        return report

    if gen_on:
        make_generator: GeneratorFactory
        if generator_factory is None:
            from scripts.train_campaign import build_eval_generator_ref as imported_factory

            make_generator = imported_factory
        else:
            make_generator = generator_factory
        try:
            generator_ref = make_generator(
                generator_arch,
                Path(checkpoint_path) if checkpoint_path is not None else Path("."),
                device=device,
                seed=run_seed,
            )
        except TypeError:
            generator_ref = make_generator(
                generator_arch,
                Path(checkpoint_path) if checkpoint_path is not None else Path("."),
            )
        generator_ref, invocation_counter = wrap_generator_with_counter(generator_ref)

    start_time = time.perf_counter()
    try:
        result = execute(
            cfg,
            [source_frames],
            objects=(objects_for_run,),
            context_ids=[clip.context_id],
            generator=generator_ref,
        )
        wall_seconds = time.perf_counter() - start_time
        delivered = np.asarray(result.delivered_frames)
        scores = score(source_frames, delivered)
        parts = extract_size_parts(result.sizes)
        report.update(
            {
                "coded_bytes": int(parts.get("transport_total") or 0),
                "parts": parts,
                "byte_subledger": extract_byte_subledger(result.sizes),
                "wire_reconciliation": wire_reconciliation(result, result.sizes),
                "scores": scores,
                "timing": actual_timing(result, wall_seconds=wall_seconds),
                "delivered_shape": list(delivered.shape),
                "delivered_frame_hashes": per_frame_sha256(delivered),
                "model_invocation_count": int(invocation_counter[0]),
                "failure": None,
            }
        )
    except Exception as exc:
        wall_seconds = time.perf_counter() - start_time
        report["timing"]["wall_seconds"] = round(wall_seconds, 2)
        report["failure"] = failure_record(exc)
        report["model_invocation_count"] = int(invocation_counter[0])
        print(f"  FAIL {type(exc).__name__}: {exc}")
        return report

    print(f"Results for {corner_name}:")
    residual_bytes = int(report["parts"].get("residual") or 0)
    panorama_bytes = int(report["parts"].get("panorama") or 0)
    appearance_bytes = int(report["parts"].get("actor_reference") or 0)
    print(
        f"  Total Bytes: {int(report['coded_bytes'] or 0):,} B | Residual: {residual_bytes:,} B | "
        f"Background: {panorama_bytes:,} B | Appearance: {appearance_bytes:,} B"
    )
    scores = report["scores"]
    if isinstance(scores, dict) and "psnr_y" in scores:
        print(
            f"  PSNR-Y: {scores['psnr_y']:.2f} dB | SSIM: {scores['ssim']:.4f} | "
            f"VMAF: {scores['vmaf']:.2f}"
        )
    timing_dict = report.get("timing")
    client_sec = timing_dict.get("client_seconds") if isinstance(timing_dict, dict) else None
    wall = timing_dict.get("wall_seconds") if isinstance(timing_dict, dict) else None
    if isinstance(client_sec, (int, float)) and wall is not None:
        print(f"  Wall Time: {wall:.1f}s (Client Dec: {client_sec:.2f}s)")
    elif wall is not None:
        print(f"  Wall Time: {wall:.1f}s")
    return report


def assemble_matrix_report(
    *,
    video: str,
    scene: str,
    frames: int,
    generator: str,
    residual_qp: int,
    clip: Any,
    base_config: PointstreamConfig,
    matrix: list[dict[str, Any]],
    checkpoint_path: Path | None,
    checkpoint_sha256: str | None,
    device: str,
    shuffled_control: bool,
    repo: Path | None = None,
) -> dict[str, Any]:
    source_frames = np.asarray(clip.frames)
    manifest = source_manifest(
        video=video,
        scene=scene,
        frames=source_frames,
        context_id=getattr(clip, "context_id", None),
    )
    revision = git_revision(repo or ps_paths.repo_root())
    clip_objects = getattr(clip, "objects", ())
    resolved_cfg = resolved_configuration(base_config, device=device, objects=clip_objects)
    identity = build_run_identity(
        code_revision=revision,
        checkpoint_sha256=checkpoint_sha256,
        source_frame_hashes=list(manifest["frame_hashes"]),
        config=resolved_cfg,
        video=video,
        scene=scene,
        frames=frames,
        generator=generator,
        residual_qp=residual_qp,
    )
    effect = generation_effect(matrix)
    comparison = assess_generator_comparison(
        checkpoint_sha256=checkpoint_sha256,
        generator_backend=generator,
        matrix=matrix,
    )
    invocation_total = sum(int(row.get("model_invocation_count") or 0) for row in matrix)
    controls = {
        "pasted_reference": [
            row["corner"] for row in matrix if row.get("control") == "pasted_reference"
        ],
        "no_generator": [
            row["corner"] for row in matrix if not row.get("generation_on")
        ],
        "shuffled_conditioning": [
            row["corner"] for row in matrix if row.get("shuffled_conditioning")
        ],
        "shuffled_control_enabled": bool(shuffled_control),
    }
    summary = {
        "doc_role": "diagnostic_matrix_report",
        "video": video,
        "scene": scene,
        "frames": frames,
        "generator_tested": generator,
        "generator_backend": generator,
        "residual_qp": residual_qp,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": int(base_config.run.seed),
        "device": device,
        "inference_parameters": inference_parameters(base_config),
        "model_invocation_count": invocation_total,
        "identity": identity,
        "source_manifest": manifest,
        "resolved_configuration": resolved_cfg,
        "controls": controls,
        "generation_effect": effect,
        "generator_comparison_valid": bool(comparison["generator_comparison_valid"]),
        "generator_comparison": comparison,
        "note": "Fixed residual QP is not matched final fidelity.",
        "timestamp_unix": time.time(),
        "matrix": matrix,
    }
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in summary]
    if missing:
        raise RuntimeError(f"diagnostic report missing keys: {missing}")
    return summary


def _slice_clip(clip_full: LongSceneClip, n_frames: int) -> LongSceneClip:
    if n_frames >= clip_full.n_frames:
        return clip_full
    return LongSceneClip(
        video=clip_full.video,
        scene=clip_full.scene,
        context_id=clip_full.context_id,
        n_frames=n_frames,
        frames=clip_full.frames[:n_frames],
        masks=clip_full.masks[:n_frames],
        objects=tuple(obj for obj in clip_full.objects if obj.frame_index < n_frames),
        paste_back_mae=clip_full.paste_back_mae,
        is_eligible=clip_full.is_eligible,
        route=clip_full.route,
        failure_reasons=clip_full.failure_reasons,
        start_frame=getattr(clip_full, "start_frame", 0),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 2x2 Generation x Residual Diagnostic Matrix")
    parser.add_argument("--frames", type=int, default=16, help="Frame count (default: 16)")
    parser.add_argument("--video", default="alcaraz_highlights", help="Video name")
    parser.add_argument("--scene", default="scene_000", help="Scene name")
    parser.add_argument("--generator", default="pix2pix", help="Generator arch for Gen-ON corners")
    parser.add_argument("--residual-qp", type=int, default=32, help="Residual QP for Res-ON corners")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/diagnostic_matrix_report.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--reuse-results",
        type=Path,
        default=None,
        help="Path to immutable results JSON to reuse matching corners from",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Generator checkpoint file or directory (hashed into identity)",
    )
    parser.add_argument(
        "--shuffled-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run shuffled/foreign-pose generation null (default on)",
    )
    parser.add_argument("--device", default=None, help="Inference device (recorded; default cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    return parser


def resolve_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_matrix(
    clip: Any,
    base_config: PointstreamConfig,
    *,
    generator_arch: str,
    residual_qp: int,
    checkpoint: Path | None,
    shuffled_control: bool,
    device: str,
    frames: int,
    reuse_path: Path | None = None,
    run_fn: RunFn | None = None,
    score_fn: ScoreFn | None = None,
    generator_factory: GeneratorFactory | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    if checkpoint is None:
        try:
            checkpoint = default_checkpoint_for(generator_arch)
        except ValueError:
            checkpoint = None
    checkpoint_sha = sha256_path(checkpoint) if checkpoint is not None else None
    identity = build_run_identity(
        code_revision=git_revision(repo or ps_paths.repo_root()),
        checkpoint_sha256=checkpoint_sha,
        source_frame_hashes=per_frame_sha256(np.asarray(clip.frames)),
        config=resolved_configuration(
            base_config,
            device=device,
            objects=getattr(clip, "objects", ()),
        ),
        video=clip.video,
        scene=clip.scene,
        frames=frames,
        generator=generator_arch,
        residual_qp=residual_qp,
    )

    prior: dict[str, Any] | None = None
    if reuse_path is not None and reuse_path.is_file():
        try:
            prior = json.loads(reuse_path.read_text())
        except Exception as exc:
            print(f"Reuse file unreadable ({type(exc).__name__}: {exc}); running all corners")
            prior = None
    reusable = reusable_corners(prior, identity)

    corners: list[tuple[str, bool, bool, bool]] = [
        ("gen_off_res_off", False, False, False),
        ("gen_off_res_on", False, True, False),
        ("gen_on_res_off", True, False, False),
        ("gen_on_res_on", True, True, False),
    ]
    if shuffled_control:
        corners.append(("gen_on_shuffled_conditioning", True, False, True))

    results: list[dict[str, Any]] = []
    for name, gen_on, res_on, shuffled in corners:
        if name in reusable:
            print(f"Reusing identity-matched result for {name}")
            results.append(reusable[name])
            continue
        results.append(
            run_diagnostic_corner(
                name,
                clip,
                base_config,
                gen_on=gen_on,
                res_on=res_on,
                generator_arch=generator_arch,
                generator_checkpoint=checkpoint,
                residual_qp=residual_qp,
                shuffle_conditioning=shuffled,
                seed=base_config.run.seed,
                device=device,
                run_fn=run_fn,
                score_fn=score_fn,
                generator_factory=generator_factory,
            )
        )

    return assemble_matrix_report(
        video=clip.video,
        scene=clip.scene,
        frames=frames,
        generator=generator_arch,
        residual_qp=residual_qp,
        clip=clip,
        base_config=base_config,
        matrix=results,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        device=device,
        shuffled_control=shuffled_control,
        repo=repo,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    load_frames = max(48, args.frames)
    clip_full = load_long_scene_clip(args.video, args.scene, n_frames=load_frames)
    clip = _slice_clip(clip_full, args.frames) if args.frames < load_frames else clip_full
    base = load_tier("balanced")
    if args.seed is not None:
        base = replace(base, run=replace(base.run, seed=args.seed))
    device = resolve_device(args.device)
    summary = run_matrix(
        clip,
        base,
        generator_arch=args.generator,
        residual_qp=args.residual_qp,
        checkpoint=args.checkpoint,
        shuffled_control=bool(args.shuffled_control),
        device=device,
        frames=args.frames,
        reuse_path=args.reuse_results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nDiagnostic Matrix complete. Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
