"""Diagnostic Matrix: Generation (OFF/ON) x Residual (OFF/ON) with Pasted-Reference Control.

Evaluates the 4 corners of the ablation lattice:
1. [Gen OFF, Res OFF]: Pasted-reference control without residual correction.
2. [Gen OFF, Res ON]:  Pasted-reference control with residual correction.
3. [Gen ON, Res OFF]:  Generative synthesis without residual correction.
4. [Gen ON, Res ON]:   Generative synthesis with residual correction.

Outputs total bytes, component breakdown, PSNR-Y, SSIM, VMAF, and timings.
"""
from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.long_scenes.loader import LongSceneClip, load_long_scene_clip
from experiments.tier.low_rate_measure import score_headlines
from scripts.train_campaign import build_eval_generator_ref
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
from src.runner.run import run


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
) -> dict[str, Any]:
    print("\n=======================================================")
    print(f"Running Diagnostic Corner: {corner_name}")
    print(f"Generation: {'ON (' + generator_arch + ')' if gen_on else 'OFF (Pasted-Reference)'} | Residual: {'ON (QP ' + str(residual_qp) + ')' if res_on else 'OFF'}")
    print("=======================================================")

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

    lattice = StageLattice.of(*stages)

    from src.runner.routing import lattice_config_from
    cfg = replace(base_config, lattice=lattice_config_from(lattice))

    generator_ref = None
    if gen_on:
        cfg = replace(cfg, generator=replace(cfg.generator, backend=generator_arch))
        if generator_checkpoint is None:
            weights_dir = ps_paths.assets() / "weights"
            if generator_arch == "pix2pix":
                generator_checkpoint = weights_dir / "pix2pix_generator.pt"
            elif generator_arch == "spade4tennis":
                generator_checkpoint = weights_dir / "spade4tennis_lite_generator.pt"
            elif generator_arch == "animate-anyone":
                generator_checkpoint = Path("/home/itec/emanuele/Models/AnimateAnyone/profiles/finetuned_tennis")
            else:
                raise ValueError(f"Unknown generator arch: {generator_arch}")

        generator_ref = build_eval_generator_ref(generator_arch, generator_checkpoint)

    if res_on:
        res_cfg = replace(
            cfg.residual,
            codec="avc",
            rate=residual_qp,
            background_downscale=1,
            block_threshold=0.0,
        )
        cfg = replace(cfg, residual=res_cfg)

    import cv2
    from src.contracts.conditioning import ConditioningBundle
    objects_for_run = clip.objects
    if gen_on:
        augmented = []
        dataset_scene_dir = ps_paths.assets() / "dataset" / clip.video / "segmentations" / clip.scene
        start_frame = 38
        for obj in clip.objects:
            obj_id = obj.object_id
            abs_frame = start_frame + obj.frame_index
            skel_dir = dataset_scene_dir / f"{obj_id}_skeleton"
            skel_img = None
            if skel_dir.is_dir():
                p = skel_dir / f"frame_{abs_frame:06d}.png"
                if p.is_file():
                    skel_bgr = cv2.imread(str(p))
                    if skel_bgr is not None:
                        skel_img = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2RGB)
            h, w = obj.appearance.shape[:2]
            if skel_img is None:
                skel_img = np.zeros((h, w, 3), dtype=np.uint8)
            elif skel_img.shape[:2] != (h, w):
                skel_img = cv2.resize(skel_img, (w, h))

            bundle = ConditioningBundle(
                appearance=np.transpose(obj.appearance, (2, 0, 1)),
                pose=np.transpose(skel_img, (2, 0, 1)),
                mask=obj.mask,
                bbox=obj.bbox,
                frame_index=obj.frame_index,
                object_id=obj.object_id,
            )
            augmented.append(replace(obj, conditioning=bundle))
        objects_for_run = tuple(augmented)

    # Execute runner
    source_frames = np.asarray(clip.frames)
    start_time = time.perf_counter()

    result = run(
        cfg,
        [source_frames],
        objects=(objects_for_run,),
        context_ids=[clip.context_id],
        generator=generator_ref,
    )
    wall_seconds = time.perf_counter() - start_time

    delivered = result.delivered_frames
    scores = score_headlines(source_frames, delivered)

    # Component breakdown
    parts = {
        "residual": int(result.sizes.residual),
        "panorama": int(result.sizes.panorama),
        "actor_reference": int(result.sizes.actor_reference),
        "metadata": int(result.sizes.metadata),
    }

    report = {
        "corner": corner_name,
        "generation_on": gen_on,
        "residual_on": res_on,
        "generator_arch": generator_arch if gen_on else "none (pasted_reference_control)",
        "residual_qp": residual_qp if res_on else None,
        "coded_bytes": int(result.sizes.transport_total),
        "parts": parts,
        "scores": scores,
        "timing": {
            "wall_seconds": round(wall_seconds, 2),
            "encoder_seconds": result.timing.get("encoder_seconds"),
            "client_seconds": result.timing.get("client_seconds"),
            "evaluation_seconds": result.timing.get("evaluation_seconds"),
        },
        "delivered_shape": list(delivered.shape),
    }

    timing_dict = report.get("timing")
    client_sec = timing_dict.get("client_seconds") if isinstance(timing_dict, dict) else None

    print(f"Results for {corner_name}:")
    print(f"  Total Bytes: {report['coded_bytes']:,} B | Residual: {parts['residual']:,} B | Background: {parts['panorama']:,} B | Appearance: {parts['actor_reference']:,} B")
    print(f"  PSNR-Y: {scores['psnr_y']:.2f} dB | SSIM: {scores['ssim']:.4f} | VMAF: {scores['vmaf']:.2f}")
    if isinstance(client_sec, (int, float)):
        print(f"  Wall Time: {wall_seconds:.1f}s (Client Dec: {client_sec:.2f}s)")
    else:
        print(f"  Wall Time: {wall_seconds:.1f}s")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2x2 Generation x Residual Diagnostic Matrix")
    parser.add_argument("--frames", type=int, default=16, help="Frame count (default: 16)")
    parser.add_argument("--video", default="alcaraz_highlights", help="Video name")
    parser.add_argument("--scene", default="scene_000", help="Scene name")
    parser.add_argument("--generator", default="pix2pix", help="Generator arch for Gen-ON corners")
    parser.add_argument("--residual-qp", type=int, default=32, help="Residual QP for Res-ON corners")
    parser.add_argument("--output", type=Path, default=Path("/tmp/diagnostic_matrix_report.json"), help="Output JSON path")
    parser.add_argument("--reuse-results", type=Path, default=None, help="Path to immutable results JSON to reuse matching corners from")
    args = parser.parse_args()

    # Load nearest valid interval (48 frames minimum in manifest)
    load_frames = max(48, args.frames)
    clip_full = load_long_scene_clip(args.video, args.scene, n_frames=load_frames)

    if args.frames < load_frames:
        n = args.frames
        clip = LongSceneClip(
            video=clip_full.video,
            scene=clip_full.scene,
            context_id=clip_full.context_id,
            n_frames=n,
            frames=clip_full.frames[:n],
            masks=clip_full.masks[:n],
            objects=tuple(obj for obj in clip_full.objects if obj.frame_index < n),
            paste_back_mae=clip_full.paste_back_mae,
            is_eligible=clip_full.is_eligible,
            route=clip_full.route,
            failure_reasons=clip_full.failure_reasons,
        )
    else:
        clip = clip_full

    base = load_tier("balanced")

    corners = [
        ("gen_off_res_off", False, False),
        ("gen_off_res_on", False, True),
        ("gen_on_res_off", True, False),
        ("gen_on_res_on", True, True),
    ]

    reusable_by_corner: dict[str, dict[str, Any]] = {}
    if args.reuse_results and args.reuse_results.is_file():
        try:
            prior_data = json.loads(args.reuse_results.read_text())
            # Reuse only if complete configuration identity matches
            if (
                prior_data.get("video") == args.video
                and prior_data.get("scene") == args.scene
                and prior_data.get("frames") == args.frames
                and prior_data.get("generator_tested") == args.generator
                and prior_data.get("residual_qp") == args.residual_qp
            ):
                for corner_item in prior_data.get("matrix", []):
                    c_name = corner_item.get("corner")
                    if c_name:
                        reusable_by_corner[c_name] = corner_item
        except Exception:
            pass

    results: list[dict[str, Any]] = []
    for name, g_on, r_on in corners:
        if name in reusable_by_corner:
            print(f"Reusing verified configuration-matched result for {name}")
            results.append(reusable_by_corner[name])
            continue
        rep = run_diagnostic_corner(
            name,
            clip,
            base,
            gen_on=g_on,
            res_on=r_on,
            generator_arch=args.generator,
            residual_qp=args.residual_qp,
        )
        results.append(rep)

    summary = {
        "doc_role": "diagnostic_matrix_report",
        "video": args.video,
        "scene": args.scene,
        "frames": args.frames,
        "generator_tested": args.generator,
        "residual_qp": args.residual_qp,
        "note": "Fixed residual QP is not matched final fidelity.",
        "timestamp_unix": time.time(),
        "matrix": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nDiagnostic Matrix complete. Saved to {args.output}")


if __name__ == "__main__":
    main()
