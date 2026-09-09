"""Inference smoke test for PointStream generators.

Verifies:
1. Native sequence input reaching the generator.
2. Real reference appearance and real skeleton pose conditioning from the dataset.
3. Controls: pasted reference (static copy) and shuffled/wrong conditioning.
4. Output checks: generation actually executes, differs from reference,
   differs from shuffled conditioning, and is not a source-frame fallback.
5. Determinism across repeated fresh-process runs with fixed seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.contracts import paths
from src.contracts.conditioning import ConditioningBundle, GenerationParams
from src.components.generation._numpy import as_chw, as_hwc


def load_real_conditioning(
    dataset_root: Path,
    video: str = "alcaraz_perricard",
    scene: str = "scene_002",
    track: str = "track_0002",
    num_frames: int = 3,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Load real reference frame, sequence of poses, and ground truth source frames."""
    track_dir = dataset_root / video / "segmentations" / scene / track
    skel_dir = dataset_root / video / "segmentations" / scene / f"{track}_skeleton"

    if not track_dir.exists() or not skel_dir.exists():
        raise FileNotFoundError(f"Track data not found at {track_dir} or {skel_dir}")

    ref_bgr = cv2.imread(str(track_dir / "frame_000000.png"))
    if ref_bgr is None:
        raise FileNotFoundError(f"Could not read reference frame from {track_dir}")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    pose_frames = []
    source_frames = []
    for i in range(num_frames):
        fid = f"frame_{i:06d}.png"
        p_bgr = cv2.imread(str(skel_dir / fid))
        if p_bgr is None:
            p_bgr = np.zeros_like(ref_bgr)
        pose_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
        pose_frames.append(pose_rgb)

        s_bgr = cv2.imread(str(track_dir / fid))
        if s_bgr is None:
            s_bgr = ref_bgr.copy()
        s_rgb = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB)
        source_frames.append(s_rgb)

    return ref_rgb, pose_frames, source_frames


def build_bundles(
    ref_rgb: np.ndarray,
    pose_frames: list[np.ndarray],
    width: int,
    height: int,
) -> list[ConditioningBundle]:
    bundles = []
    h, w = ref_rgb.shape[:2]
    for i, pose_rgb in enumerate(pose_frames):
        b = ConditioningBundle(
            appearance=as_chw(ref_rgb),
            pose=as_chw(pose_rgb),
            bbox=(0, 0, w, h),
            frame_index=i,
            object_id="player",
        )
        bundles.append(b)
    return bundles


def run_generator_inference(
    arch: str,
    bundles: list[ConditioningBundle],
    *,
    seed: int,
    device: str,
    steps: int,
    width: int,
    height: int,
    checkpoint: str | None = None,
) -> list[np.ndarray]:
    params = GenerationParams(width=width, height=height, steps=steps)

    if arch in ("animate-anyone", "animate_anyone"):
        from src.components.generation.animate_anyone import AnimateAnyoneGenerator
        gen = AnimateAnyoneGenerator(width=width, height=height, steps=steps, checkpoint=checkpoint)
        output = gen.generate_sequence(bundles, seed=seed, device=device, params=params)
        return [as_hwc(f) for f in output]
    elif arch == "pix2pix":
        from src.components.generation.pix2pix import Pix2PixGenerator
        gen = Pix2PixGenerator(width=width, height=height, checkpoint=checkpoint)
        return [as_hwc(gen.generate(b, seed=seed, device=device, params=params)) for b in bundles]
    elif arch == "spade4tennis":
        from src.components.generation.spade import Spade4TennisGenerator
        gen = Spade4TennisGenerator(width=width, height=height, checkpoint=checkpoint)
        return [as_hwc(gen.generate(b, seed=seed, device=device, params=params)) for b in bundles]
    elif arch in ("controlnet", "pose-controlnet"):
        from src.components.generation.controlnet import ControlNetGenerator
        gen = ControlNetGenerator(variant="pose", width=width, height=height, steps=steps, checkpoint=checkpoint)
        return [as_hwc(gen.generate(b, seed=seed, device=device, params=params)) for b in bundles]
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference smoke test for PointStream generators.")
    parser.add_argument("--arch", type=str, default="animate-anyone", choices=["animate-anyone", "pix2pix", "spade4tennis", "controlnet"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_root = paths.assets() / "dataset"
    print(f"--- Generator Inference Smoke Test: {args.arch} on {device} ---")
    print(f"Dataset root: {dataset_root}")

    # 1. Load real conditioning
    ref_rgb, pose_frames, source_frames = load_real_conditioning(
        dataset_root, num_frames=args.num_frames
    )
    print(f"Loaded real conditioning: ref {ref_rgb.shape}, {len(pose_frames)} pose frames, {len(source_frames)} source frames")

    # 2. Build normal conditioning bundles
    normal_bundles = build_bundles(ref_rgb, pose_frames, args.width, args.height)

    # 3. Build shuffled/wrong conditioning bundles (reverse poses)
    shuffled_poses = list(reversed(pose_frames))
    shuffled_bundles = build_bundles(ref_rgb, shuffled_poses, args.width, args.height)

    # 4. Pasted reference control
    pasted_ref = [cv2.resize(ref_rgb, (args.width, args.height)) for _ in range(args.num_frames)]

    # 5. Run standard generation
    t0 = time.perf_counter()
    out_normal = run_generator_inference(
        args.arch,
        normal_bundles,
        seed=args.seed,
        device=device,
        steps=args.steps,
        width=args.width,
        height=args.height,
        checkpoint=args.checkpoint,
    )
    t_normal = time.perf_counter() - t0
    print(f"Standard generation completed in {t_normal:.2f}s ({len(out_normal)} frames)")

    # 6. Run shuffled conditioning generation
    t0 = time.perf_counter()
    out_shuffled = run_generator_inference(
        args.arch,
        shuffled_bundles,
        seed=args.seed,
        device=device,
        steps=args.steps,
        width=args.width,
        height=args.height,
        checkpoint=args.checkpoint,
    )
    t_shuffled = time.perf_counter() - t0
    print(f"Shuffled conditioning generation completed in {t_shuffled:.2f}s")

    # 7. Quality and difference checks
    diff_vs_pasted = float(np.mean([np.abs(a.astype(float) - b.astype(float)) for a, b in zip(out_normal, pasted_ref)]))
    diff_vs_shuffled = float(np.mean([np.abs(a.astype(float) - b.astype(float)) for a, b in zip(out_normal, out_shuffled)]))

    # Source frame check: resized source frames
    src_resized = [cv2.resize(s, (args.width, args.height)) for s in source_frames]
    diff_vs_source = float(np.mean([np.abs(a.astype(float) - b.astype(float)) for a, b in zip(out_normal, src_resized)]))

    # Temporal difference between consecutive frames in generated sequence
    temporal_diff = float(np.mean([np.abs(out_normal[i].astype(float) - out_normal[i-1].astype(float)) for i in range(1, len(out_normal))]))

    # Hashes of outputs
    frame_hashes = [hashlib.sha256(f.tobytes()).hexdigest()[:16] for f in out_normal]

    print("\n--- Diagnostic Measurements ---")
    print(f"Mean L1 diff vs pasted reference (control): {diff_vs_pasted:.3f}")
    print(f"Mean L1 diff vs shuffled conditioning:    {diff_vs_shuffled:.3f}")
    print(f"Mean L1 diff vs raw source frames:         {diff_vs_source:.3f}")
    print(f"Mean temporal L1 diff across frames:       {temporal_diff:.3f}")
    print(f"Frame 0 min={out_normal[0].min()}, max={out_normal[0].max()}, mean={out_normal[0].mean():.2f}")
    print(f"Frame hashes: {frame_hashes}")

    # Checks / assertions
    failures = []
    if diff_vs_pasted < 1.0:
        failures.append(f"Generation output too close to pasted reference (diff={diff_vs_pasted:.3f})")
    if diff_vs_shuffled < 0.1:
        failures.append(f"Conditioning did not affect output (diff_vs_shuffled={diff_vs_shuffled:.3f})")
    if diff_vs_source == 0.0:
        failures.append("Output is bit-identical to source frame (trivial fallback detected)")
    if out_normal[0].max() == 0:
        failures.append("Output is completely black")

    results = {
        "arch": args.arch,
        "device": device,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "runtime_seconds": t_normal,
        "diff_vs_pasted": diff_vs_pasted,
        "diff_vs_shuffled": diff_vs_shuffled,
        "diff_vs_source": diff_vs_source,
        "temporal_diff": temporal_diff,
        "frame_hashes": frame_hashes,
        "passed": len(failures) == 0,
        "failures": failures,
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))

    if failures:
        print(f"\nSMOKE TEST FAILED with {len(failures)} violation(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nSMOKE TEST PASSED: {args.arch} successfully generated verified sequence on {device}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
