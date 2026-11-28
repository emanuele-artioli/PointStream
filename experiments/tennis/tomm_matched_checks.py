"""GPU verification for matched latency (Shanghai) and rendered Q90 / 5-frame pose bridge (Sydney)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Any

import numpy as np
from PIL import Image

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b
from experiments.tennis import tomm_v5_pareto_pilot as v5

SHANGHAI_CARRIER_SHA256 = "3ebc2e455509db026919f17c17d2bfce3eba899fd29a0888c47440b576a219af"
SYDNEY_CARRIER_SHA256 = "a70a444b65f905285830a5fc2561e752708ac24ea0a5e02d538d85ac742dd99c"
FIT_SHA256 = "85e3c65c9777b799f9ed0df3a5c8da567fe8a92e4ac83d115719527acb739eb4"


@dataclass(frozen=True)
class TimingConfig:
    name: str
    scheduler: str
    steps: int
    width: int
    height: int
    overlap: int
    pose_smooth: bool


SHANGHAI_CONFIGS: tuple[TimingConfig, ...] = (
    TimingConfig("ddim30_512_ov4", "ddim", 30, 512, 784, 4, False),
    TimingConfig("dpm10_512_ov8_smooth", "dpm", 10, 512, 784, 8, True),
    TimingConfig("dpm6_512_ov8_smooth", "dpm", 6, 512, 784, 8, True),
    TimingConfig("dpm10_384_ov8_smooth", "dpm", 10, 384, 576, 8, True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 2**20):
            digest.update(block)
    return digest.hexdigest()


def apply_linear_gap(kp: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
    """Linearly interpolate interior gap [gap_start, gap_end) from bracketed endpoints."""
    out = kp.copy()
    left_idx = gap_start - 1
    right_idx = gap_end
    span = gap_end - gap_start + 1
    left = kp[left_idx]
    right = kp[right_idx]
    both_ok = (left[:, 2] > 0.0) & (right[:, 2] > 0.0)
    for k, tick in enumerate(range(gap_start, gap_end), start=1):
        alpha = k / span
        out[tick, :, :2] = np.where(
            both_ok[:, None],
            (1.0 - alpha) * left[:, :2] + alpha * right[:, :2],
            left[:, :2],
        )
        out[tick, :, 2] = np.where(both_ok, np.minimum(left[:, 2], right[:, 2]), 0.0)
    return out


def apply_hold_gap(kp: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
    """Hold the last observed pose at gap_start - 1 across [gap_start, gap_end)."""
    out = kp.copy()
    for tick in range(gap_start, gap_end):
        out[tick] = kp[gap_start - 1]
    return out


def run_pilot(args: argparse.Namespace) -> None:
    aa = Path(sys.prefix).resolve() / "opt/Moore-AnimateAnyone"
    if str(aa) not in sys.path:
        sys.path.insert(1, str(aa))
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 8, "full": 48}[stage_name]
    if args.frames != expected:
        raise ValueError(f"frames={args.frames} mismatches stage {stage_name} ({expected})")

    if sha256(args.shanghai_carrier) != SHANGHAI_CARRIER_SHA256:
        raise ValueError("shanghai carrier sha256 mismatch")
    if sha256(args.sydney_carrier) != SYDNEY_CARRIER_SHA256:
        raise ValueError("sydney carrier sha256 mismatch")

    fit_root = scratch / "base-fit"
    fit_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.base) as tf:
        tf.extractall(fit_root, filter="data")
    fit_path = fit_root / "publish/tennis-base@fit-100.pt"
    if sha256(fit_path) != FIT_SHA256:
        raise ValueError("fit sha256 mismatch")

    q90_root = scratch / "sydney-q90"
    q90_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.sydney_q90_tar) as tf:
        tf.extractall(q90_root, filter="data")
    q90_pub = q90_root / "publish"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    limit_mib = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "16000"))
    total_mib = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit_mib * 2**20 / total_mib))

    weights = dict(zip(
        (
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "image_encoder/config.json",
            "image_encoder/pytorch_model.bin",
            "original/denoising_unet.pth",
            "original/reference_unet.pth",
            "original/pose_guider.pth",
            "original/motion_module.pth",
        ),
        map(Path, args.weight),
        strict=True,
    ))

    load_t0 = time.perf_counter()
    pipe, fit_info = v5.build_pipeline(weights, fit_path, scratch, "cuda")
    load_s = time.perf_counter() - load_t0

    pub = scratch / "publish"
    pub.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []

    # 1) Shanghai matched timing & speedup check (8f smoke / 48f full)
    _, _, sh_decoded = packet.unpack(args.shanghai_carrier.read_bytes())
    sh_units = ["match102_near"] if stage_name == "smoke" else ["match102_near", "match102_far"]
    for unit in sh_units:
        run_data = sh_decoded[unit][0]
        raw_kp = np.asarray(run_data["kp"][: args.frames], dtype=np.float32)
        smooth_kp = v5.smooth_keypoints(raw_kp)
        boxes = np.asarray(run_data["boxes"][: args.frames], dtype=np.float64)
        ref_crop = np.asarray(run_data["reference"], dtype=np.uint8)
        seed = int(run_data["seed"])
        ticks = np.asarray(run_data["t"][: args.frames], dtype=np.int32)
        raw_poses = np.stack([k6b.pose_map(k6b.to_crop(raw_kp[t], boxes[t])) for t in range(args.frames)])
        smooth_poses = np.stack([k6b.pose_map(k6b.to_crop(smooth_kp[t], boxes[t])) for t in range(args.frames)])

        for cfg in SHANGHAI_CONFIGS:
            vcfg = v5.PilotConfig(cfg.name, cfg.scheduler, cfg.steps, cfg.width, cfg.height, cfg.overlap, cfg.pose_smooth, 0.0)
            poses_in = smooth_poses if cfg.pose_smooth else raw_poses
            crops, render_s = v5.render_config(pipe, vcfg, ref_crop, poses_in, seed)
            distinct = len({hashlib.sha256(c.tobytes()).hexdigest() for c in crops})
            out_dir = pub / "shanghai" / cfg.name
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / f"{unit}.npz"
            np.savez_compressed(npz_path, t=ticks, box=boxes, crops=crops)
            run_records.append({
                "suite": "shanghai_timing",
                "config": asdict(cfg),
                "unit": unit,
                "frames": args.frames,
                "seed": seed,
                "render_seconds": render_s,
                "sec_per_frame": render_s / args.frames,
                "distinct_frames": distinct,
                "npz_relpath": f"shanghai/{cfg.name}/{unit}.npz",
                "npz_sha256": sha256(npz_path),
            })

    # 2) Sydney rendered AVIF Q90 + rendered 5-frame pose gap bridge check (8f smoke / 24f full)
    _, _, sy_decoded = packet.unpack(args.sydney_carrier.read_bytes())
    sy_frames = 8 if stage_name == "smoke" else 24
    gap_start, gap_end = (2, 7) if stage_name == "smoke" else (9, 14)
    sy_units = ["match127_near"] if stage_name == "smoke" else ["match127_near", "match127_far"]
    sy_cfg = v5.PilotConfig("dpm10_512_ov8_smooth", "dpm", 10, 512, 784, 8, True, 0.0)

    for unit in sy_units:
        run_data = sy_decoded[unit][0]
        raw_kp = np.asarray(run_data["kp"][:sy_frames], dtype=np.float32)
        gap_bridge_kp = apply_linear_gap(raw_kp, gap_start, gap_end)
        gap_hold_kp = apply_hold_gap(raw_kp, gap_start, gap_end)
        boxes = np.asarray(run_data["boxes"][:sy_frames], dtype=np.float64)
        lossless_ref = np.asarray(run_data["reference"], dtype=np.uint8)
        q90_ref = np.asarray(Image.open(q90_pub / f"{unit}_q90.png").convert("RGB"))
        seed = int(run_data["seed"])
        ticks = np.asarray(run_data["t"][:sy_frames], dtype=np.int32)

        def to_smooth_poses(kp_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
            sm = v5.smooth_keypoints(kp_arr)
            return np.stack([k6b.pose_map(k6b.to_crop(sm[t], b_arr[t])) for t in range(len(sm))])

        arms_sy = [
            ("ref_lossless_uninterrupted", lossless_ref, to_smooth_poses(raw_kp, boxes)),
            ("ref_q90_uninterrupted", q90_ref, to_smooth_poses(raw_kp, boxes)),
            ("ref_lossless_gap5_bridge", lossless_ref, to_smooth_poses(gap_bridge_kp, boxes)),
            ("ref_lossless_gap5_hold", lossless_ref, to_smooth_poses(gap_hold_kp, boxes)),
        ]
        if stage_name == "smoke":
            arms_sy = arms_sy[:2]
        for arm_name, ref_in, poses_in in arms_sy:
            crops, render_s = v5.render_config(pipe, sy_cfg, ref_in, poses_in, seed)
            distinct = len({hashlib.sha256(c.tobytes()).hexdigest() for c in crops})
            out_dir = pub / "sydney" / arm_name
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / f"{unit}.npz"
            np.savez_compressed(npz_path, t=ticks, box=boxes, crops=crops)
            run_records.append({
                "suite": "sydney_ablation",
                "arm": arm_name,
                "unit": unit,
                "frames": sy_frames,
                "gap_range": [gap_start, gap_end],
                "seed": seed,
                "render_seconds": render_s,
                "sec_per_frame": render_s / sy_frames,
                "distinct_frames": distinct,
                "npz_relpath": f"sydney/{arm_name}/{unit}.npz",
                "npz_sha256": sha256(npz_path),
            })

    dispatch = json.loads((stage / "dispatch.json").read_text()) if (stage / "dispatch.json").exists() else {}
    summary = {
        "stage": stage_name,
        "shanghai_frames": args.frames,
        "sydney_frames": sy_frames,
        "sydney_gap_range": [gap_start, gap_end],
        "device": torch.cuda.get_device_name(0),
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "timing_contaminated": dispatch.get("gpu_sharing", {}).get("mode") == "shared",
        "model_load_seconds": load_s,
        "fit_info": fit_info,
        "records": run_records,
    }
    (pub / "matched_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (stage / "matched_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    archive = stage / "published.tar"
    tmp = stage / ".matched.tar"
    with tarfile.open(tmp, "x") as tf:
        tf.add(pub, arcname="publish")
    tmp.rename(archive)


def validate(_args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 8, "full": 48}[stage_name]
    data = json.loads((stage / "matched_summary.json").read_text())
    assert data["shanghai_frames"] == expected
    records = data["records"]
    assert records
    for r in records:
        assert r["distinct_frames"] >= r["frames"] - 1
        assert r["render_seconds"] > 0
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "shanghai_frames": expected,
                    "sydney_frames": data["sydney_frames"],
                    "records": len(records),
                    "cuda_device": data["device"],
                    "gpu_peak_mib": data["gpu_peak_mib"],
                    "archive_sha256": sha256(stage / "published.tar"),
                },
                "timing_contaminated": data["timing_contaminated"],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shanghai-carrier", type=Path, required=True)
    p.add_argument("--sydney-carrier", type=Path, required=True)
    p.add_argument("--sydney-q90-tar", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--weight", type=Path, action="append", required=True)
    p.add_argument("--frames", type=int, choices=(8, 48), required=True)
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        stage = Path(os.environ["PS_STAGE_DIR"])
        argv = json.loads((stage / "dispatch.json").read_text())["command"]
        validate(p.parse_args(["--validate", *argv[3:]]))
    else:
        args = p.parse_args()
        if args.validate:
            validate(args)
        else:
            run_pilot(args)


if __name__ == "__main__":
    main()
