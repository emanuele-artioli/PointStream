"""Fleet adapter for one paid player carrier, one continuous generation per player."""

from __future__ import annotations

import argparse
import json
import math
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import player_requalification as replay

MODULE = "experiments.tennis.common_receiver_job"
WEIGHT_PATHS = (
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "image_encoder/config.json",
    "image_encoder/pytorch_model.bin",
    "original/denoising_unet.pth",
    "original/reference_unet.pth",
    "original/pose_guider.pth",
    "original/motion_module.pth",
)


def generation_forecast(
    timing: dict, render_seconds: float, available: float, full_cap: int, margin: float = 1.25, target_frames: int = 240
) -> dict:
    """Scale the actual 44-tick render, keeping measured initialization and isolation fixed."""
    wall = timing["worker_wall_seconds"]
    if (
        not all(math.isfinite(v) and v > 0 for v in (wall, render_seconds, available, full_cap))
        or render_seconds > wall
    ):
        raise ValueError("invalid generation smoke clock")
    if not math.isfinite(margin) or margin < 1.25 or target_frames not in (120, 240):
        raise ValueError("invalid forecast margin")
    estimate = margin * (wall - render_seconds + render_seconds * target_frames / 44) + 90
    if estimate > min(full_cap, available - 60):
        raise ValueError(f"forecast {estimate:.1f}s exceeds full cap or remaining lifecycle")
    return {
        "forecast_seconds": estimate,
        "remaining_seconds": available,
        "full_cap_seconds": full_cap,
        "basis": f"{margin}*(44-tick wall minus render-and-save + {target_frames}/44 times render-and-save) + 90; reserve 60s teardown",
    }


def actual_full_cost(timing: dict, cap: int) -> dict:
    wall = timing["worker_wall_seconds"]
    if not math.isfinite(wall) or not 0 < wall <= cap:
        raise ValueError("measured full worker exceeded its cap")
    return {"actual_full_worker_seconds": wall, "full_cap_seconds": cap}


def check_generated(args: argparse.Namespace, output: Path) -> tuple[dict, list[Path]]:
    if replay.sha256(args.carrier) != args.carrier_sha256:
        raise ValueError("carrier identity")
    binding = manifest.verify(args.manifest, args.carrier_sha256)
    gen_path = output / "generation.json"
    receipt = json.loads(gen_path.read_text())
    isolation = receipt["isolation"]
    if (
        isolation.get("landlock_abi", 0) < 3
        or isolation.get("interfaces") != ["lo"]
        or not isolation.get("denials")
        or set(isolation["denials"].values()) - {1, 13}
        or receipt.get("conditioning_only") is not False
        or receipt.get("carrier_sha256") != args.carrier_sha256
        or receipt.get("frames") != args.frames
        or receipt.get("weight_hashes") != replay.PINNED_WEIGHTS
        or receipt.get("common_fit_sha256") != replay.BASE_SHA
        or receipt.get("gpu_uuid") != os.environ["CUDA_VISIBLE_DEVICES"]
        or not 0 < receipt["gpu_peak_mib"] <= receipt["gpu_memory_limit_mib"]
    ):
        raise ValueError("incomplete receiver, CUDA or model identity")
    if not all(str(path.resolve(strict=True)) in isolation["denials"] for path in args.deny):
        raise ValueError("missing denied source probe")
    (row,) = receipt["outputs"]
    name = args.unit + ".npz"
    crops_file = output / name
    if (
        row["unit"] != args.unit
        or row["crops_file"] != name
        or row["crops_file_sha256"] != replay.sha256(crops_file)
        or row["kernels"].get("kernel_launches", 0) <= 0
        or not 0 < row["render_and_save_seconds"] < 2700
    ):
        raise ValueError("crop identity or CUDA kernel evidence")
    if "RTX A6000" in receipt["device"]:
        kernels = row["kernels"]
        if (
            "flash" not in kernels.get("attention_family", [])
            or not set(kernels.get("tensor_core_like", [])).intersection({"cutlass", "xmma", "s16816", "hmma"})
            or (args.frames == 44 and len(row.get("repeat_8_sha256", "")) != 64)
        ):
            raise ValueError("A6000 qualification or repeatability proof missing")
    _, roles, units = packet.unpack(args.carrier.read_bytes())
    (unit,) = units[args.unit]
    if (
        sum(roles.values()) != args.carrier.stat().st_size
        or binding["manifest_bytes"] != args.manifest.stat().st_size
    ):
        raise ValueError("unaccounted receiver input")
    with np.load(crops_file, allow_pickle=False) as data:
        crops = data["crops"]
        if (
            set(data.files) != {"t", "box", "crops"}
            or row["seed"] != unit["seed"]
            or not np.array_equal(data["t"], unit["t"][: args.frames])
            or not np.array_equal(data["box"], np.asarray(unit["boxes"][: args.frames]))
            or not replay.complete_crops(crops, args.frames)
            or len({replay.array_sha(frame) for frame in crops}) != row["distinct_frames"]
        ):
            raise ValueError("output coverage, packet geometry or crop content")
    return receipt, [gen_path, crops_file]


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.frames != {"smoke": 44, "full": args.run_length}.get(os.environ["PS_STAGE"])
        or len(args.weight) != 8
        or len(args.deny) < 4
        or len(set(args.deny)) != len(args.deny)
        or not 1 <= args.full_cap <= 2700
    ):
        raise ValueError("invalid receiver stage contract")
    if replay.sha256(args.carrier) != args.carrier_sha256:
        raise ValueError("carrier identity")
    profile = packet._embedded_profile(args.carrier.read_bytes())
    packet._profile(profile)
    runs = [r for r in profile["runs"] if r["unit"] == args.unit]
    if len(runs) != 1 or runs[0]["end"] - runs[0]["start"] != args.run_length:
        raise ValueError("requested generation length differs from the paid player run")


def bind_allocation(stage: Path) -> None:
    """Carry the exclusive fleet claim's memory contract into the receiver sandbox."""
    dispatch = json.loads((stage / "dispatch.json").read_text())
    if dispatch["gpu_sharing"]["mode"] == "shared":
        return  # The fleet already binds the memory ceiling for shared claims.
    spec = json.loads((stage.parent / "spec.json").read_text())
    gpu = dispatch["environment"]["gpu"]
    uuid = gpu["uuid"]
    if (
        dispatch["gpu_sharing"] != {"mode": "exclusive"}
        or spec["gpu_sharing"] != {"mode": "exclusive"}
        or dispatch["contention_policy"] != {"policy": "stop", "resume_attempts": 0}
        or spec["gpu_memory_mib"] != 18000
        or gpu["name"] not in {"NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A6000"}
        or not uuid.startswith("GPU-")
        or uuid != os.environ.get("PS_CLAIMED_GPU_UUID")
        or uuid != os.environ.get("CUDA_VISIBLE_DEVICES")
        or gpu["pointstream_claimed"]
        or gpu["pointstream_shared_claims"] != []
        or gpu["compute_processes"] != []
        or not 0 <= gpu["memory_used_mib"] <= 256
        or not 0 <= gpu["utilization_pct"] <= 5
        or gpu["memory_free_mib"] < 18000 + 4096
    ):
        raise ValueError("exclusive idle GPU admission or memory contract mismatch")
    os.environ["PS_GPU_MEMORY_LIMIT_MIB"] = str(spec["gpu_memory_mib"])


def execute(args: argparse.Namespace) -> None:
    from experiments.tennis import common_receiver_isolation

    start = time.monotonic()
    validate_args(args)
    manifest.verify(args.manifest, args.carrier_sha256)
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    bind_allocation(stage)
    # Publish only explicit output files; exclude temporary public model copies.
    output = scratch / "common-generation"
    fit = args.base / "publish" / "tennis-base@fit-100.pt"
    weights = dict(zip(WEIGHT_PATHS, map(str, args.weight), strict=True))
    worker: dict[str, Any] = {
        "carrier": str(args.carrier),
        "expected_sha256": args.carrier_sha256,
        "manifest": str(args.manifest),
        "weights": weights,
        "fit": str(fit),
        "frames": args.frames,
        "units": [args.unit],
    }
    common_receiver_isolation.launch(
        source_root=Path(__file__).resolve().parents[2],
        output=output,
        worker_config=worker,
        inputs=[args.carrier, args.manifest, *args.weight, fit],
        denied=args.deny,
        seconds=550 if args.frames == 44 else args.full_cap,
        kind="generation",
    )
    _, files = check_generated(args, output)
    archive = stage / "published.tar"
    if archive.exists():
        raise FileExistsError(archive)
    tmp = stage / ".common-generation.tar"
    with tarfile.open(tmp, "x") as tar:
        for file in files:
            tar.add(file, arcname="publish/" + file.name, recursive=False)
    tmp.rename(archive)
    timing = {
        "archive_sha256": replay.sha256(archive),
        "worker_wall_seconds": time.monotonic() - start,
    }
    (stage / "worker_timing.json").write_text(json.dumps(timing) + "\n")


def validation_workspace() -> tempfile.TemporaryDirectory:
    # The fleet releases stage-local scratch after the producer process exits.
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    scratch.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="receiver-validate-", dir=scratch)


def validate(parser: argparse.ArgumentParser) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    dispatch = json.loads((stage / "dispatch.json").read_text())
    argv = dispatch["command"]
    if argv[1:4] != ["-m", MODULE, "run"]:
        raise ValueError("wrong scientific entrypoint")
    args = parser.parse_args(argv[3:])
    validate_args(args)
    archive = stage / "published.tar"
    timing = json.loads((stage / "worker_timing.json").read_text())
    if timing["archive_sha256"] != replay.sha256(archive):
        raise ValueError("published archive differs")
    with validation_workspace() as tmp:
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if [m.name for m in members] != [
                "publish/generation.json",
                f"publish/{args.unit}.npz",
            ] or not all(m.isfile() and 0 < m.size < 400 * 2**20 for m in members):
                raise ValueError("unexpected generation archive")
            tar.extractall(tmp, filter="data")
        receipt, _ = check_generated(args, Path(tmp) / "publish")
    (row,) = receipt["outputs"]
    if args.frames == 44:
        available = dispatch["campaign_clock"]["budget_end"] - time.monotonic()
        profile = packet._embedded_profile(args.carrier.read_bytes())
        margin = 1.5 if args.run_length == 120 or profile["recipe"]["fps"] != [50, 1] else 1.25
        cost = generation_forecast(timing, row["render_and_save_seconds"], available, args.full_cap, margin, args.run_length)
    else:
        cost = actual_full_cost(timing, args.full_cap)
    report = {
        "passed": True,
        "checks": {
            "source_denied": True,
            "paid_packet_geometry": True,
            "crop_identity_and_coverage": True,
            "native_cuda": True,
            "cost": cost,
        },
    }
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    for key in ("carrier", "manifest", "base"):
        run.add_argument("--" + key, type=Path, required=True)
    run.add_argument("--weight", type=Path, required=True, action="append")
    run.add_argument("--deny", type=Path, required=True, action="append")
    run.add_argument("--carrier-sha256", required=True)
    run.add_argument("--unit", required=True)
    run.add_argument("--frames", type=int, choices=(44, 120, 240), required=True)
    run.add_argument("--run-length", type=int, choices=(120, 240), default=240)
    run.add_argument("--full-cap", required=True, type=int)
    sub.add_parser("validate-stage")
    args = parser.parse_args()
    if args.action == "run":
        execute(args)
    else:
        validate(parser)


if __name__ == "__main__":
    main()
