"""Fleet admission and independent frame checks for the paid two-player finisher."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_finish as finish
from experiments.tennis import common_receiver_isolation as isolation
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import player_requalification as replay
from experiments.tennis.common_receiver_job import (
    actual_full_cost,
    bind_allocation,
    validation_workspace,
)

UNITS = (
    "nadal_kyrgios__002_at120__person_000000000000",
    "nadal_kyrgios__002_at120__person_000000000002",
)

MODULE = "experiments.tennis.common_finish_return_job"


def configure(args: argparse.Namespace) -> tuple[dict, list[Path]]:
    count = {"smoke": 4, "full": 240}[os.environ["PS_STAGE"]]
    if (
        args.ticks != count
        or len(args.generation) != 2
        or len(args.deny) < 4
        or len(set(args.deny)) != len(args.deny)
        or not 1 <= args.full_cap <= 2700
    ):
        raise ValueError("incomplete receiver finish request")
    ffmpeg = Path(sys.prefix) / "bin/ffmpeg"
    files = [args.carrier, args.manifest, ffmpeg, args.sam]
    configs: dict[str, dict] = {}
    receipts: dict[str, dict] = {}
    for unit, directory, seal_hash in args.generation:
        if unit not in UNITS or unit in receipts:
            raise ValueError("unknown or duplicate generated player")
        prefix = Path(directory) / "publish"
        seal = prefix / "generation.json"
        if replay.sha256(seal) != seal_hash:
            raise ValueError("previous generation receipt identity")
        meta = json.loads(seal.read_text())
        (row,) = meta["outputs"]
        if (
            row["unit"] != unit
            or meta["frames"] != 240
            or meta["carrier_sha256"] != args.carrier_sha256
            or row["crops_file"] != unit + ".npz"
        ):
            raise ValueError("prior generation did not cover same 240 ticks and packet")
        crop = prefix / row["crops_file"]
        configs[unit] = {"path": str(crop), "sha256": row["crops_file_sha256"]}
        receipts[unit] = {"path": str(seal), "sha256": seal_hash}
        files.extend([crop, seal])
    return {
        "carrier": str(args.carrier),
        "carrier_sha256": args.carrier_sha256,
        "manifest": str(args.manifest),
        "ffmpeg": str(ffmpeg),
        "sam": str(args.sam),
        "generated": configs,
        "generation_receipts": receipts,
        "ticks": finish.SMOKE if count == 4 else finish.FULL,
    }, files


def verify(output: Path, config: dict, denied: list[Path]) -> tuple[dict, list[Path]]:
    receipt_path = output / "finish.json"
    report = json.loads(receipt_path.read_text())
    ticks = config["ticks"]
    binding = manifest.verify(Path(config["manifest"]), config["carrier_sha256"])
    land = report["isolation"]
    if (
        report["ticks"] != ticks
        or [row["tick"] for row in report["rows"]] != ticks
        or report["carrier_sha256"] != config["carrier_sha256"]
        or report["generated"] != config["generated"]
        or report["generation_receipts"] != config["generation_receipts"]
        or not report["kernels"]
        or report["sam_weight"] != manifest.SAM_WEIGHT
        or report["native_sha256"] != manifest.NATIVE
        or report["total_receiver_bytes"]
        != Path(config["carrier"]).stat().st_size + binding["manifest_bytes"]
        or report["manifest_sha256"] != binding["manifest_sha256"]
        or set(report["order"]) != set(UNITS)
        or land["landlock_abi"] < 3
        or land["interfaces"] != ["lo"]
        or not land["mutation_denied"]
        or any(land["denials"].get(str(p.resolve(strict=True))) not in (1, 13) for p in denied)
    ):
        raise ValueError("invalid receiver inputs, isolation or frame coverage")
    files = [receipt_path, output / "frames.npy", output / "base.yuv"]
    if (
        report["frames_file_sha256"] != replay.sha256(files[1])
        or report["yuv_sha256"] != replay.sha256(files[2])
        or files[2].stat().st_size != 240 * 408 * 480
    ):
        raise ValueError("native or composite digest")
    frames = np.load(files[1], mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8 or frames.shape != (len(ticks), 1080, 1920, 3):
        raise ValueError("composite RGB timeline")
    masks = {}
    for unit in UNITS:
        path = output / (unit + "-masks.npz")
        files.append(path)
        if replay.sha256(path) != report["mask_files"][unit]:
            raise ValueError("receiver matte digest")
        with np.load(path, allow_pickle=False) as z:
            masks[unit] = z["masks"]
            if (
                set(z.files) != {"t", "masks"}
                or not np.array_equal(z["t"], ticks)
                or masks[unit].dtype != bool
                or masks[unit].shape != (len(ticks), 784, 512)
            ):
                raise ValueError("receiver matte shape or ticks")
    cv2.setNumThreads(1)
    native = np.memmap(files[2], np.uint8, mode="r", shape=(240, 408, 480))
    profile, _, geometry = finish.geometry(
        Path(config["carrier"]).read_bytes(), config["carrier_sha256"]
    )
    if report["order"] != profile["order"] or report["yuv_sha256"] != finish.YUV_SHA:
        raise ValueError("paid compositing order or native YUV identity")
    generated = {}
    for name in UNITS:
        with np.load(config["generated"][name]["path"], allow_pickle=False) as stored:
            generated[name] = stored["crops"]
    for i, (tick, row) in enumerate(zip(ticks, report["rows"], strict=True)):
        if row["frame_sha256"] != replay.array_sha(frames[i]) or set(row["status"]) != set(UNITS):
            raise ValueError(f"composite digest or object disposition at {tick}")
        expected = cv2.resize(
            finish.yuv_to_rgb(native[tick], 480, 272, False),
            (1920, 1080),
            interpolation=cv2.INTER_CUBIC,
        )
        if row["base_sha256"] != replay.array_sha(expected):
            raise ValueError("native base differs at this tick")
        union = np.zeros(expected.shape[:2], bool)
        for name in profile["order"]:
            x, y, w, h = finish.k6b.int_box(geometry[name][2][tick])
            x0, y0, x1, y1 = max(0, x), max(0, y), min(1920, x + w), min(1080, y + h)
            count = 0
            if x1 > x0 and y1 > y0:
                crop_roi = np.s_[y0 - y : y1 - y, x0 - x : x1 - x]
                mask = cv2.resize(
                    masks[name][i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )[crop_roi].astype(bool)
                count = int(mask.sum())
                fg = cv2.resize(generated[name][tick], (w, h), interpolation=cv2.INTER_LANCZOS4)[
                    crop_roi
                ]
                expected[y0:y1, x0:x1][mask] = fg[mask]
                union[y0:y1, x0:x1] |= mask
            if count != row["projected_pixels"][name]:
                raise ValueError("published player coverage differs from projected mask")
        if int(union.sum()) != row["coverage_pixels"] or not np.array_equal(expected, frames[i]):
            raise ValueError(f"independent raster differs for tick {tick}")
        for unit in UNITS:
            status = row["status"][unit]
            area = row["projected_pixels"][unit]
            if (
                status
                not in {
                    "composed",
                    "empty_mask",
                    "offscreen",
                    "insufficient_prompt",
                    "missing_pose",
                }
                or (status == "composed") != (area > 0)
                or (status == "composed" and not masks[unit][i].any())
            ):
                raise ValueError(f"mask evidence/state differs at tick {tick}")
        if not row["coverage_pixels"] and row["base_sha256"] != row["frame_sha256"]:
            raise ValueError("unexplained change on fallback frame")
    if not all(any(row["status"][name] == "composed" for row in report["rows"]) for name in UNITS):
        raise ValueError("missing learned reconstruction for a player")
    return report, files


def execute(args: argparse.Namespace) -> None:
    start = time.monotonic()
    config, inputs = configure(args)
    out = Path(os.environ["PS_SCRATCH_DIR"]) / "completed-receiver"
    stage = Path(os.environ["PS_STAGE_DIR"])
    bind_allocation(stage)
    isolation.launch(
        source_root=Path(__file__).resolve().parents[2],
        output=out,
        worker_config=config,
        inputs=inputs,
        denied=args.deny,
        seconds=550 if args.ticks == 4 else args.full_cap,
        kind="finish",
    )
    verify_start = time.monotonic()
    _, files = verify(out, config, args.deny)
    verify_seconds = time.monotonic() - verify_start
    tmp = stage / ".finish.tar"
    dest = stage / "published.tar"
    if dest.exists():
        raise FileExistsError(dest)
    publish_start = time.monotonic()
    with tarfile.open(tmp, "x") as tar:
        for path in files:
            tar.add(path, arcname="publish/" + path.name, recursive=False)
    tmp.rename(dest)
    digest = replay.sha256(dest)
    timing = {
        "worker_wall_seconds": time.monotonic() - start,
        "archive_seconds": time.monotonic() - publish_start,
        "verify_seconds": verify_seconds,
        "archive_bytes": dest.stat().st_size,
        "archive_sha256": digest,
    }
    (stage / "worker_timing.json").write_text(json.dumps(timing) + "\n")


def validate(parser: argparse.ArgumentParser) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    dispatch = json.loads((stage / "dispatch.json").read_text())
    args = dispatch["command"]
    if args[1:4] != ["-m", MODULE, "run"]:
        raise ValueError("invalid finisher dispatch")
    request = parser.parse_args(args[3:])
    config, _ = configure(request)
    timing = json.loads((stage / "worker_timing.json").read_text())
    archive = stage / "published.tar"
    if replay.sha256(archive) != timing["archive_sha256"]:
        raise ValueError("finisher archive identity")
    with validation_workspace() as tmp:
        with tarfile.open(archive) as tar:
            names = ["finish.json", "frames.npy", "base.yuv", *(u + "-masks.npz" for u in UNITS)]
            if [m.name for m in tar.getmembers()] != ["publish/" + n for n in names] or not all(
                m.isfile() and 0 < m.size < 2_000_000_000 for m in tar.getmembers()
            ):
                raise ValueError("finisher archive contents")
            tar.extractall(tmp, filter="data")
        report, _ = verify(Path(tmp) / "publish", config, request.deny)
    if request.ticks == 240:
        cost = actual_full_cost(timing, request.full_cap)
    else:
        # Decode/load are fixed. Rendering, evidence verification and archive publication grow with output.
        four = report["loop_seconds"]
        wall = timing["worker_wall_seconds"]
        arch = timing["archive_seconds"]
        verify_seconds = timing["verify_seconds"]
        remaining = dispatch["campaign_clock"]["budget_end"] - time.monotonic()
        if (
            not all(
                math.isfinite(v) and v > 0 for v in (four, wall, arch, verify_seconds, remaining)
            )
            or wall < four + arch + verify_seconds
        ):
            raise ValueError("invalid finisher clock")
        # Verification reopens both full-generation NPZs at either scale. Charge processing time proportional to the output bytes.
        ratio = 1_550_000_000 / timing["archive_bytes"]
        forecast = (
            1.25
            * (wall - four - arch - verify_seconds + 60 * four + ratio * (arch + verify_seconds))
            + 45
        )
        if forecast > min(request.full_cap, remaining - 60):
            raise ValueError(f"finisher forecast {forecast:.1f}s exceeds affordable full stage")
        cost = {
            "forecast_seconds": forecast,
            "remaining_seconds": remaining,
            "full_cap_seconds": request.full_cap,
        }
    proof = {
        "passed": True,
        "checks": {
            "source_denied": True,
            "both_players_composed": True,
            "decoded_frame_and_mask_identities": True,
            "native_cuda_kernels": True,
        },
        "cost": cost,
    }
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(proof) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    r = subs.add_parser("run")
    for name in ("carrier", "manifest", "sam"):
        r.add_argument("--" + name, type=Path, required=True)
    r.add_argument("--carrier-sha256", required=True)
    r.add_argument("--full-cap", type=int, required=True)
    r.add_argument("--ticks", type=int, choices=(4, 240), required=True)
    r.add_argument("--deny", type=Path, action="append", required=True)
    r.add_argument(
        "--generation",
        nargs=3,
        action="append",
        required=True,
        metavar=("UNIT", "BUNDLE", "RECEIPT_SHA256"),
    )
    subs.add_parser("validate-stage")
    args = parser.parse_args()
    if args.action == "run":
        execute(args)
    else:
        validate(parser)


if __name__ == "__main__":
    main()
