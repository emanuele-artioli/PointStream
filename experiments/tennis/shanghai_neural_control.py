"""Produce one label-blind native DCVC-UF comparison for the Shanghai ball study."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from experiments.tennis import w3_neural_baseline as w3
from experiments.tennis.k4b_features import convert_command
from experiments.tennis.shanghai_ball_display_freeze import SOURCE, sha
from src.codecs import svtav1

FRAMES = {"smoke": 48, "full": 240}


def run(args):
    import torch

    start = time.monotonic()
    n = FRAMES[os.environ["PS_STAGE"]]
    if args.frames != n or sha(args.source) != SOURCE:
        raise ValueError("Unexpected source or frame range")
    for name in ("image_ckpt", "video_ckpt"):
        if sha(getattr(args, name)) != w3.INPUTS[name]:
            raise ValueError(f"Unexpected {name}")
    model = torch.cuda.get_device_name(0)
    if "6000 Ada" not in model or torch.cuda.device_count() != 1:
        raise ValueError("Expected one visible qualified Ada GPU")
    if (
        os.environ.get("PS_GPU_SHARING") != "shared"
        or os.environ.get("PS_GPU_MEMORY_LIMIT_MIB") != "8192"
    ):
        raise ValueError("Requires qualified 8 GiB shared allocation")
    torch.cuda.set_per_process_memory_fraction(
        2048 * 2**20 / torch.cuda.get_device_properties(0).total_memory
    )
    root = Path(os.environ["PS_SCRATCH_DIR"])
    receipts = root / "shared-resource-receipts"
    receipts.mkdir()
    os.environ["PS_K5D_SHARED_NUMERICAL"] = "pointstream-parent2-child6-v1"
    os.environ["PS_K5D_RESOURCE_RECEIPTS"] = str(receipts)
    publish = root / "publish"
    publish.mkdir()
    ffmpeg = svtav1.tool("ffmpeg")
    source = root / "source.yuv"
    command = convert_command(ffmpeg["path"], args.source, source, 0, n)
    subprocess.run(command, check=True, capture_output=True, timeout=180)
    if source.stat().st_size != n * 1920 * 1080 * 3 // 2:
        raise ValueError("Incomplete native input")
    args.stream_policy = "legacy"
    point = w3.code_point(
        args,
        source,
        root,
        25,
        n,
        {"qp": args.qp, "scale": args.scale, "target_kbps": 20.4, "est_kbps": None},
    )
    frame_bytes = w3.SCALES[args.scale][0] * w3.SCALES[args.scale][1] * 3 // 2
    if not point["accepted"]:
        raise ValueError(
            f"DCVC independent receiver check failed: {point.get('failure', point.get('max_decode_diff'))}"
        )
    raw = Path(point["decoded"])
    reports = json.loads((raw.parent / "reports.json").read_text())
    with raw.open("rb") as f:
        hashes = [hashlib.sha256(f.read(frame_bytes)).hexdigest() for _ in range(n)]
        if f.read(1) or raw.stat().st_size != frame_bytes * n:
            raise ValueError("Decoded YUV size")
    passes = [p for d in reports["decode"].values() for p in d["passes"]]
    if len(passes) != 4 or any(
        p["frame_sha256"] != hashes or p["yuv420_sha256"] != hashes for p in passes
    ):
        raise ValueError("Incomplete four-pass YUV verification")
    for name, path in [
        ("stream.psdc", raw.parent / "stream.psdc"),
        ("decoded.yuv", raw),
        ("reports.json", raw.parent / "reports.json"),
    ]:
        shutil.copyfile(path, publish / name)
    resources = [json.loads(p.read_text()) for p in sorted(receipts.glob("child-*.json"))]
    summary = dict(
        stage=os.environ["PS_STAGE"],
        frames=n,
        source_sha256=SOURCE,
        conversion=command,
        ffmpeg=ffmpeg,
        gpu=model,
        resources=resources,
        native=point,
        fps=25,
        qp=args.qp,
        scale=args.scale,
        raw_sha256=sha(raw),
        yuv_frames=hashes,
        elapsed_seconds=time.monotonic() - start,
    )
    (publish / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


def validate(args):
    stage = Path(os.environ["PS_STAGE_DIR"])
    n = FRAMES[os.environ["PS_STAGE"]]
    width, height = w3.SCALES[args.scale]
    frame_bytes = width * height * 3 // 2
    with tarfile.open(stage / "published.tar") as tar:

        def member(name):
            return tar.extractfile("publish/" + name)

        report = json.load(member("result.json"))
        point = report["native"]
        frames = member("decoded.yuv")
        h = hashlib.sha256()
        digests = []
        for _ in range(n):
            data = frames.read(frame_bytes)
            if len(data) != frame_bytes:
                raise ValueError("Incomplete decoded frame")
            h.update(data)
            digests.append(hashlib.sha256(data).hexdigest())
        packet = member("stream.psdc").read()
        full_reports = json.load(member("reports.json"))
        passes = [p for d in full_reports["decode"].values() for p in d["passes"]]
        checks = dict(
            source=report["source_sha256"] == SOURCE,
            count=report["frames"] == n and not frames.read(1),
            native=point["accepted"]
            and point["qp"] == args.qp
            and point["size"] == [width, height]
            and report.get("qp", 24) == args.qp
            and report.get("scale", 2) == args.scale,
            allocator=Counter(r["action"] for r in report["resources"])
            == {"encode": 1, "decode": 2}
            and all(r["budget_mib"] == 6144 for r in report["resources"]),
            packet=len(packet) == point["bytes"]
            and hashlib.sha256(packet).hexdigest() == point["container_sha256"],
            rate=point["kbps"] == len(packet) * 8 * 25 / n / 1000,
            raw=h.hexdigest() == report["raw_sha256"],
            repeat=len(passes) == 4
            and all(p["frame_sha256"] == p["yuv420_sha256"] == digests for p in passes),
        )
    forecast = 1.5 * report["elapsed_seconds"] * 240 / n + 60
    if n == 48:
        checks["forecast"] = forecast < 1650
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks, forecast_seconds=forecast)) + "\n"
    )
    if not ok:
        raise ValueError(checks)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "image-ckpt", "video-ckpt"]:
        cli.add_argument("--" + key, type=Path)
    cli.add_argument("--frames", type=int)
    cli.add_argument("--qp", type=int, default=24, choices=range(64))
    cli.add_argument("--scale", type=int, default=2, choices=(2, 4))
    cli.add_argument("--validate", action="store_true")
    args = cli.parse_args()
    validate(args) if args.validate else run(args)
