"""Apply the frozen ball judge to authentic near-22 kbps receiver images."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis import ball_machine_three as shared
from experiments.tennis import ball_machine_twenty as earlier
from experiments.tennis import common_player_packet, k5b_trackers, k5d_source_oracle, w8_payload
from src.pointstream import ball
from src.codecs import svtav1
from src.pointstream.tracked_patch import decode_native, yuv_to_rgb


def staged_member(archive, member, target):
    entry = archive.getmember(member)
    if not entry.isfile():
        raise ValueError("Expected regular archive member")
    with archive.extractfile(entry) as source, target.open("xb") as dest:
        shutil.copyfileobj(source, dest, 1 << 20)


def run(args):
    import torch

    started = time.monotonic()
    if torch.cuda.device_count() != 1 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("WASB resource allocation")
    if os.environ.get("PS_GPU_SHARING") == "shared":
        limit = int(os.environ["PS_GPU_MEMORY_LIMIT_MIB"])
        if limit != 8000:
            raise ValueError("Unexpected GPU limit")
        torch.cuda.set_per_process_memory_fraction(
            limit * 1024**2 / torch.cuda.get_device_properties(0).total_memory, 0
        )
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    root = Path(os.environ["PS_SCRATCH_DIR"])
    with tarfile.open(args.packets) as tar:
        binding = json.load(tar.extractfile("binding.json"))
        staged_member(tar, "ordinary.ivf", root / "ordinary.ivf")
        staged_member(tar, "joint.mkv", root / "joint.mkv")
    if (
        earlier.sha(root / "joint.mkv") != binding["joint_display"]["file"]
        or earlier.sha(root / "ordinary.ivf") != binding["ordinary"]["stream"]
    ):
        raise ValueError("Staged receiver identity")
    decoder = svtav1.tool("dav1d")
    decode_native(
        (root / "ordinary.ivf").read_bytes(),
        root / "ordinary.yuv",
        dict(width=240, height=136, fps=[50, 1], frames=240),
        decoder["path"],
    )
    if earlier.sha(root / "ordinary.yuv") != binding["ordinary"]["yuv"]:
        raise ValueError("Ordinary native-decoded identity")
    ordinary = np.memmap(root / "ordinary.yuv", np.uint8, "r", shape=(240, 204, 240))
    with tarfile.open(args.neural) as tar:
        replay = json.load(tar.extractfile("publish/stroke.json"))
        staged_member(tar, "publish/full.yuv", root / "full.yuv")
    if (
        replay["frames_verified"] != 240
        or replay["packet_sha256"] != binding["native_neural_sha256"]
        or replay["full_yuv_sha256"]
        != "48f7dd5973fc376f6fabe736e37cdfc0f79588479d29ceb4e9e14943e48e2dbf"
        or earlier.sha(root / "full.yuv") != replay["full_yuv_sha256"]
        or (root / "full.yuv").stat().st_size != 240 * 816 * 960
    ):
        raise ValueError("Native neural decoder identity or frame count")
    raw = np.memmap(root / "full.yuv", np.uint8, "r", shape=(240, 816, 960))
    wire = args.wire.read_bytes()
    if (
        hashlib.sha256(wire).hexdigest() != binding["joint_wire_sha256"]
        or len(wire) != binding["costs_bytes"]["joint"]
    ):
        raise ValueError("Joint packet size or identity")
    parse_ns = []
    for _ in range(11):
        start_parse = time.perf_counter_ns()
        profile = common_player_packet._embedded_profile(wire)
        parts, _ = w8_payload.unpack(wire, profile)
        positions, _ = ball.decode_frames(parts["trajectory"], 240, 960, 544)
        parse_ns.append(time.perf_counter_ns() - start_parse)
    direct = {}
    for tick, (x, y) in binding["marks"].items():
        if int(tick) < args.frames:
            pos = positions[int(tick)]
            direct[tick] = (
                None if pos is None else math.hypot(2 * pos[0] - x, pos[1] * 1080 / 544 - y)
            )
    k5d_source_oracle.bind_config(args.wasb_code, shared.CONFIG)
    if earlier.sha(args.wasb_weights / "wasb_tennis_best.pth.tar") != shared.WEIGHT:
        raise ValueError("WASB checkpoint")
    start_model = time.monotonic()
    zoo = k5b_trackers.Zoo(args.wasb_code, args.wasb_weights, names=("wasb-step1",))
    model = zoo.settings["wasb-step1"]
    if next(model["net"].parameters()).device.type != "cuda":
        raise RuntimeError("CPU detector")
    cfg = model["cfg"]
    settings = [
        cfg["model"]["frames_in"],
        cfg["model"]["frames_out"],
        cfg["detector"]["step"],
        cfg["detector"]["postprocessor"]["score_threshold"],
    ]
    initialization = time.monotonic() - start_model
    results = {}
    for arm in ("ordinary", "joint", "neural"):
        images = np.empty((args.frames, 720, 1280, 3), np.uint8)
        if arm == "joint":
            full_digest = hashlib.sha256()
            for i, frame in enumerate(
                earlier.rgb_frames(root / "joint.mkv", args.frames, (1920, 1080))
            ):
                full_digest.update(frame)
                images[i] = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            if (
                full_digest.hexdigest()
                != binding["joint_display"]["full" if args.frames == 240 else "smoke"]
            ):
                raise ValueError("Joint receiver pixels changed")
        else:
            frames, w, h = (ordinary, 240, 136) if arm == "ordinary" else (raw, 960, 544)
            for i, frame in enumerate(frames[: args.frames]):
                native = yuv_to_rgb(frame, w, h, False)
                full = cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                images[i] = cv2.resize(full, (1280, 720), interpolation=cv2.INTER_AREA)
        digest = hashlib.sha256(images).hexdigest()
        if (
            arm == "ordinary"
            and args.frames == 240
            and digest != binding["ordinary"]["prior_judge_input"]
        ):
            raise ValueError("Ordinary 240-frame judge input changed")
        inference_start = time.monotonic()
        pred = zoo.run("wasb-step1", images)
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - inference_start
        rows = []
        for tick_text, (x, y) in binding["marks"].items():
            t = int(tick_text)
            if t >= args.frames:
                continue
            present, px, py = pred[t]
            error = math.hypot(1.5 * px - x, 1.5 * py - y) if present else None
            rows.append(
                dict(
                    tick=t,
                    error=error,
                    hit5=error is not None and error <= 5,
                    hit10=error is not None and error <= 10,
                )
            )
        if arm == "ordinary" and args.frames == 240 and rows != binding["ordinary_prior"]:
            raise ValueError("Same judge disagrees with archived ordinary control")
        results[arm] = dict(
            rows=rows,
            hits5=sum(r["hit5"] for r in rows),
            hits10=sum(r["hit10"] for r in rows),
            input_sha256=digest,
            inference_seconds=inference_seconds,
        )
        del images
    record = dict(
        frames=args.frames,
        direct_errors=direct,
        direct_parse_median_ns=statistics.median(parse_ns),
        parse_samples_ns=parse_ns,
        results=results,
        costs_bytes=binding["costs_bytes"],
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        model_seconds=initialization,
        elapsed_seconds=time.monotonic() - started,
        settings=settings,
        weights=shared.WEIGHT,
        config=shared.CONFIG,
        neural_full_yuv_sha256=replay["full_yuv_sha256"],
        neural_packet_sha256=replay["packet_sha256"],
        joint_wire_sha256=binding["joint_wire_sha256"],
        label_sha256=binding["labels_sha256"],
        ordinary_prior_agreement=args.frames == 240,
    )
    pub = root / "publish"
    pub.mkdir()
    (pub / "result.json").write_text(json.dumps(record, indent=2) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar") as tar:
        record = json.load(tar.extractfile("publish/result.json"))
    smoke = os.environ["PS_STAGE"] == "smoke"
    n = 24 if smoke else 240
    ticks = [t for t in range(10, 240, 20) if t < n]
    forecast = (
        1.5 * (record["elapsed_seconds"] - record["model_seconds"]) * 240 / n
        + record["model_seconds"]
        + 50
    )
    checks = dict(
        frames=record["frames"] == n,
        arms=set(record["results"]) == {"ordinary", "joint", "neural"},
        labels=all([v["tick"] for v in r["rows"]] == ticks for r in record["results"].values()),
        cuda=record["capability"] == [8, 9],
        config=record["config"] == shared.CONFIG and record["weights"] == shared.WEIGHT,
        settings=record["settings"] == [3, 3, 1, 0.5],
        costs=record["costs_bytes"]
        == dict(ordinary=12974, joint=12927, joint_cold=257266, neural=12270),
        comparison=smoke or record["ordinary_prior_agreement"],
        direct=smoke
        or sum(d is not None and d <= 10 for d in record["direct_errors"].values()) == 10,
        forecast=not smoke or forecast < 570,
    )
    valid = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=valid, checks=checks, forecast=forecast if smoke else None)) + "\n"
    )
    if not valid:
        raise ValueError(str(checks))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("packets", "wire", "neural", "wasb-code", "wasb-weights"):
        cli.add_argument("--" + name, type=Path)
    cli.add_argument("--frames", type=int, choices=(24, 240))
    cli.add_argument("--validate", action="store_true")
    args = cli.parse_args()
    validate() if args.validate else run(args)
