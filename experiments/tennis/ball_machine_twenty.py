"""Judge the exact decoded 20 kbps example and read its paid ball symbols."""

import argparse
import hashlib
import json
import math
import os
import statistics
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.tennis import ball_machine_three as shared
from experiments.tennis import k5b_trackers, k5d_source_oracle
from src.codecs import svtav1
from src.pointstream import ball, ball_stream, tracked_patch


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rgb_frames(path, n, expected_size, hashes=None):
    reader = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if [
            reader.get(k)
            for k in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
                cv2.CAP_PROP_FRAME_COUNT,
            )
        ] != [*expected_size, 50, 240]:
            raise ValueError("Unexpected input movie")
        for t in range(n):
            ok, frame = reader.read()
            if not ok:
                raise ValueError("Missing input frame")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if hashes and hashlib.sha256(image).hexdigest() != hashes[str(t)]:
                raise ValueError("Lossless score copy differs from the actual receiver")
            yield image
    finally:
        reader.release()


def run(a):
    start = time.monotonic()
    if len(os.sched_getaffinity(0)) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("Judge allocation")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    root = Path(os.environ["PS_SCRATCH_DIR"])
    working = root / "judge"
    working.mkdir()
    with tarfile.open(a.packets) as archive:
        archive.extractall(working, filter="data")
    binding = json.loads((working / "binding.json").read_text())
    if (
        sha(a.source) != binding["source_sha256"]
        or sha(working / "shown.mkv") != binding["display_sha256"]
    ):
        raise ValueError("Source/display identity")
    k5d_source_oracle.bind_config(a.wasb_code, shared.CONFIG)
    if sha(a.wasb_weights / "wasb_tennis_best.pth.tar") != shared.WEIGHT:
        raise ValueError("WASB identity")
    model_start = time.monotonic()
    zoo = k5b_trackers.Zoo(a.wasb_code, a.wasb_weights, names=("wasb-step1",))
    settings = zoo.settings["wasb-step1"]
    if next(settings["net"].parameters()).device.type != "cuda":
        raise ValueError("CUDA judge required")
    model_seconds = time.monotonic() - model_start
    wire = (working / "pointstream.psbl").read_bytes()
    if hashlib.sha256(wire).hexdigest() != binding["original_wire_sha256"]:
        raise ValueError("Wrong wire packet")
    parse_ns = []
    for _ in range(11):
        begin = time.perf_counter_ns()
        parsed = ball_stream.unpack(wire)
        positions, _looks = ball.decode_frames(parsed.ball_payload, 240, 960, 544)
        parse_ns.append(time.perf_counter_ns() - begin)
    decoder = svtav1.tool("dav1d")
    labels = [r for r in binding["labels"] if r["tick"] < a.frames]
    direct = {}
    for item in labels:
        tick = item["tick"]
        x, y = item["ball"]["center_xy"]
        pt = positions[tick]
        distance = None if pt is None else math.hypot(pt[0] * 2 - x, pt[1] * 1080 / 544 - y)
        old = next(row for row in binding["source_positions"] if row["tick"] == tick)["error_px"]
        if (distance is None) != (old is None) or (
            distance is not None and abs(distance - old) > 1e-8
        ):
            raise ValueError("Direct output changed from the visual experiment")
        direct[tick] = distance
    results = {}
    for arm in ["source", "pointstream", *binding["references"]]:
        if arm == "source":
            gen = rgb_frames(a.source, a.frames, (1920, 1080))
            size = None
        elif arm == "pointstream":
            gen = rgb_frames(working / "shown.mkv", a.frames, (960, 544), binding["hashes"])
            size = len(wire)
        else:
            ref = binding["references"][arm]
            path = working / (arm + ".ivf")
            if sha(path) != ref["sha256"]:
                raise ValueError("Wrong native control")
            size = path.stat().st_size
            output = working / (arm + ".yuv")
            w, h = ref["width"], ref["height"]
            tracked_patch.decode_native(
                path.read_bytes(),
                output,
                dict(width=w, height=h, fps=[50, 1], frames=240),
                decoder["path"],
            )
            if sha(output) != ref["yuv_sha256"]:
                raise ValueError("Actual native decode differs")
            raw = np.memmap(output, np.uint8, "r", shape=(240, h * 3 // 2, w))
            gen = (tracked_patch.yuv_to_rgb(f, w, h, False) for f in raw[: a.frames])
        images = np.empty((a.frames, 720, 1280, 3), np.uint8)
        for t, im in enumerate(gen):
            display = (
                im
                if arm == "source"
                else cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            )
            images[t] = cv2.resize(display, (1280, 720), interpolation=cv2.INTER_AREA)
        begin = time.monotonic()
        preds = zoo.run("wasb-step1", images)
        torch.cuda.synchronize()
        secs = time.monotonic() - begin
        rows = []
        for r in labels:
            t = r["tick"]
            if r["ball"]["status"] != "visible":
                raise ValueError("Unexpected source support")
            x, y = r["ball"]["center_xy"]
            present, px, py = preds[t]
            error = math.hypot(1.5 * px - x, 1.5 * py - y) if present else None
            rows.append(
                dict(
                    tick=t,
                    error=error,
                    hit5=error is not None and error <= 5,
                    hit10=error is not None and error <= 10,
                )
            )
        results[arm] = dict(
            bytes=size,
            kbps=size / 600 if size else None,
            rows=rows,
            input_rgb_sha256=hashlib.sha256(images).hexdigest(),
            inference_seconds=secs,
        )
    report = dict(
        frames=a.frames,
        inputs_sha256=sha(a.packets),
        original_source=sha(a.source),
        labels=binding["label_sha256"],
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        weights=shared.WEIGHT,
        config=shared.CONFIG,
        settings=[
            settings["cfg"]["model"]["frames_in"],
            settings["cfg"]["model"]["frames_out"],
            settings["cfg"]["detector"]["step"],
            settings["cfg"]["detector"]["postprocessor"]["score_threshold"],
        ],
        model_seconds=model_seconds,
        elapsed_seconds=time.monotonic() - start,
        decoder=decoder,
        results=results,
        direct_errors=direct,
        complete_packet_feature_parse_median_ns=statistics.median(parse_ns),
        feature_parse_trials_ns=parse_ns,
    )
    publish = root / "publish"
    publish.mkdir()
    (publish / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar") as archive:
        r = json.load(archive.extractfile("publish/result.json"))
    smoke = os.environ["PS_STAGE"] == "smoke"
    n = 24 if smoke else 240
    tests = dict(
        count=r["frames"] == n,
        arms=len(r["results"]) == 5,
        weights=r["weights"] == shared.WEIGHT,
        gpu=r["capability"] == [8, 9],
        settings=r["settings"] == [3, 3, 1, 0.5],
        packet=r["results"]["pointstream"]["bytes"] == 11979,
        plain=r["results"]["plain160Q61"]["bytes"] == 12038,
        all_ticks=all(len(x["rows"]) == (1 if smoke else 12) for x in r["results"].values()),
    )
    forecast = (
        1.5 * (r["elapsed_seconds"] - r["model_seconds"]) * 240 / n + r["model_seconds"] + 50
        if smoke
        else None
    )
    tests["fits"] = not smoke or forecast < 570
    result = dict(passed=all(tests.values()), checks=tests, forecast=forecast)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "packets", "wasb-code", "wasb-weights"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--frames", type=int, choices=(24, 240))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate() if args.validate else run(args)
