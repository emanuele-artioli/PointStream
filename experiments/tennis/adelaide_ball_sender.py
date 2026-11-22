"""Run the frozen SVT62 ball-placement sender on an outcome-unopened match."""

import argparse
import hashlib
import json
import os
import tarfile
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.tennis import ball_confirmation as prior
from experiments.tennis import fresh_ball_sender, k5b1b, k5b_trackers, k5c_independent_ball
from experiments.tennis import k5d_encoder, windows
from src.codecs import svtav1
from src.pointstream import ball

SOURCE = "5a81a835c16a1b511f4b581748254b809c3a5d96105113974f6a49b7c8e2182b"
LABELS = "6bbf0e3bb81a586aa3981edc40f4ea5eab0a0b36542f626d61fe12614cf14f1d"
FPS = Fraction(25)
SIZE = (1280, 720)


def source_frames(path, n):
    if prior.sha(path) != SOURCE:
        raise ValueError("Wrong original video")
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        props = [
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FRAME_COUNT,
        ]
        if [cap.get(k) for k in props] != [1920, 1080, 25, 255]:
            raise ValueError("Unexpected original stream geometry or cadence")
        frames = np.empty((n, 720, 1280, 3), np.uint8)
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Source ended early")
            frames[i] = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), SIZE, interpolation=cv2.INTER_AREA
            )
        return frames
    finally:
        cap.release()


def run(a):
    start = time.monotonic()
    cpus = os.sched_getaffinity(0)
    if (
        len(cpus) != 1
        or prior.sha(a.archive) != fresh_ball_sender.MODEL_CODE
        or prior.sha(a.weights) != fresh_ball_sender.MODEL_WEIGHT
    ):
        raise ValueError("CPU allocation or tracker inputs differ")
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "adelaide-sender"
    work.mkdir()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    with tarfile.open(a.archive) as archive:
        archive.extractall(work / "model", filter="data")
    tracker = k5b_trackers.BallTrack(work / "model", a.weights, device="cpu")
    if tracker.net.training or next(tracker.net.parameters()).device.type != "cpu":
        raise ValueError("Wrong tracker inference mode")

    def finite(_module, _inputs, out):
        if not torch.isfinite(out[0]).all():
            raise ValueError("Nonfinite tracker output")

    tracker.net.register_forward_hook(finite)
    source = source_frames(a.source, a.frames)
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    convert = k5d_encoder.convert_rgb(tools["ffmpeg"]["path"], source, work / "source.yuv", FPS)
    native = work / "native"
    native.mkdir()
    rec = svtav1.code(
        work / "source.yuv",
        native,
        width=1280,
        height=720,
        fps=FPS,
        frames=a.frames,
        crf=62,
        preset=4,
        threads=1,
        full_range=False,
        encoder=tools["SvtAv1EncApp"]["path"],
        decoder=tools["dav1d"]["path"],
        timeout=300,
    )
    decoded = windows.rgb_frames(Path(rec["decoded"]), 1280, 720, a.frames, work / "base.npy")
    pred = tracker.run(source)
    if a.frames == 16 and pred != tracker.run(source):
        raise ValueError("Repeated smoke tracker changed")
    initial = k5b1b.track_of(pred, (720, 1280))
    (fit,) = k5c_independent_ball.fit_streak(initial, source, [decoded], (720, 1280))
    positions, appearance = fit["track"], fit["appearances"]
    guard = k5d_encoder.pooled_psnr(
        source, k5c_independent_ball.render_frames(decoded, positions, appearance)
    ) < k5d_encoder.pooled_psnr(source, decoded)
    if guard:
        positions, appearance = [None] * a.frames, [None] * a.frames
    payload = ball.encode_frames(positions, appearance, 1280, 720)
    if ball.decode_frames(payload, a.frames, 1280, 720) != (positions, appearance):
        raise ValueError("Sender quantization did not round trip")
    result = dict(
        positions=positions,
        source_sha256=SOURCE,
        native=rec,
        tools=tools,
        convert=convert,
        native_sha256=prior.sha(Path(rec["stream"])),
        model=tracker.meta,
        code=fresh_ball_sender.MODEL_CODE,
        weight=fresh_ball_sender.MODEL_WEIGHT,
        guard=guard,
        rgb_sha256=hashlib.sha256(source).hexdigest(),
        frames=a.frames,
        cpus=sorted(cpus),
        seconds=time.monotonic() - start,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "sender.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "sender.json").read_text())
    n = 16 if os.environ["PS_STAGE"] == "smoke" else 64
    prior.checked_track(r["positions"], n)
    checks = dict(
        source=r["source_sha256"] == SOURCE,
        coder=r["native"]["decoded_frames"] == n,
        sender=r["weight"] == fresh_ball_sender.MODEL_WEIGHT
        and r["code"] == fresh_ball_sender.MODEL_CODE,
        allocation=len(r["cpus"]) == 1,
        length=r["frames"] == len(r["positions"]) == n,
    )
    forecast = 1.5 * r["seconds"] * 64 / n + 80
    checks["fits"] = n != 16 or forecast <= 1400
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=all(checks.values()), checks=checks, forecast_seconds=forecast))
        + "\n"
    )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ("source", "archive", "weights"):
        p.add_argument("--" + field, type=Path)
    p.add_argument("--frames", type=int, choices=(16, 64))
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
