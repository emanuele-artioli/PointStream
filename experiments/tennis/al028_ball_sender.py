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

SOURCE = "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
SOURCE_PROFILES = {
    "al028": (SOURCE, 492),
    "shanghai": ("0afb0fb191881def1235285b58fe54814556c83fed7815cd99a00b92d26bc86c", 250),
}
LABELS = "6bbf0e3bb81a586aa3981edc40f4ea5eab0a0b36542f626d61fe12614cf14f1d"
FPS = Fraction(25)
SIZE = (960, 544)


def source_frames(path, n, scene="al028"):
    expected, total = SOURCE_PROFILES[scene]
    if not 0 < n <= total or prior.sha(path) != expected:
        raise ValueError("Wrong original video")
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        props = [
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FRAME_COUNT,
        ]
        if [cap.get(k) for k in props] != [1920, 1080, 25, total]:
            raise ValueError("Unexpected original stream geometry or cadence")
        frames = np.empty((n, 544, 960, 3), np.uint8)
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
    raise RuntimeError("Retired: this historical wrapper misinterprets tracker coordinates; use al028_ball_chunked and al028_ball_fit instead")
    start = time.monotonic()
    cpus = os.sched_getaffinity(0)
    if (
        len(cpus) != 1
        or prior.sha(a.archive) != fresh_ball_sender.MODEL_CODE
        or prior.sha(a.weights) != fresh_ball_sender.MODEL_WEIGHT
    ):
        raise ValueError("CPU allocation or tracker inputs differ")
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "al028-ball-sender"
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
        width=960,
        height=544,
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
    decoded = windows.rgb_frames(Path(rec["decoded"]), 960, 544, a.frames, work / "base.npy")
    pred = tracker.run(source)
    if a.frames == 24 and pred != tracker.run(source):
        raise ValueError("Repeated smoke tracker changed")
    initial = k5b1b.track_of(pred, (544, 960))
    (fit,) = k5c_independent_ball.fit_streak(initial, source, [decoded], (544, 960))
    positions, appearance = fit["track"], fit["appearances"]
    guard = k5d_encoder.pooled_psnr(
        source, k5c_independent_ball.render_frames(decoded, positions, appearance)
    ) < k5d_encoder.pooled_psnr(source, decoded)
    if guard:
        positions, appearance = [None] * a.frames, [None] * a.frames
    payload = ball.encode_frames(positions, appearance, 960, 544)
    if ball.decode_frames(payload, a.frames, 960, 544) != (positions, appearance):
        raise ValueError("Sender quantization did not round trip")
    (Path(os.environ["PS_STAGE_DIR"]) / "ball.payload").write_bytes(payload)
    result = dict(
        payload_sha256=__import__("hashlib").sha256(payload).hexdigest(),
        payload_bytes=len(payload),
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
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    prior.checked_track(r["positions"], n)
    payload=(stage / "ball.payload").read_bytes()
    tracks, appearances=ball.decode_frames(payload, n, 960, 544)
    checks = dict(
        ball_payload=hashlib.sha256(payload).hexdigest()==r["payload_sha256"] and len(payload)==r["payload_bytes"] and len(tracks)==len(appearances)==n,
        source=r["source_sha256"] == SOURCE,
        coder=r["native"]["decoded_frames"] == n,
        sender=r["weight"] == fresh_ball_sender.MODEL_WEIGHT
        and r["code"] == fresh_ball_sender.MODEL_CODE,
        allocation=len(r["cpus"]) == 1,
        length=r["frames"] == len(r["positions"]) == n,
    )
    forecast = 1.5 * r["seconds"] * 240 / n + 80
    checks["fits"] = n != 24 or forecast <= 2300
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
    p.add_argument("--frames", type=int, choices=(24, 240))
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
