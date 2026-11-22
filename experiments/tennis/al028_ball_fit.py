"""Fit the frozen streak to two authenticated full-context sender chunks."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2

from experiments.tennis import al028_ball_sender as sender
from experiments.tennis import fresh_ball_sender, k5b1b, k5c_independent_ball, k5d_encoder, windows
from src.codecs import svtav1
from src.pointstream import ball


def run(a):
    t = time.monotonic()
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError("single-core allocation required")
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    if a.frames != n:
        raise ValueError("unexpected frame count")
    chunks = [json.loads(p.read_text()) for p in (a.chunk0, a.chunk1)]
    keys = (
        "source_sha256",
        "source_rgb_sha256",
        "source_frames",
        "fps",
        "context_median_sha256",
        "context_small_sha256",
        "context_picks",
        "model_code_sha256",
        "model_weight_sha256",
        "raw_prediction_coordinates",
    )
    if any(chunks[0][k] != chunks[1][k] for k in keys):
        raise ValueError("chunk source/context/model mismatch")
    if (
        chunks[0]["source_sha256"] != sender.SOURCE
        or chunks[0]["model_code_sha256"] != fresh_ball_sender.MODEL_CODE
        or chunks[0]["model_weight_sha256"] != fresh_ball_sender.MODEL_WEIGHT
        or [c["start"] for c in chunks] != [0, 120]
        or any(c["frames"] != 120 or len(c["predictions"]) != 120 for c in chunks)
    ):
        raise ValueError("unqualified chunks")
    preds = chunks[0]["predictions"] + chunks[1]["predictions"]
    preds = [(v, x * 960 / 1280, y * 544 / 720) for v, x, y in preds[:n]]
    cv2.setNumThreads(1)
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "al028-ball-fit"
    work.mkdir()
    source = sender.source_frames(a.source, n)
    if n == 240 and hashlib.sha256(source).hexdigest() != chunks[0]["source_rgb_sha256"]:
        raise ValueError("decoded source differs from tracker context")
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    conversion = k5d_encoder.convert_rgb(
        tools["ffmpeg"]["path"], source, work / "source.yuv", sender.FPS
    )
    native = work / "native"
    native.mkdir()
    rec = svtav1.code(
        work / "source.yuv",
        native,
        width=960,
        height=544,
        fps=sender.FPS,
        frames=n,
        crf=62,
        preset=4,
        threads=1,
        full_range=False,
        encoder=tools["SvtAv1EncApp"]["path"],
        decoder=tools["dav1d"]["path"],
        timeout=600,
    )
    decoded = windows.rgb_frames(Path(rec["decoded"]), 960, 544, n, work / "base.npy")
    initial = k5b1b.track_of(preds, (544, 960))
    (fit,) = k5c_independent_ball.fit_streak(initial, source, [decoded], (544, 960))
    positions, appearance = fit["track"], fit["appearances"]
    guard = k5d_encoder.pooled_psnr(
        source, k5c_independent_ball.render_frames(decoded, positions, appearance)
    ) < k5d_encoder.pooled_psnr(source, decoded)
    if guard:
        positions, appearance = [None] * n, [None] * n
    payload = ball.encode_frames(positions, appearance, 960, 544)
    if ball.decode_frames(payload, n, 960, 544) != (positions, appearance):
        raise ValueError("ball packet did not round trip")
    stage = Path(os.environ["PS_STAGE_DIR"])
    (stage / "ball.payload").write_bytes(payload)
    result = dict(
        frames=n,
        source_sha256=sender.SOURCE,
        source_rgb_sha256=chunks[0]["source_rgb_sha256"],
        chunk_sha256=[sender.prior.sha(p) for p in (a.chunk0, a.chunk1)],
        model_code_sha256=chunks[0]["model_code_sha256"],
        model_weight_sha256=chunks[0]["model_weight_sha256"],
        context_median_sha256=chunks[0]["context_median_sha256"],
        native=rec,
        tools=tools,
        convert=conversion,
        positions=positions,
        guard=guard,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=len(payload),
        seconds=time.monotonic() - t,
    )
    (stage / "sender.json").write_text(json.dumps(result, indent=2) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "sender.json").read_text())
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    payload = (stage / "ball.payload").read_bytes()
    decoded, apps = ball.decode_frames(payload, n, 960, 544)
    checks = dict(
        source=r["source_sha256"] == sender.SOURCE,
        count=r["frames"] == len(decoded) == len(apps) == n,
        payload=hashlib.sha256(payload).hexdigest() == r["payload_sha256"],
        model=r["model_weight_sha256"] == fresh_ball_sender.MODEL_WEIGHT,
        forecast=n != 24 or 1.5 * r["seconds"] * 10 + 80 < 1300,
    )
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            dict(
                passed=all(checks.values()),
                checks=checks,
                forecast_seconds=1.5 * r["seconds"] * 10 + 80,
            )
        )
        + "\n"
    )
    if not all(checks.values()):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    for name in ("source", "chunk0", "chunk1"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--frames", type=int, choices=(24, 240))
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)


if __name__ == "__main__":
    main()
