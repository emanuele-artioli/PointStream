"""Fixed full-context source-ball sender for a preregistered original test clip."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import al028_ball_chunked as chunks
from experiments.tennis import (
    fresh_ball_sender,
    k5b_trackers,
    k5b1b,
    k5c_independent_ball,
    k5d_encoder,
    windows,
)
from experiments.tennis.shanghai_ball_display_freeze import sha
from src.codecs import svtav1
from src.pointstream import ball


def source(path, meta, n):
    assert sha(path) == meta["source"]["sha256"]
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        assert cap.isOpened()
        video = meta["video"]
        fps = Fraction(video["avg_frame_rate"])
        assert abs(cap.get(cv2.CAP_PROP_FPS) - float(fps)) < 0.0001
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == int(video["nb_frames"])
        assert int(video["nb_frames"]) >= n and [video["width"], video["height"]] == [1920, 1080]
        frames = np.empty((n, 544, 960, 3), np.uint8)
        for i in range(n):
            good, frame = cap.read()
            assert good and frame.shape == (1080, 1920, 3)
            frames[i] = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (960, 544), interpolation=cv2.INTER_AREA
            )
        return frames, fps
    finally:
        cap.release()


def model(a, work):
    with tarfile.open(a.archive) as tar:
        tar.extractall(work / "model", filter="data")
    tracker = k5b_trackers.BallTrack(work / "model", a.weights, device="cpu")
    assert not tracker.net.training and next(tracker.net.parameters()).device.type == "cpu"

    def finite(_module, _inputs, out):
        if not torch.isfinite(out[0]).all():
            raise ValueError("nonfinite model output")

    tracker.net.register_forward_hook(finite)
    return tracker


def run(a):
    start = time.monotonic()
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert (
        sha(a.archive) == fresh_ball_sender.MODEL_CODE
        and sha(a.weights) == fresh_ball_sender.MODEL_WEIGHT
    )
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "sender"
    work.mkdir()
    meta = json.loads(a.metadata.read_text())
    n = a.frames
    assert n in (24, 240)
    prior = json.loads(a.verified_prefix.read_text()) if a.verified_prefix else None
    audit_seconds = 0.0
    part_shas = []
    if a.part_a is not None or a.part_b is not None:
        assert a.part_a and a.part_b and prior is None
        all_frames, fps = source(a.source, meta, 240)
        frames = all_frames[:n]
        full_sha = hashlib.sha256(all_frames).hexdigest()
        raw = []
        for path, first in ((a.part_a, 0), (a.part_b, 120)):
            part = json.loads(path.read_text())
            assert (
                part["start"] == first and part["stop"] == first + 120 and len(part["raw"]) == 120
            )
            assert (
                part["source_sha256"] == meta["source"]["sha256"]
                and part["source_rgb_sha256"] == full_sha
            )
            assert part["model_sha256"] == sha(a.weights) and part["code_sha256"] == sha(a.archive)
            assert part["match"] == meta["match"] and part["fps"] == str(fps)
            raw.extend(part["raw"])
            part_shas.append(sha(path))
        raw = raw[:n]
        setup_seconds = 0.0
    else:
        tracker = model(a, work)
        setup_seconds = time.monotonic() - start
        frames, fps = source(a.source, meta, n)
        context = chunks.prepare_context(frames)
        raw = chunks.predict(tracker, context, 0, n)
        if n == 24 and prior is None:
            audit_started = time.monotonic()
            assert raw == tracker.run(frames) and raw == chunks.predict(
                tracker, context, 0, 12
            ) + chunks.predict(tracker, context, 12, n)
            audit_seconds = time.monotonic() - audit_started
    predictions = [(v, x * 960 / 1280, y * 544 / 720) for v, x, y in raw]
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    convert = k5d_encoder.convert_rgb(tools["ffmpeg"]["path"], frames, work / "source.yuv", fps)
    native = work / "native"
    native.mkdir()
    rec = svtav1.code(
        work / "source.yuv",
        native,
        width=960,
        height=544,
        fps=fps,
        frames=n,
        crf=62,
        preset=4,
        threads=1,
        full_range=False,
        encoder=tools["SvtAv1EncApp"]["path"],
        decoder=tools["dav1d"]["path"],
        timeout=450,
    )
    decoded = windows.rgb_frames(Path(rec["decoded"]), 960, 544, n, work / "base.npy")
    (fit,) = k5c_independent_ball.fit_streak(
        k5b1b.track_of(predictions, (544, 960)), frames, [decoded], (544, 960)
    )
    pos, apps = fit["track"], fit["appearances"]
    guard = k5d_encoder.pooled_psnr(
        frames, k5c_independent_ball.render_frames(decoded, pos, apps)
    ) < k5d_encoder.pooled_psnr(frames, decoded)
    if guard:
        pos, apps = [None] * n, [None] * n
    payload = ball.encode_frames(pos, apps, 960, 544)
    assert ball.decode_frames(payload, n, 960, 544) == (pos, apps)
    stage = Path(os.environ["PS_STAGE_DIR"])
    if n == 24 and prior is not None:
        assert prior["frames"] == 24 and prior["source_sha256"] == meta["source"]["sha256"]
        assert prior["source_rgb_sha256"] == hashlib.sha256(frames).hexdigest()
        assert prior["payload_sha256"] == hashlib.sha256(payload).hexdigest()
        assert prior["model_sha256"] == sha(a.weights) and prior["code_sha256"] == sha(a.archive)
    (stage / "ball.payload").write_bytes(payload)
    result = dict(
        frames=n,
        source_sha256=meta["source"]["sha256"],
        fps=str(fps),
        model_sha256=sha(a.weights),
        code_sha256=sha(a.archive),
        source_rgb_sha256=hashlib.sha256(frames).hexdigest(),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=len(payload),
        present=sum(p is not None for p in pos),
        guard=bool(guard),
        tools=tools,
        conversion=convert,
        native=rec,
        seconds=time.monotonic() - start,
        equivalence_audit_seconds=audit_seconds,
        setup_seconds=setup_seconds,
        source_inference_parts_sha256=part_shas,
        prior_equivalence_receipt_sha256=sha(a.verified_prefix) if a.verified_prefix else None,
    )
    (stage / "sender.json").write_text(json.dumps(result, indent=2) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "sender.json").read_text())
    p = (stage / "ball.payload").read_bytes()
    pos, apps = ball.decode_frames(p, r["frames"], 960, 544)
    production_seconds = r["seconds"] - r.get("equivalence_audit_seconds", 0.0)
    assert 0 < production_seconds <= r["seconds"]
    fixed = r.get("setup_seconds", 0.0)
    assert 0 <= fixed < production_seconds
    forecast = 1.5 * (fixed + (production_seconds - fixed) * 240 / r["frames"]) + 90
    checks = dict(
        native=r["native"]["decoded_frames"] == r["frames"],
        payload=sha(stage / "ball.payload") == r["payload_sha256"]
        and ball.encode_frames(pos, apps, 960, 544) == p,
        model=r["model_sha256"] == fresh_ball_sender.MODEL_WEIGHT,
        source=bool(r["source_sha256"]),
        timing=r["frames"] != 24 or forecast <= 2450,
    )
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks, forecast=forecast))
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("metadata", "source", "archive", "weights"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--verified-prefix", type=Path)
    p.add_argument("--part-a", type=Path)
    p.add_argument("--part-b", type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
