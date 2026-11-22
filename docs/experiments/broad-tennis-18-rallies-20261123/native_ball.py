"""Run the existing CPU tennis ball sender and native packet on a variable-length rally."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import al028_ball_chunked as chunks
from experiments.tennis import (
    fresh_ball_sender,
    k5b1b,
    k5b_trackers,
    k5c_independent_ball,
    k5d_encoder,
    windows,
)
from src.codecs import svtav1
from src.pointstream import ball, ball_stream

W, H = 960, 544


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source(path, total, fps):
    video = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames = []
    try:
        assert video.isOpened() and abs(video.get(cv2.CAP_PROP_FPS) - float(fps)) < 0.001
        for _ in range(total):
            ok, image = video.read()
            assert ok and image.shape == (720, 1280, 3)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA))
        assert not video.read()[0]
    finally:
        video.release()
    return np.stack(frames)


def run(args):
    began = time.monotonic()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert sha(args.source) == args.source_sha256
    assert sha(args.archive) == fresh_ball_sender.MODEL_CODE
    assert sha(args.weights) == fresh_ball_sender.MODEL_WEIGHT
    args.out.mkdir(exist_ok=False)
    with tarfile.open(args.archive) as model:
        model.extractall(args.out / "model", filter="data")
    tracker = k5b_trackers.BallTrack(args.out / "model", args.weights, device="cpu")
    assert not tracker.net.training and next(tracker.net.parameters()).device.type == "cpu"

    def finite(_layer, _inputs, output):
        assert torch.isfinite(output[0]).all()

    tracker.net.register_forward_hook(finite)
    fps = Fraction(args.fps)
    all_frames = source(args.source, args.total, fps)
    n = 24 if args.pilot else args.total
    frames = all_frames[:n]
    intervals = [{"start": 0, "stop": n, "predicted": "court"}]
    if args.route is not None:
        registered = json.loads(args.route.read_text())
        assert registered["source_sha256"] == args.source_sha256
        assert registered["native_fps"] == [fps.numerator, fps.denominator]
        intervals = registered["shots"]
        assert [i for s in intervals for i in range(s["start"], s["stop"])] == list(range(n))
    fallback = [
        i for s in intervals if s["predicted"] == "non_court" for i in range(s["start"], s["stop"])
    ]
    setup = time.monotonic() - began
    inference_start = time.monotonic()
    raw = [(False, 0.0, 0.0)] * n
    for shot in intervals:
        lo, hi = shot["start"], shot["stop"]
        if shot["predicted"] == "court":
            context = chunks.prepare_context(all_frames if args.pilot else frames[lo:hi])
            raw[lo:hi] = chunks.predict(tracker, context, 0, hi - lo)
        else:
            assert shot["predicted"] == "non_court"
    inference = time.monotonic() - inference_start
    if args.pilot:
        assert raw == chunks.predict(tracker, context, 0, 12) + chunks.predict(
            tracker, context, 12, n
        )
    assert len(raw) == n
    positions = [(valid, x * W / 1280, y * H / 720) for valid, x, y in raw]
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    work = args.out / "fit"
    work.mkdir()
    raw_path = args.out / "input.yuv"
    k5d_encoder.convert_rgb(tools["ffmpeg"]["path"], frames, raw_path, fps)
    encoded = svtav1.code(
        raw_path,
        work,
        width=W,
        height=H,
        fps=fps,
        frames=n,
        crf=62,
        preset=4,
        threads=1,
        full_range=False,
        encoder=tools["SvtAv1EncApp"]["path"],
        decoder=tools["dav1d"]["path"],
        timeout=120,
    )
    decoded = windows.rgb_frames(Path(encoded["decoded"]), W, H, n, work / "decoded.npy")
    (fit,) = k5c_independent_ball.fit_streak(
        k5b1b.track_of(positions, (H, W)), frames, [decoded], (H, W)
    )
    for tick in fallback:
        fit["track"][tick] = fit["appearances"][tick] = None
    rendered = k5c_independent_ball.render_frames(decoded, fit["track"], fit["appearances"])
    guard = k5d_encoder.pooled_psnr(frames, rendered) < k5d_encoder.pooled_psnr(frames, decoded)
    track, appearance = ([None] * n, [None] * n) if guard else (fit["track"], fit["appearances"])
    payload = ball.encode_frames(track, appearance, W, H)
    assert ball.decode_frames(payload, n, W, H) == (track, appearance)
    rates = []
    if args.pilot:
        inputs = [(Path(encoded["stream"]), W, H)]
        budget_grid = [None]
    else:
        baseline = json.loads((args.av1 / "result.json").read_text())
        assert baseline["source_sha256"] == args.source_sha256 and baseline["num_frames"] == n
        inputs = [
            (args.av1 / c["file"], arm["width"], arm["height"])
            for arm in baseline["arms"]
            for c in arm["available"]
        ]
        budget_grid = [int(Fraction(p * n, 8) / fps) for p in (20400, 33000)]
    choices = {}
    for path, width, height in inputs:
        native = path.read_bytes()
        stream = ball_stream.Stream(
            W, H, fps, n, "svt-av1-ivf", f"svtav1-local-scaled-{width}x{height}", native, payload
        )
        packed = ball_stream.pack(stream)
        assert ball_stream.unpack(packed) == stream
        for ceiling in budget_grid:
            if ceiling is None or len(packed) <= ceiling:
                key = (ceiling, width, height)
                if key not in choices or len(packed) > len(choices[key][0]):
                    choices[key] = (packed, stream, path)
    for (ceiling, width, height), (packed, stream, parent) in choices.items():
        stem = f"{width}-{ceiling}"
        result = args.out / (stem + ".psbl")
        result.write_bytes(packed)
        child = args.out / (stem + "-base.ivf")
        child.write_bytes(ball_stream.unpack(result.read_bytes()).native)
        yuv = args.out / (stem + "-decoded.yuv")
        svtav1.run(
            svtav1.decode_command(tools["dav1d"]["path"], child, yuv, threads=1), timeout=120
        )
        assert yuv.stat().st_size == n * width * height * 3 // 2 and sha(child) == sha(parent)
        yy = windows.rgb_frames(yuv, width, height, n, args.out / (stem + "-decoded.npy"))
        rgb = np.stack([cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC) for img in yy])
        composed = k5c_independent_ball.render_frames(rgb, track, appearance)
        assert composed.shape == (n, H, W, 3)
        assert all(np.array_equal(composed[tick], rgb[tick]) for tick in fallback)
        rates.append(
            dict(
                ceiling=ceiling,
                coded_size=[width, height],
                bytes=len(packed),
                base_bytes=len(stream.native),
                framing_bytes=len(packed) - len(stream.native) - len(payload),
                sha256=sha(result),
                base_sha256=sha(child),
                displayed_sha256=hashlib.sha256(composed.tobytes()).hexdigest(),
            )
        )
    receipt = dict(
        source_sha256=args.source_sha256,
        source_frames=args.total,
        rendered=n,
        fps=[fps.numerator, fps.denominator],
        source_context_sha256=hashlib.sha256(all_frames.tobytes()).hexdigest(),
        tracker="RacketVision BallTrack CPU",
        code_sha256=sha(args.archive),
        weights_sha256=sha(args.weights),
        device="cpu",
        torch=torch.__version__,
        tools=tools,
        raw_present=sum(p[0] for p in raw),
        guard=bool(guard),
        ball_positions=track,
        payload_bytes=len(payload),
        intervals=intervals,
        fallback_ticks=len(fallback),
        fallback_rgb_equals_native_base=True,
        packet_arms=rates,
        setup_wall_s=setup,
        inference_wall_s=inference,
        wall_s=time.monotonic() - began,
        worker_sha256=sha(Path(__file__)),
    )
    assert rates
    (args.out / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        n,
        receipt["raw_present"],
        sum(p is not None for p in track),
        len(payload),
        round(receipt["wall_s"], 1),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "archive", "weights", "out", "av1"):
        p.add_argument("--" + k, type=Path, required=k != "av1")
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--total", type=int, required=True)
    p.add_argument("--fps", default="25/1")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--route", type=Path)
    a = p.parse_args()
    if a.pilot and a.route is not None:
        p.error("select either the initial prefix smoke or one complete verified route")
    if not a.pilot and a.av1 is None:
        p.error("full pilot needs the existing baseline directory")
    run(a)
