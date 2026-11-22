"""Compare a variable-length paid ball reconstruction on original TrackNet labels."""

import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
from statistics import mean
import sys

import cv2
import numpy as np
import torch

from experiments.tennis import k5b_trackers, k5c_independent_ball, windows
from src.codecs import svtav1
from src.pointstream import ball, ball_stream

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "joint-metric-crosscheck-20261123"))
from score import ALEX_SHA256, EXPECTED, WEIGHTS, image_psnr, quality, region_scores, sha  # noqa: E402


def displays(native, size, fps, n, output, label):
    width, height = size
    bitstream = output / (label + ".ivf")
    bitstream.write_bytes(native)
    yuv = output / (label + ".yuv")
    svtav1.run(svtav1.decode_command("dav1d", bitstream, yuv, threads=1), timeout=120)
    assert len(svtav1.ivf_frames(native)) == n
    assert yuv.stat().st_size == n * width * height * 3 // 2
    probe = windows.rgb_frames(yuv, width, height, n, output / (label + ".npy"))
    return np.stack(
        [cv2.resize(image, (960, 544), interpolation=cv2.INTER_CUBIC) for image in probe]
    )


def source(path, count):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames = []
    try:
        for _ in range(count):
            ok, img = cap.read()
            assert ok and img.shape == (720, 1280, 3)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        assert not cap.read()[0]
    finally:
        cap.release()
    return frames


def run(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.out.mkdir(exist_ok=False)
    basis = json.loads((HERE / "tracknet-native.json").read_text())
    verified = next(r for r in basis["rallies"] if r["rally_id"] == args.source.stem)
    assert (
        sha(args.source) == verified["clip_sha256"] and sha(args.labels) == verified["label_sha256"]
    )
    sender = json.loads((args.sender / "result.json").read_text())
    anchor = json.loads((args.anchor / "result.json").read_text())
    n = sender["rendered"]
    assert sender["source_sha256"] == anchor["source_sha256"] == verified["clip_sha256"]
    assert n == anchor["num_frames"] == verified["native_range"][1]
    fps = Fraction(*basis["cadence"])
    prices = (args.ceiling,) if args.ceiling else (20400, 33000)
    for price in prices:
        cap = int(Fraction(price * n, 8) / fps)
        candidate = next(
            (
                p
                for p in sender["packet_arms"]
                if p["ceiling"] == cap and p["coded_size"] == [640, 360]
            ),
            None,
        )
        if candidate is not None:
            break
    if candidate is None:
        raise ValueError("no affordable 640-wide native ball packet")
    ordinary = next(a for a in anchor["arms"] if a["width"] == 640)
    recorded = {p["crf"]: p for p in ordinary["available"]}
    q_points = []

    def probe(q):
        if q in recorded:
            r = recorded[q]
            stream = args.anchor / r["file"]
            assert sha(stream) == r["sha256"]
            return stream
        stream = args.out / f"ordinary-q{q:g}.ivf"
        command = svtav1.encode_command(
            "SvtAv1EncApp",
            args.anchor / "input-640.yuv",
            stream,
            width=640,
            height=360,
            fps=fps,
            frames=n,
            crf=q,
            preset=4,
            full_range=False,
        )
        svtav1.run(command, timeout=120)
        recorded[q] = dict(file=str(stream), bytes=stream.stat().st_size, sha256=sha(stream))
        q_points.append((q, stream.stat().st_size, sha(stream)))
        return stream

    threshold = candidate["bytes"]
    above = [q for q in recorded if recorded[q]["bytes"] >= threshold]
    below = [q for q in recorded if recorded[q]["bytes"] <= threshold]
    low = max(above, default=min(recorded))
    high = min(below, default=max(recorded))
    for _ in range(7):
        q = math.floor((low + high) * 2) / 4
        if not low < q < high:
            break
        if probe(q).stat().st_size > threshold:
            low = q
        else:
            high = q
    possible = [(q, r) for q, r in recorded.items() if r["bytes"] <= cap]
    key, selected = min(possible, key=lambda pair: abs(pair[1]["bytes"] - threshold))
    control = (
        probe(key) if Path(selected["file"]).is_absolute() is False else Path(selected["file"])
    )
    stream_file = args.sender / f"640-{cap}.psbl"
    assert sha(stream_file) == candidate["sha256"]
    stream = ball_stream.unpack(stream_file.read_bytes())
    assert stream.frames == n and stream.fps == fps and (stream.width, stream.height) == (960, 544)
    points, looks = ball.decode_frames(stream.ball_payload, n, 960, 544)
    first = displays(control.read_bytes(), (640, 360), fps, n, args.out, "ordinary")
    second = displays(stream.native, (640, 360), fps, n, args.out, "base")
    joint = k5c_independent_ball.render_frames(second, points, looks)
    originals = source(args.source, n)
    with args.labels.open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == n and [r["file name"] for r in records] == [
        f"{i:04d}.jpg" for i in range(n)
    ]
    labels = k5b_trackers.tracknet_labels(
        dict(
            rows=[
                (
                    r["file name"],
                    int(r["visibility"]),
                    float(r["x-coordinate"] or 0),
                    float(r["y-coordinate"] or 0),
                )
                for r in records
            ]
        )
    )
    assert sha(WEIGHTS) == EXPECTED and sha(args.alex) == ALEX_SHA256
    dists = quality.load_dists(WEIGHTS, "cpu")
    lpips = quality.load_lpips(args.alex, "cpu")
    rows = []
    for i, (kind, x, y) in enumerate(labels):
        a, b = [
            cv2.resize(arm[i], (1280, 720), interpolation=cv2.INTER_CUBIC) for arm in (first, joint)
        ]
        orig = originals[i]
        point = points[i]
        error = (
            None
            if point is None or kind != 1
            else math.hypot(point[0] * 1280 / 960 - x, point[1] * 720 / 544 - y)
        )
        row = dict(
            tick=i,
            kind=kind,
            point=point,
            error_native_px=error,
            ordinary_psnr=image_psnr(orig, a),
            pointstream_psnr=image_psnr(orig, b),
        )
        if kind == 1:
            xi, yi = max(0, min(1280 - 64, int(x) - 32)), max(0, min(720 - 64, int(y) - 32))
            mask = np.ones((64, 64), dtype=bool)
            region = orig[yi : yi + 64, xi : xi + 64]
            row["crop"] = {
                name: region_scores(region, pred[yi : yi + 64, xi : xi + 64], mask, dists, lpips)
                for name, pred in [("ordinary", a), ("pointstream", b)]
            }
        rows.append(row)
    pooled = {
        name: {
            k: mean(row["crop"][name][k] for row in rows if "crop" in row)
            for k in ("dists", "lpips", "luma_ssim", "rgb_psnr")
        }
        for name in ("ordinary", "pointstream")
    }
    result = dict(
        source_sha256=verified["clip_sha256"],
        label_sha256=verified["label_sha256"],
        frames=n,
        fps=list(basis["cadence"]),
        rates_bytes=dict(ordinary=selected["bytes"], pointstream=threshold, ceiling=cap),
        width=640,
        ordinary_sha256=sha(control),
        pointstream_sha256=candidate["sha256"],
        calibrations=q_points,
        visible=sum(r["kind"] == 1 for r in rows),
        absent=sum(r["kind"] == 0 for r in rows),
        within_ten=sum(
            r["error_native_px"] is not None and r["error_native_px"] <= 10 for r in rows
        ),
        false_on_absent=sum(r["kind"] == 0 and r["point"] is not None for r in rows),
        mean_crops=pooled,
        whole_psnr={k: mean(r[f"{k}_psnr"] for r in rows) for k in pooled},
        rows=rows,
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k not in ("rows", "calibrations")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "labels", "sender", "anchor", "alex", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--ceiling", type=int, choices=(20400, 33000))
    run(p.parse_args())
