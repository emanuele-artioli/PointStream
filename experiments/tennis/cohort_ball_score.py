"""Evaluate sealed native cohort streams against sparse official source labels."""

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import k8, k8b
from experiments.tennis.cohort_ball_sender import source
from experiments.tennis.k5d_encoder import crop64
from experiments.tennis.shanghai_ball_frozen_score import BACKBONE
from experiments.tennis.shanghai_ball_rate_crf import TOOLS, sha
from experiments.visor.h3b import dists_crops
from src.codecs import quality, svtav1
from src.pointstream import ball, ball_stream, tracked_patch


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def load_bank(archive, meta):
    with tarfile.open(archive) as tar:

        def member(name):
            f = tar.extractfile("publish/" + name)
            if f is None:
                raise ValueError(f"missing packet member {name}")
            return f.read()

        report = json.loads(member("rates.json"))
        assert report["frames"] == 240 and report["source_sha256"] == meta["source"]["sha256"]
        assert report["match"] == meta["match"] and report["rally"] == meta["rally"]
        assert Fraction(report["fps"]) == Fraction(meta["video"]["avg_frame_rate"])
        assert report["tools"]["dav1d"]["sha256"] == TOOLS["dav1d"]
        streams = {}
        for r in report["rows"]:
            data = member(r["native_file"])
            assert (
                digest(data) == r["native_sha256"]
                and len(data) == r["bytes"]
                and len(svtav1.ivf_frames(data)) == 240
            )
            if r["native_sha256"] in report["affordable"]:
                streams[r["native_sha256"]] = data
        b = report["candidate"]
        data = member(b["ball_file"])
        assert digest(data) == b["ball_sha256"] and len(data) == b["ball_bytes"]
        paid = ball_stream.unpack(data)
        assert (
            ball_stream.pack(paid) == data
            and paid.frames == 240
            and paid.fps == Fraction(report["fps"])
        )
        assert (
            digest(paid.native) == b["native_sha256"]
            and digest(paid.ball_payload) == report["payload_sha256"]
        )
    return report, streams, paid


def marks(path, expected, n):
    assert sha(path) == expected
    rows = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as pairs:
        for row in csv.DictReader(pairs):
            t = int(row["Frame"])
            if not 0 <= t < n:
                continue
            if t in seen:
                raise ValueError("repeated official tick")
            seen.add(t)
            kind = int(row["Visibility"])
            x = y = None
            if kind == 1:
                x, y = int(row["X"]), int(row["Y"])
                if not (0 <= x < 1920 and 0 <= y < 1080):
                    raise ValueError("official visible annotation outside original frame")
            rows.append(dict(tick=t, visibility=kind, x=x, y=y))
    return sorted(rows, key=lambda r: r["tick"])


def native(data, row, folder, decoder):
    path = folder / (row["native_sha256"] + ".ivf")
    if not path.exists():
        path.write_bytes(data)
    assert sha(path) == row["native_sha256"]
    decoded = folder / (row["native_sha256"] + ".yuv")
    w, h = row["width"], row["height"]
    if not decoded.exists():
        subprocess.run(
            svtav1.decode_command(decoder, path, decoded, threads=1),
            check=True,
            capture_output=True,
            timeout=180,
        )
    assert decoded.stat().st_size == 240 * w * h * 3 // 2
    return np.memmap(decoded, np.uint8, "r", shape=(240, h * 3 // 2, w))


def display(raw, w, h):
    return cv2.resize(
        tracked_patch.yuv_to_rgb(raw, w, h, False), (960, 544), interpolation=cv2.INTER_CUBIC
    )


def measure(data, row, frames, labels, model, work, decoder, ctx, whole):
    video = native(data, row, work, decoder)
    visible = {r["tick"]: r for r in labels if r["visibility"] == 1}
    scores = []
    seen = hashlib.sha256()
    sse = 0
    for t in range(len(frames)) if whole else sorted(visible):
        rgb = display(video[t], row["width"], row["height"])
        if ctx is not None:
            rgb = ball.render(rgb, t, *ctx)
        if whole:
            seen.update(rgb)
            delta = rgb.astype(np.int16) - frames[t]
            sse += int(np.einsum("ijk,ijk->", delta, delta, dtype=np.int64))
        if t in visible:
            r = visible[t]
            x, y = r["x"] / 2, r["y"] * 544 / 1080
            target = crop64(frames[t], x, y)
            value = float(
                dists_crops(model, [crop64(rgb, x, y)], target, np.ones((64, 64), bool), "cpu")[0]
            )
            assert math.isfinite(value)
            scores.append(dict(tick=t, dists=value))
    return dict(
        rows=scores,
        ball_dists=mean(r["dists"] for r in scores) if scores else None,
        rgb_sha256=seen.hexdigest() if whole else None,
        whole_rgb_psnr=10 * math.log10(255**2 * frames.size / sse) if whole and sse else None,
    )


def run(a):
    began = time.monotonic()
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert a.frames == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    meta = json.loads(a.metadata.read_text())
    assert sha(a.rate) == a.rate_sha
    rates, streams, paid = load_bank(a.rate, meta)
    frames, fps = source(a.source, meta, a.frames)
    assert len(frames) == a.frames and fps == paid.fps
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "score"
    work.mkdir()
    ffmpeg = svtav1.tool("ffmpeg")
    assert ffmpeg["sha256"] == TOOLS["ffmpeg"]
    scenes, _ = k8.cuts(a.source, float(fps), a.frames, ffmpeg["path"], work)
    cuts = sorted(
        {
            s["frame"]
            for s in scenes
            if s["score"] >= k8b.DECISION["scene_threshold"] and 0 < s["frame"] < a.frames
        }
    )
    fits = k8b.register_chunk(str(a.source), 0, a.frames, ffmpeg["path"])
    assert [r["frame"] for r in fits] == list(range(a.frames))
    court = [
        dict(first=i, stop=j, **k8b.prediction(fits[i:j]))
        for i, j in k8b.clip_shots(fits, cuts, float(fps))
    ]
    # Stream choices and source-only shot diagnostics precede the official coordinate read.
    labels = marks(a.labels, a.labels_sha, a.frames)
    pos, apps = ball.decode_frames(paid.ball_payload, 240, 960, 544)
    assert ball.encode_frames(pos, apps, 960, 544) == paid.ball_payload
    velocities = ball.velocities(pos)
    direct = []
    for r in labels:
        t = r["tick"]
        error = None
        if r["visibility"] == 1 and pos[t] is not None:
            error = math.hypot(2 * pos[t][0] - r["x"], pos[t][1] * 1080 / 544 - r["y"])
        direct.append(
            dict(
                tick=t,
                visibility=r["visibility"],
                present=pos[t] is not None,
                error_native_pixels=error,
            )
        )
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == TOOLS["dav1d"] and sha(a.backbone) == BACKBONE
    net = quality.load_dists(a.backbone, "cpu")
    sample = crop64(frames[0], 480, 272)
    self_score = float(dists_crops(net, [sample], sample, np.ones((64, 64), bool), "cpu")[0])
    assert abs(self_score) < 1e-5
    b, o = rates["candidate"], rates["ordinary"]
    candidate = measure(
        paid.native, b, frames, labels, net, work, decoder["path"], (pos, velocities, apps), True
    )
    baseline = measure(
        streams[o["native_sha256"]], o, frames, labels, net, work, decoder["path"], None, True
    )
    ordinary = []
    for r in rates["rows"]:
        if r["native_sha256"] not in streams:
            continue
        vals = (
            baseline
            if r == o
            else measure(
                streams[r["native_sha256"]],
                r,
                frames,
                labels,
                net,
                work,
                decoder["path"],
                None,
                False,
            )
        )
        ordinary.append(
            dict(
                width=r["width"],
                crf=r["crf"],
                bytes=r["bytes"],
                kbps=r["bytes"] * 8 * float(fps) / 240000,
                sha256=r["native_sha256"],
                ball_dists=vals["ball_dists"],
                rows=vals["rows"],
            )
        )
    best = None
    if any(r["visibility"] == 1 for r in labels):
        best = min(ordinary, key=lambda r: (r["ball_dists"], r["bytes"], r["sha256"]))
    result = dict(
        match=meta["match"],
        rally=meta["rally"],
        source_sha256=meta["source"]["sha256"],
        source_rgb_sha256=digest(frames),
        court_segments=court,
        court_ticks=sum(r["stop"] - r["first"] for r in court if r["predicted"] == "court"),
        labels_sha256=a.labels_sha,
        rate_archive_sha256=a.rate_sha,
        fps=str(fps),
        frames=a.frames,
        selection=rates["selection"],
        gap_fraction=rates["matched_gap"],
        target_bytes=rates["target_bytes"],
        candidate_bytes=b["ball_bytes"],
        candidate_kbps=b["ball_bytes"] * 8 * float(fps) / 240000,
        candidate_sha256=b["ball_sha256"],
        candidate=candidate,
        ordinary_bytes=o["bytes"],
        ordinary_kbps=o["bytes"] * 8 * float(fps) / 240000,
        ordinary_sha256=o["native_sha256"],
        ordinary=baseline,
        all_affordable=ordinary,
        best_affordable=best,
        direct=direct,
        visible=sum(r["visibility"] == 1 for r in labels),
        marked_not_visible=sum(r["visibility"] == 0 for r in labels),
        unknown_codes=sum(r["visibility"] not in (0, 1) for r in labels),
        unlisted=a.frames - len(labels),
        self_score=self_score,
        score_model=quality.dists_record(a.backbone),
        elapsed=time.monotonic() - began,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "scores.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "scores.json").read_text())
    checks = dict(
        cohort=r["visible"] + r["marked_not_visible"] + r["unknown_codes"] + r["unlisted"]
        == r["frames"],
        rows=all(len(r[arm]["rows"]) == r["visible"] for arm in ("ordinary", "candidate")),
        anchors=bool(r["all_affordable"])
        and (r["best_affordable"] is not None) == bool(r["visible"]),
        whole=all(
            r[arm]["whole_rgb_psnr"] > 0 and r[arm]["rgb_sha256"]
            for arm in ("ordinary", "candidate")
        ),
        native=bool(r["candidate_sha256"] and r["ordinary_sha256"]),
    )
    forecast = 1.5 * r["elapsed"] * 240 / r["frames"] + 90
    if r["frames"] == 24:
        checks["forecast"] = forecast <= 2200
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks, forecast=forecast))
    )
    if not ok:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "metadata", "rate", "labels", "backbone"):
        p.add_argument("--" + name, type=Path)
    for name in ("rate-sha", "labels-sha"):
        p.add_argument("--" + name)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
