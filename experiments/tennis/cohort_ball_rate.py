"""Construct physical ball and AV1 streams using the frozen cohort byte-only policy."""

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.tennis.shanghai_ball_rate_crf import TOOLS, sha
from src.codecs import svtav1
from src.pointstream import ball, ball_stream


def search(evaluate, field, target):
    lower, upper = 160, 280
    seen = []
    for step in range(10):
        probe = lower if step == 0 else upper if step == 1 else (lower + upper) // 2
        if probe in seen:
            break
        row = evaluate(probe)
        seen.append(probe)
        if step < 1:
            continue
        if evaluate(160)[field] <= target or evaluate(280)[field] >= target:
            break
        if step == 1:
            continue
        if row[field] > target:
            lower = probe
        elif row[field] < target:
            upper = probe
        else:
            break
    return seen


def pair_order(candidate, plain, target):
    c, p = candidate["ball_bytes"], plain["bytes"]
    gap = max(c, p) / min(c, p) - 1
    category = 0 if gap <= 0.005 else 1 if gap <= 0.02 else 2
    error = abs(c - target) / target
    return (
        category,
        gap if category == 2 else error,
        error if category == 2 else gap,
        -candidate["width"],
        c,
        p,
        candidate["crf"],
        plain["crf"],
    )


def packet(native, payload, fps, frames, width, height):
    stream = ball_stream.Stream(
        960, 544, fps, frames, "svt-av1-ivf", f"svtav1-4.2-scaled-{width}x{height}", native, payload
    )
    packed = ball_stream.pack(stream)
    assert ball_stream.unpack(packed) == stream
    return packed


def run(a):
    began = time.monotonic()
    stage = Path(os.environ["PS_STAGE_DIR"])
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "codec"
    work.mkdir()
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    pub.mkdir()
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    meta = json.loads(a.metadata.read_text())
    sender = json.loads(a.sender.read_text())
    fps = Fraction(meta["video"]["avg_frame_rate"])
    n = a.frames
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert (
        sha(a.source) == meta["source"]["sha256"] == sender["source_sha256"]
        and Fraction(sender["fps"]) == fps
    )
    assert sha(a.payload) == sender["payload_sha256"] and sender["frames"] == 240
    centers, looks = ball.decode_frames(a.payload.read_bytes(), 240, 960, 544)
    payload = ball.encode_frames(centers[:n], looks[:n], 960, 544)
    if n == 240:
        assert payload == a.payload.read_bytes()
    tools = {k: svtav1.tool(k) for k in TOOLS}
    assert all(v["sha256"] == TOOLS[k] for k, v in tools.items())
    target = round(Fraction(20400 * n, 8) / fps)
    rows = []
    searches = []
    for width, height in ((320, 192), (640, 384)):
        converted = work / f"{width}.yuv"
        command = [
            tools["ffmpeg"]["path"],
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(a.source),
            "-frames:v",
            str(n),
            "-vf",
            f"scale={width}:{height}:flags=area,format=yuv420p",
            "-f",
            "rawvideo",
            str(converted),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=150)
        assert converted.stat().st_size == n * width * height * 3 // 2
        cache = {}

        def evaluate(q):
            if q in cache:
                return cache[q]
            start = time.monotonic()
            nativefile = pub / f"{width}-{q}.ivf"
            subprocess.run(
                svtav1.encode_command(
                    tools["SvtAv1EncApp"]["path"],
                    converted,
                    nativefile,
                    width=width,
                    height=height,
                    fps=fps,
                    frames=n,
                    crf=q / 4,
                    preset=4,
                    full_range=False,
                ),
                check=True,
                capture_output=True,
                timeout=240,
            )
            data = nativefile.read_bytes()
            assert len(svtav1.ivf_frames(data)) == n
            packed = packet(data, payload, fps, n, width, height)
            ballfile = pub / f"{width}-{q}.psbl"
            ballfile.write_bytes(packed)
            row = dict(
                width=width,
                height=height,
                crf=q / 4,
                bytes=len(data),
                ball_bytes=len(packed),
                native_file=nativefile.name,
                ball_file=ballfile.name,
                native_sha256=sha(nativefile),
                ball_sha256=sha(ballfile),
                seconds=time.monotonic() - start,
            )
            cache[q] = row
            return row

        probes = search(evaluate, "ball_bytes", target)
        best = min(
            (cache[q] for q in probes),
            key=lambda r: (abs(r["ball_bytes"] - target), r["ball_bytes"], r["crf"]),
        )
        plain = search(evaluate, "bytes", best["ball_bytes"])
        rows.extend(cache.values())
        searches.append(
            dict(
                width=width,
                candidate_quarters=probes,
                ordinary_quarters=plain,
                ordinary_target_bytes=best["ball_bytes"],
            )
        )
    first, second = min(
        ((x, y) for x in rows for y in rows if x["width"] == y["width"]),
        key=lambda pair: pair_order(*pair, target),
    )
    gap = max(first["ball_bytes"], second["bytes"]) / min(first["ball_bytes"], second["bytes"]) - 1
    near = abs(first["ball_bytes"] - target) / target <= 0.005
    group = (
        ("target" if near else "other-matched")
        if gap <= 0.005
        else "sensitivity"
        if gap <= 0.02
        else "unmatched"
    )
    ceiling = max(first["ball_bytes"], second["bytes"])
    affordable = [r["native_sha256"] for r in rows if r["bytes"] <= ceiling]
    assert second["native_sha256"] in affordable
    receipt = dict(
        match=meta["match"],
        rally=meta["rally"],
        source_sha256=sha(a.source),
        sender_sha256=sha(a.sender),
        payload_sha256=sha(a.payload),
        frames=n,
        fps=str(fps),
        target_bytes=target,
        selection=group,
        matched_gap=gap,
        near_target=near,
        candidate=first,
        ordinary=second,
        affordable=affordable,
        probes=searches,
        rows=rows,
        tools=tools,
        elapsed=time.monotonic() - began,
    )
    for out in (stage, pub):
        (out / "rates.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def validate():
    report = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "rates.json").read_text())
    a, b = report["candidate"], report["ordinary"]
    rows = report["rows"]
    checks = dict(
        pair=a["width"] == b["width"] and a["ball_bytes"] > a["bytes"],
        anchors=b["native_sha256"] in report["affordable"],
        search=all(
            len(s[key]) <= 10
            for s in report["probes"]
            for key in ("candidate_quarters", "ordinary_quarters")
        ),
        unique=len({r["native_sha256"] for r in rows}) == len(rows),
    )
    if report["frames"] == 24:
        forecast = (
            sum(20 * max(r["seconds"] for r in rows if r["width"] == w) for w in (320, 640))
            * 10
            * 1.5
            + 90
        )
        checks["bounded"] = forecast <= 2300
    else:
        forecast = None
        checks["native_count"] = report["frames"] == 240
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks, forecast=forecast))
    )
    if not ok:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "source", "sender", "payload"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
