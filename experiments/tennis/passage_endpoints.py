"""Native endpoint price/quality controls for the shared-gap broadcast comparison."""

import argparse
import hashlib
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import passage_native as passage
from src.codecs import svtav1

SHOTS = {"initial": (passage.gap.PRIOR_SHA, 281601), "return": (passage.gap.NEXT_SHA, 48480)}


def run(a):
    start = time.monotonic()
    digest, cap = SHOTS[a.shot]
    n = a.frames
    if len(os.sched_getaffinity(0)) != 1 or passage.gap.sha(a.source) != digest:
        raise ValueError("Invalid source or CPU allocation")
    cv2.setNumThreads(1)
    root = Path(os.environ["PS_SCRATCH_DIR"])
    pub = root / "publish"
    pub.mkdir()
    tools = {k: svtav1.tool(k) for k in passage.gap.TOOLS}
    if not all(tools[k]["sha256"] == h for k, h in passage.gap.TOOLS.items()):
        raise ValueError("Wrong native tools")
    samples = cap * n / 240
    trials, choices, commands = [], [], []
    for w, h in ((480, 272), (960, 544)):
        source = root / f"ref-{w}.yuv"
        commands.append(passage.convert(tools["ffmpeg"]["path"], a.source, source, (w, h), n))
        truth = np.memmap(root / "ref-480.yuv", np.uint8, "r", shape=(n, 408, 480))

        def probe(q):
            dest = root / f"{w}-q{q}"
            dest.mkdir()
            v = svtav1.code(
                source,
                dest,
                width=w,
                height=h,
                fps=Fraction(50),
                frames=n,
                crf=q,
                preset=4,
                threads=1,
                full_range=False,
                encoder=tools["SvtAv1EncApp"]["path"],
                decoder=tools["dav1d"]["path"],
                timeout=600,
            )
            if v["decoded_frames"] != n or v["temporal_units"] != n:
                raise ValueError("Incomplete control")
            return v

        points = {q: probe(q) for q in (28, 70)}
        if n == 240 and points[70]["file_bytes"] <= samples < points[28]["file_bytes"]:
            lo, hi = 28, 70
            for _ in range(6):
                q = round((lo + hi) * 2) / 4
                if q in points:
                    break
                points[q] = probe(q)
                if points[q]["file_bytes"] > samples:
                    lo = q
                else:
                    hi = q
        trials.extend(
            dict(
                width=w,
                crf=q,
                bytes=p["file_bytes"],
                seconds=p["encode_seconds"] + p["decode_seconds"],
            )
            for q, p in points.items()
        )
        low = [q for q, p in points.items() if p["file_bytes"] <= samples]
        high = [q for q, p in points.items() if p["file_bytes"] > samples]
        picks = ([max(low, key=lambda q: points[q]["file_bytes"])] if low else []) + (
            [min(high, key=lambda q: points[q]["file_bytes"])] if high else []
        )
        for q in picks:
            v = points[q]
            native = Path(v["stream"]).read_bytes()
            name = f"{w}-q{q}.ivf"
            (pub / name).write_bytes(native)
            frames = np.memmap(v["decoded"], np.uint8, "r", shape=(n, h * 3 // 2, w))
            sse = 0
            for ref, frame in zip(truth, frames, strict=True):
                y = (
                    frame[:h]
                    if w == 480
                    else cv2.resize(frame[:h], (480, 272), interpolation=cv2.INTER_AREA)
                )
                e = ref[:272].astype(np.int32) - y
                sse += int(np.square(e).sum(dtype=np.int64))
            choices.append(
                dict(
                    width=w,
                    crf=q,
                    bytes=len(native),
                    kbps=len(native) * 0.4 / n,
                    sse=sse,
                    luma_psnr=10 * math.log10(255**2 * 480 * 272 * n / sse),
                    file=name,
                    stream_sha256=hashlib.sha256(native).hexdigest(),
                    native=v,
                )
            )
    report = dict(
        shot=a.shot,
        source_sha256=digest,
        frames=n,
        cap=cap,
        tools=tools,
        commands=commands,
        trials=trials,
        choices=choices,
        seconds=time.monotonic() - start,
    )
    (pub / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def validate(a):
    import tarfile

    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar") as archive:
        r = json.load(archive.extractfile("publish/result.json"))
        tests = dict(
            source=r["source_sha256"] == SHOTS[a.shot][0],
            count=r["frames"] in (24, 240),
            widths={x["width"] for x in r["choices"]} == {480, 960},
        )
        for row in r["choices"]:
            data = archive.extractfile("publish/" + row["file"]).read()
            tests[row["file"]] = (
                hashlib.sha256(data).hexdigest() == row["stream_sha256"]
                and len(data) == row["bytes"]
                and len(svtav1.ivf_frames(data)) == r["frames"]
                and 0 < row["luma_psnr"] < 100
            )
    estimate = None
    if os.environ["PS_STAGE"] == "smoke":
        # Each full geometry has at most eight encodes. The smoke includes the extreme slow CRF.
        worst = sum(max(x["seconds"] for x in r["trials"] if x["width"] == w) for w in (480, 960))
        estimate = (
            1.45 * (8 * 10 * worst + 10 * (r["seconds"] - sum(x["seconds"] for x in r["trials"])))
            + 90
        )
        tests["fits"] = estimate < 1350
    result = dict(passed=all(tests.values()), checks=tests, estimate=estimate)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--shot", choices=SHOTS, required=True)
    parser.add_argument("--frames", type=int, choices=(24, 240))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate(args) if args.validate else run(args)
