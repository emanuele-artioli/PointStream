"""Recover source-court decisions on a locally authenticated original video."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import cv2

from experiments.tennis import k8b

HERE = Path(__file__).resolve().parent


def main(args):
    spec = json.loads((HERE / (args.event + "-court.json")).read_text())
    sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    assert sha == spec["staged_inputs"][0]["sha256"]
    meta = k8b.probe(args.source, "ffprobe")
    assert (meta["width"], meta["height"]) == (1920, 1080) and meta["frames"] >= 240
    start = time.monotonic()
    smoke = k8b.register_chunk(str(args.source), 0, 12, "ffmpeg")
    elapsed = time.monotonic() - start
    expected = 12 if args.event == "alcaraz031" else 0
    assert sum(bool(r["fits"]) for r in smoke) == expected
    count = 49 if args.event == "rublev023" else 240
    forecast = elapsed * count / 12 * 1.5 + 100
    record = dict(
        event=args.event,
        source_sha256=sha,
        source=meta,
        cv2=cv2.__version__,
        platform=platform.platform(),
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        smoke_seconds=elapsed,
        smoke=smoke,
        full_forecast=forecast,
        proposed_count=count,
    )
    if forecast < 1200:
        start = time.monotonic()
        rows = k8b.register_chunk(str(args.source), 0, count, "ffmpeg")
        assert len(rows) == count and [bool(r["fits"]) for r in rows[:12]] == [
            bool(r["fits"]) for r in smoke
        ]
        fit = sum(bool(r["fits"]) for r in rows)
        upper = (fit + 240 - count) / 240
        record.update(
            full_seconds=time.monotonic() - start, rows=rows, fit_count=fit, upper_fraction=upper
        )
        if count == 240 or upper < 0.8:
            record["eligible"] = upper >= 0.8
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        args.event,
        {
            k: record[k]
            for k in ("smoke_seconds", "full_forecast", "fit_count", "eligible")
            if k in record
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("event", choices=["alcaraz031", "rublev023"])
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    main(p.parse_args())
