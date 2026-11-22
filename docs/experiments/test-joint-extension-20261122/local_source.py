"""Checkpoint the native court fit on every original tick with the prior homography intact."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from experiments.tennis import court, k8b

HERE = Path(__file__).resolve().parent


def run(a):
    cv2.setNumThreads(1)
    record = next(
        r
        for r in json.loads((HERE / "courts-spec.json").read_text())["staged_inputs"]
        if r["name"] == a.event
    )
    assert hashlib.sha256(a.source.read_bytes()).hexdigest() == record["sha256"]
    meta = k8b.probe(a.source, "ffprobe")
    assert (meta["width"], meta["height"]) == (1920, 1080) and meta["frames"] >= 240
    a.out.mkdir(exist_ok=False)
    begin = time.monotonic()
    prev = None
    rows = []
    hits = 0
    with closing(k8b.frames(k8b.decode_command("ffmpeg", a.source, 0, 240), 240)) as stream:
        for t, rgb in enumerate(stream):
            fit = court.register_v2(rgb, prev)
            prev = np.array(fit["homography"]) if fit["homography"] is not None else None
            rows.append(dict(frame=t, **fit))
            hits += bool(fit["fits"])
            with (a.out / "ticks.jsonl").open("a") as f:
                print(json.dumps(rows[-1]), file=f)
            elapsed = time.monotonic() - begin
            if t == 11:
                forecast = elapsed * 240 / 12 * 1.5 + 60
                if forecast > 1500:
                    # Repeatable <=45 min checkpoints; source fitting never skips the hard frames.
                    print(a.event, "slow full-shot forecast", round(forecast), flush=True)
            if len(rows) - hits > 48 or elapsed > 1470:
                break
    elapsed = time.monotonic() - begin
    eligible = (
        True if len(rows) == 240 and hits >= 192 else False if len(rows) - hits > 48 else None
    )
    result = dict(
        event=a.event,
        source_sha256=record["sha256"],
        native=meta,
        rows=rows,
        fit_count=hits,
        frames=len(rows),
        court_eligible=eligible,
        seconds=elapsed,
        cv2=cv2.__version__,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selector_sha256=hashlib.sha256(Path(court.__file__).read_bytes()).hexdigest(),
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    )
    (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(a.event, len(rows), hits, eligible, round(elapsed), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("event")
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    run(p.parse_args())
