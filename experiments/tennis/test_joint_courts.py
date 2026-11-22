"""Source-only native-cadence court admission for a frozen official TEST subset."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from experiments.tennis import k8b
from experiments.visor.b1 import file_sha256


def run(args):
    start = time.monotonic()
    stage = os.environ["PS_STAGE"]
    out = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    out.mkdir()
    results = {}
    assert args.frames == {"smoke": 12, "full": 240}[stage]
    for source, event, expected, fps, length in zip(
        args.source, args.event, args.sha, args.fps, args.length, strict=True
    ):
        assert file_sha256(source) == expected
        meta = k8b.probe(source, "ffprobe")
        assert (meta["width"], meta["height"], meta["r_frame_rate"], meta["frames"]) == (
            1920,
            1080,
            fps,
            length,
        )
        rows = k8b.register_chunk(str(source), 0, args.frames, "ffmpeg")
        assert [r["frame"] for r in rows] == list(range(args.frames))
        ticks = list(np.linspace(0, args.frames - 1, 6, dtype=int))
        previews = []
        for t, rgb in enumerate(
            k8b.frames(k8b.decode_command("ffmpeg", source, 0, args.frames), args.frames)
        ):
            if t in ticks:
                previews.append(
                    cv2.resize(rgb, (448, 252), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                )
        assert len(previews) == 6
        image = np.vstack([np.hstack(previews[:3]), np.hstack(previews[3:])])
        assert cv2.imwrite(str(out / f"{event}-source.jpg"), image)
        count = sum(bool(r["fits"]) for r in rows)
        results[event] = dict(
            source_sha256=expected,
            frames=args.frames,
            native=meta,
            rows=rows,
            fit_count=count,
            court_eligible=count / args.frames >= 0.8,
            preview_ticks=[int(x) for x in ticks],
            preview_sha256=hashlib.sha256((out / f"{event}-source.jpg").read_bytes()).hexdigest(),
        )
    receipt = dict(
        stage=stage, frames=args.frames, events=results, seconds=time.monotonic() - start
    )
    data = json.dumps(receipt, allow_nan=False, indent=2) + "\n"
    (out / "result.json").write_text(data)
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(data)


def validate(expected_events, full_seconds=2300):
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "result.json").read_text())
    n = r["frames"]
    checks = dict(
        events=list(r["events"]) == expected_events,
        coverage=all(
            len(x["rows"]) == n and 0 <= x["fit_count"] <= n for x in r["events"].values()
        ),
        forecast=r["stage"] != "smoke" or r["seconds"] * 20 * 1.5 + 100 < full_seconds,
    )
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks)) + "\n"
    )
    if not ok:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name, typ in (
        ("source", Path),
        ("event", str),
        ("sha", str),
        ("fps", str),
        ("length", int),
    ):
        p.add_argument("--" + name, nargs="+", type=typ)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--full-seconds", type=int, default=2300)
    a = p.parse_args()
    validate(a.event or ["match108", "match102", "match120"], a.full_seconds) if a.validate else run(a)
