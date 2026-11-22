"""Checkpoint exact half-stream production inference using the same 240-frame median and real temporal context."""

import argparse, hashlib, json, os, time
from pathlib import Path
import cv2
import torch
from experiments.tennis import al028_ball_chunked as chunks
from experiments.tennis import cohort_ball_sender as sender
from experiments.tennis import fresh_ball_sender


def run(a):
    began = time.monotonic()
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert a.frames == {"smoke": 12, "full": 120}[os.environ["PS_STAGE"]] and a.first in (0, 120)
    assert (
        sender.sha(a.archive) == fresh_ball_sender.MODEL_CODE
        and sender.sha(a.weights) == fresh_ball_sender.MODEL_WEIGHT
    )
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    root = Path(os.environ["PS_SCRATCH_DIR"]) / "pointstream-part"
    root.mkdir()
    tracker = sender.model(a, root)
    meta = json.loads(a.metadata.read_text())
    frames, fps = sender.source(a.source, meta, 240)
    context = chunks.prepare_context(frames)
    fixed = time.monotonic() - began
    start = time.monotonic()
    raw = chunks.predict(tracker, context, a.first, a.first + a.frames)
    infer = time.monotonic() - start
    if a.frames == 12:
        assert raw == chunks.predict(tracker, context, a.first, a.first + 6) + chunks.predict(
            tracker, context, a.first + 6, a.first + 12
        )
    assert len(raw) == a.frames
    r = dict(
        match=meta["match"],
        frames=a.frames,
        start=a.first,
        stop=a.first + a.frames,
        fps=str(fps),
        source_sha256=meta["source"]["sha256"],
        source_rgb_sha256=hashlib.sha256(frames).hexdigest(),
        model_sha256=sender.sha(a.weights),
        code_sha256=sender.sha(a.archive),
        median_pick=[int(x) for x in context[2]],
        raw=raw,
        fixed=fixed,
        infer=infer,
        elapsed=time.monotonic() - began,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "part.json").write_text(
        json.dumps(r, indent=2, allow_nan=False) + "\n"
    )


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "part.json").read_text())
    checks = dict(
        range=r["start"] in (0, 120) and r["stop"] - r["start"] == r["frames"],
        coverage=len(r["raw"]) == r["frames"],
        model=r["model_sha256"] == fresh_ball_sender.MODEL_WEIGHT,
        context=r["median_pick"][0] == 0 and r["median_pick"][-1] == 239,
        captured=len(r["source_rgb_sha256"]) == 64,
    )
    forecast = 1.5 * (r["fixed"] + 120 / r["frames"] * r["infer"]) + 60
    if r["frames"] == 12:
        checks["forecast"] = forecast < 2400
    passed = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=passed, checks=checks, forecast=forecast)) + "\n"
    )
    if not passed:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("metadata", "source", "archive", "weights"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--first", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)
