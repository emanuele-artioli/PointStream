"""Apply the source-qualified RGB ball detector to displays fixed by the independent scoring pass."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from experiments.tennis import cohort_ball_score as cohort
from experiments.tennis import k5d_source_oracle as oracle
from experiments.tennis import shanghai_ball_frozen_score as prior
from experiments.tennis.k5b_monotrack_cpu import CpuZoo
from src.codecs import svtav1
from src.pointstream import ball


def detect(net, frames, labels, work, name):
    n = len(frames)
    prepared = np.lib.format.open_memmap(
        work / (name + "-detector.npy"), mode="w+", dtype=np.uint8, shape=(n, 720, 1280, 3)
    )
    for t, img in enumerate(frames):
        prepared[t] = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    start = time.monotonic()
    prediction = net.run("wasb-step1", prepared)
    assert len(prediction) == n
    rows = []
    for r in labels:
        present, x, y = prediction[r["tick"]]
        error = (
            math.hypot(1.5 * x - r["x"], 1.5 * y - r["y"])
            if present and r["visibility"] == 1
            else None
        )
        rows.append(
            dict(
                tick=r["tick"],
                visibility=r["visibility"],
                present=bool(present),
                error_native_pixels=error,
            )
        )
    return dict(rows=rows, seconds=time.monotonic() - start)


def run(a):
    start = time.monotonic()
    n = a.frames
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    meta = json.loads(a.metadata.read_text())
    assert cohort.sha(a.source) == meta["source"]["sha256"] and cohort.sha(a.rate) == a.rate_sha
    receipt = json.loads((a.score if n == 240 else a.smoke_score).read_text())
    assert (
        receipt["rate_archive_sha256"] == a.rate_sha
        and receipt["source_sha256"] == meta["source"]["sha256"]
    )
    assert receipt["frames"] == n and receipt["labels_sha256"] == a.labels_sha
    rates, streams, paid = cohort.load_bank(a.rate, meta)
    frames, _ = cohort.source(a.source, meta, n)
    assert cohort.digest(frames) == receipt["source_rgb_sha256"]
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "judge"
    work.mkdir()
    pos, looks = ball.decode_frames(paid.ball_payload, 240, 960, 544)
    velocity = ball.velocities(pos)
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == cohort.TOOLS["dav1d"]
    display = {"source": frames}
    for name in ("ordinary", "candidate"):
        r = rates[name]
        native = paid.native if name == "candidate" else streams[r["native_sha256"]]
        raw = cohort.native(native, r, work, decoder["path"])
        out = np.lib.format.open_memmap(
            work / (name + ".npy"), mode="w+", dtype=np.uint8, shape=frames.shape
        )
        for t in range(n):
            img = cohort.display(raw[t], r["width"], r["height"])
            out[t] = ball.render(img, t, pos, velocity, looks) if name == "candidate" else img
        assert hashlib.sha256(out).hexdigest() == receipt[name]["rgb_sha256"]
        display[name] = out
    labels = cohort.marks(a.labels, a.labels_sha, n)
    oracle.bind_config(a.wasb_code, prior.CONFIG)
    assert cohort.sha(a.wasb_weights / "wasb_tennis_best.pth.tar") == prior.WEIGHT
    net = CpuZoo(a.wasb_code, a.wasb_weights, names=("wasb-step1",))
    config = net.settings["wasb-step1"]["cfg"]
    assert next(net.settings["wasb-step1"]["net"].parameters()).device.type == "cpu"
    assert (
        config["model"]["frames_in"],
        config["model"]["frames_out"],
        config["detector"]["step"],
        config["detector"]["postprocessor"]["score_threshold"],
    ) == (3, 3, 1, 0.5)
    detections = {"source": detect(net, display["source"], labels, work, "source")}
    visible = sum(r["visibility"] == 1 for r in labels)
    correct = sum(
        r["error_native_pixels"] is not None and r["error_native_pixels"] <= 10
        for r in detections["source"]["rows"]
    )
    qualified = visible > 0 and correct / visible >= 0.8
    if n == 24 or qualified:
        for name in ("ordinary", "candidate"):
            detections[name] = detect(net, display[name], labels, work, name)
    result = dict(
        match=meta["match"],
        frames=n,
        visible=visible,
        source_hits10=correct,
        source_qualified=qualified,
        detections=detections,
        source_sha256=receipt["source_sha256"],
        labels_sha256=a.labels_sha,
        rate_archive_sha256=a.rate_sha,
        judge_checkpoint_sha256=prior.WEIGHT,
        rgb_sha256={name: hashlib.sha256(arr).hexdigest() for name, arr in display.items()},
        judge_config_sha256=prior.CONFIG,
        elapsed=time.monotonic() - start,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "judge.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "judge.json").read_text())
    outputs = r["detections"]
    expected = (
        {"source", "ordinary", "candidate"}
        if r["frames"] == 24 or r["source_qualified"]
        else {"source"}
    )
    checks = dict(
        outputs=set(outputs) == expected,
        cohort=all(
            sum(x["visibility"] == 1 for x in a["rows"]) == r["visible"] for a in outputs.values()
        ),
        source=r["source_qualified"]
        == (r["visible"] > 0 and r["source_hits10"] / r["visible"] >= 0.8),
        hashes=len(set(r["rgb_sha256"])) == 3
        and all(len(h) == 64 for h in r["rgb_sha256"].values()),
    )
    if r["frames"] == 24:
        checks["forecast"] = 1.5 * r["elapsed"] * 10 + 80 < 2300
    passed = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=passed, checks=checks)) + "\n"
    )
    if not passed:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "source",
        "metadata",
        "rate",
        "score",
        "smoke-score",
        "labels",
        "wasb-code",
        "wasb-weights",
    ):
        p.add_argument("--" + key, type=Path)
    for key in ("rate-sha", "labels-sha"):
        p.add_argument("--" + key)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
