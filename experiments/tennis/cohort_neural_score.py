"""Score a sealed four-pass neural receiver at its physical price with official TEST marks."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from experiments.tennis import cohort_ball_score as cohort
from experiments.tennis import shanghai_ball_display_freeze as frozen
from src.codecs import quality


def run(a):
    begin = time.monotonic()
    n = a.frames
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    meta = json.loads(a.metadata.read_text())
    assert frozen.sha(a.neural) == a.neural_sha and frozen.sha(a.source) == meta["source"]["sha256"]
    assert frozen.sha(a.backbone) == cohort.BACKBONE
    report = json.loads(frozen.member(a.neural, "result.json"))
    point = report["native"]
    assert report["frames"] == 240 and report["source_sha256"] == meta["source"]["sha256"]
    assert (
        point["accepted"]
        and point["identical_frame_hashes"]
        and point["size"] == [480, 272]
        and point["qp"] == 24
    )
    packet = frozen.member(a.neural, "stream.psdc")
    assert len(packet) == point["bytes"] and frozen.digest(packet) == point["container_sha256"]
    data = frozen.member(a.neural, "decoded.yuv")
    w, h = point["size"]
    assert len(data) == 240 * w * h * 3 // 2 and frozen.digest(data) == report["raw_sha256"]
    raw = np.frombuffer(data, np.uint8).reshape(240, h * 3 // 2, w)
    assert [hashlib.sha256(frame).hexdigest() for frame in raw] == report["yuv_frames"]
    frames, fps = cohort.source(a.source, meta, n)
    assert (
        fps == Fraction(*report["native_fps"])
        and abs(point["kbps"] - len(packet) * 8 * float(fps) / 240000) < 1e-8
    )
    receipt = json.loads((a.score if n == 240 else a.smoke_score).read_text())
    assert (
        receipt["source_rgb_sha256"] == cohort.digest(frames) and receipt["match"] == meta["match"]
    )
    assert receipt["labels_sha256"] == a.labels_sha and receipt["frames"] == n
    labels = cohort.marks(a.labels, a.labels_sha, n)
    net = quality.load_dists(a.backbone, "cpu")
    result = cohort.measure_yuv(raw, w, h, frames, labels, net)
    assert [r["tick"] for r in result["rows"]] == [r["tick"] for r in receipt["candidate"]["rows"]]
    result.update(
        match=meta["match"],
        frames=n,
        qp=24,
        size=point["size"],
        fps=str(fps),
        bytes=len(packet),
        kbps=point["kbps"],
        source_sha256=report["source_sha256"],
        source_rgb_sha256=receipt["source_rgb_sha256"],
        packet_sha256=point["container_sha256"],
        neural_archive_sha256=a.neural_sha,
        labels_sha256=a.labels_sha,
        neural_raw_sha256=report["raw_sha256"],
        visible=len(result["rows"]),
        dists_model=quality.dists_record(a.backbone),
        elapsed=time.monotonic() - begin,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "neural-score.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "neural-score.json").read_text())
    checks = dict(
        whole=r["whole_rgb_psnr"] > 0,
        price=abs(r["kbps"] - r["bytes"] * 8 * float(Fraction(r["fps"])) / 240000) < 1e-8,
        cohort=(r["ball_dists"] is not None) == bool(r["visible"])
        and r["visible"] == len(r["rows"]),
        verified=all(
            len(r[key]) == 64
            for key in (
                "packet_sha256",
                "neural_archive_sha256",
                "source_rgb_sha256",
                "rgb_sha256",
                "labels_sha256",
            )
        ),
    )
    if r["frames"] == 24:
        checks["forecast"] = 1.5 * r["elapsed"] * 10 + 70 < 1500
    passed = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=passed, checks=checks)) + "\n"
    )
    if not passed:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "metadata", "score", "smoke-score", "labels", "neural", "backbone"):
        p.add_argument("--" + key, type=Path)
    for key in ("neural-sha", "labels-sha"):
        p.add_argument("--" + key)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
