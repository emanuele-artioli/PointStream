"""Evaluate sealed cohort rate-bank decodes at the fixed official visible locations."""

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import cohort_ball_score as score
from experiments.tennis.k5d_encoder import crop64
from experiments.visor.h3b import dists_crops
from src.codecs import quality, svtav1
from src.pointstream import ball, ball_stream


def prepare(a, meta, full_prefix):
    """Produce each physical decoded crop once. Validate the selected pixels/bytes against primary receipts."""
    start = time.monotonic()
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "curve"
    work.mkdir()
    rates, _, selected = score.load_bank(a.rate, meta)
    frames, fps = score.source(a.source, meta, 240)
    assert (
        score.digest(frames) == full_prefix["source_rgb_sha256"]
        and full_prefix["rate_archive_sha256"] == a.rate_sha
    )
    labels = [v for v in score.marks(a.labels, a.labels_sha, 240) if v["visibility"] == 1]
    assert len(labels) == full_prefix["visible"] and len(labels) > 0
    if a.frames == 24:
        labels = [
            labels[i] for i in np.linspace(0, len(labels) - 1, min(6, len(labels)), dtype=int)
        ]
    fixed = time.monotonic() - start
    positions, looks = ball.decode_frames(selected.ball_payload, 240, 960, 544)
    velocity = ball.velocities(positions)
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == score.TOOLS["dav1d"]
    targets = np.array(
        [crop64(frames[r["tick"]], r["x"] / 2, r["y"] * 544 / 1080) for r in labels], np.uint8
    )
    bank = []
    limbs = []
    with tarfile.open(a.rate) as tar:
        for row in rates["rows"]:
            row_start = time.monotonic()
            plain = tar.extractfile("publish/" + row["native_file"]).read()
            combined = tar.extractfile("publish/" + row["ball_file"]).read()
            assert (
                score.digest(plain) == row["native_sha256"]
                and score.digest(combined) == row["ball_sha256"]
            )
            packet = ball_stream.unpack(combined)
            assert (
                ball_stream.pack(packet) == combined
                and packet.native == plain
                and packet.ball_payload == selected.ball_payload
            )
            assert (
                packet.fps == fps and packet.frames == 240 and len(svtav1.ivf_frames(plain)) == 240
            )
            raw = score.native(plain, row, work, decoder["path"])
            fixed += time.monotonic() - row_start
            both = [[], []]
            for label in labels:
                tick = label["tick"]
                xx, yy = label["x"] / 2, label["y"] * 544 / 1080
                ordinary = score.display(raw[tick], row["width"], row["height"])
                candidate = ball.render(ordinary, tick, positions, velocity, looks)
                both[0].append(crop64(ordinary, xx, yy))
                both[1].append(crop64(candidate, xx, yy))
            for arm, data, crops in zip(
                ("ordinary", "candidate"), (plain, combined), both, strict=True
            ):
                bank.append(
                    dict(
                        arm=arm,
                        width=row["width"],
                        height=row["height"],
                        crf=row["crf"],
                        bytes=len(data),
                        kbps=len(data) * 8 * float(fps) / 240000,
                        sha256=score.digest(data),
                    )
                )
                limbs.append(np.array(crops, np.uint8))
    return bank, limbs, targets, [r["tick"] for r in labels], fixed


def run(a):
    start = time.monotonic()
    n = a.frames
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    meta = json.loads(a.metadata.read_text())
    assert score.sha(a.source) == meta["source"]["sha256"] and score.sha(a.rate) == a.rate_sha
    primary = json.loads(a.score.read_text())
    assert (
        primary["frames"] == 240
        and primary["labels_sha256"] == a.labels_sha
        and primary["match"] == meta["match"]
    )
    bank, limbs, targets, ticks, fixed = prepare(a, meta, primary)
    assert score.sha(a.backbone) == score.BACKBONE
    assert len(ticks) == (min(6, primary["visible"]) if n == 24 else primary["visible"])
    net = quality.load_dists(a.backbone, "cpu")
    mask = np.ones((64, 64), bool)
    per_arm = [[] for _ in bank]
    for i, img in enumerate(targets):
        vals = dists_crops(net, [v[i] for v in limbs], img, mask, "cpu")
        assert all(math.isfinite(v) for v in vals)
        for v, vals_per_arm in zip(vals, per_arm, strict=True):
            vals_per_arm.append(v)
    for b, values in zip(bank, per_arm, strict=True):
        b["dists"] = float(np.mean(values)) if values else None
        b["per_tick_dists"] = values
    for arm in ("candidate", "ordinary"):
        match = next(b for b in bank if b["sha256"] == primary[arm + "_sha256"])
        wanted = [r for r in primary[arm]["rows"] if r["tick"] in set(ticks)]
        assert len(match["per_tick_dists"]) == len(wanted)
        for tick, value, reference in zip(ticks, match["per_tick_dists"], wanted, strict=True):
            assert tick == reference["tick"] and abs(value - reference["dists"]) < 2e-5
    result = dict(
        frames=n,
        match=meta["match"],
        fps=primary["fps"],
        source_sha256=primary["source_sha256"],
        source_rgb_sha256=primary["source_rgb_sha256"],
        labels_sha256=a.labels_sha,
        rate_archive_sha256=a.rate_sha,
        ticks=ticks,
        native_frames=240,
        full_visible=primary["visible"],
        fixed_seconds=fixed,
        bank=bank,
        score_model=quality.dists_record(a.backbone),
        elapsed=time.monotonic() - start,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "curve.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "curve.json").read_text())
    bank = r["bank"]
    checks = dict(
        arms={b["arm"] for b in bank} == {"ordinary", "candidate"},
        quality=all(len(b["per_tick_dists"]) == len(r["ticks"]) for b in bank),
        prices=all(
            abs(b["kbps"] - b["bytes"] * 8 * float(Fraction(r["fps"])) / 240000) < 1e-8
            for b in bank
        ),
    )
    if r["frames"] == 24:
        fixed = r["fixed_seconds"]
        forecast = 1.5 * (fixed + (r["elapsed"] - fixed) * r["full_visible"] / len(r["ticks"])) + 80
        checks["forecast"] = forecast < 2300
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks)) + "\n"
    )
    if not ok:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "metadata", "rate", "score", "labels", "backbone"):
        p.add_argument("--" + k, type=Path)
    for k in ("rate-sha", "labels-sha"):
        p.add_argument("--" + k)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)
