"""Score a sealed native TEST decode on campus, exporting only verified review samples and numbers."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis.player_requalification import sha256
from src.codecs import quality

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
sys.path.insert(0, str(CARD))
import harvest  # noqa: E402
import score_native as study  # noqa: E402


def record(args):
    stage = Path(os.environ["PS_STAGE_DIR"])
    crf = args.crf
    grid = json.loads((CARD / "match102-source-grid.json").read_text())
    assert sha256(args.source) == grid["source_sha256"] and sha256(args.weights) == study.WEIGHT
    canonical = json.loads((CARD / "match102-canonical-source.json").read_text())
    assert (
        sha256(args.source_pixels) == canonical["lossless_sha256"]
        and canonical["source_sha256"] == grid["source_sha256"]
    )
    assert args.frames == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    work = Path(os.environ["PS_SCRATCH_DIR"])
    start = time.monotonic()
    pub = work / "publish"
    pub.mkdir()
    if args.kind in ("calibration", "cold-neural"):
        with tarfile.open(args.data) as z:
            original = json.load(z.extractfile("publish/result.json"))
            reports = json.load(z.extractfile("publish/reports.json"))
        native = original["native"]
        expected_q, size = (34, [480, 272]) if args.kind == "calibration" else (42, [960, 544])
        assert native["qp"] == expected_q and native["accepted"] and native["size"] == size
        assert original["source_sha256"] == grid["source_sha256"] and original["frames"] == 240
        passes = [p for group in reports["decode"].values() for p in group["passes"]]
        assert len(passes) == 4 and all(
            p["yuv420_sha256"] == original["yuv_frames"]
            and p["frame_sha256"] == original["yuv_frames"]
            for p in passes
        )
        arm = dict(
            name=f"dcvc-{size[0]}-q{expected_q}",
            width=size[0],
            height=size[1],
            bytes=native["bytes"],
            kind="yuv_tar",
            archive=str(args.data),
            member="publish/decoded.yuv",
            frame_sha256=original["yuv_frames"],
            stream_sha256=native["container_sha256"],
            decoded_sha256=original["raw_sha256"],
        )
    else:
        if args.kind == "full":
            rec_path = args.data / "publish/finish.json"
            output = args.data / "publish/frames.npy"
        else:
            rec_path = args.receipt
            output = args.data
        rec = json.loads(rec_path.read_text())
        sender = json.loads((CARD / "match102-sender.json").read_text())
        row = next(r for r in sender["rates"] if r["crf"] == crf)
        assert (
            rec["wire"]["joint_sha256"] == row["joint_sha256"]
            and rec["wire"]["joint_bytes"] == row["joint_bytes"]
            and rec["wire"]["cache_sha256"] == row["init_sha256"]
        )
        sample = argparse.Namespace(
            receipt=rec_path,
            frames=output,
            source_manifest=CARD / "match102-source-review-frames.json",
            out=pub / "review",
            job=args.producer_job,
        )
        harvest.export(sample)
        arm = dict(
            name=f"joint-crf{crf}",
            kind="rgb_npy" if args.kind == "full" else "rgb_zseq",
            width=1920,
            height=1080,
            bytes=row["joint_bytes"],
            path=str(output),
            sha256=rec.get("frames_file_sha256", rec.get("frames_sha256")),
            receipt=str(rec_path),
            receipt_sha256=sha256(rec_path),
        )
    supp, source_mask = study.get_support(args.features)
    model = quality.load_dists(args.weights, "cpu")
    opts = argparse.Namespace(
        scratch=work,
        source=args.source,
        source_pixels=args.source_pixels,
        frames=args.frames,
        source_hashes=grid["native_rgb_typed_sha256"],
        full_cap=args.full_cap,
    )
    t0 = time.monotonic()
    result = study.score(opts, arm, model, supp, source_mask)
    score_s = time.monotonic() - t0
    assert result["frames"] == args.frames and len(result["rows"]) == args.frames * 2
    local = None
    if args.kind == "calibration" and args.frames == 240:
        local = 0.437958
        assert abs(result["pooled"]["dists"] - local) < 0.0002
    result["source_raw_rgb_crosschecked"] = args.frames
    result["producer_job"] = args.producer_job
    result["cross_machine_calibration"] = local
    result["job_kind"] = args.kind
    result["score_wall_s"] = score_s
    result["total_wall_s"] = time.monotonic() - start
    (pub / "score.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    with tarfile.open(stage / "published.tar", "x") as z:
        z.add(pub, arcname="publish")
    (stage / "timing.json").write_text(
        json.dumps(
            dict(
                archive_sha256=sha256(stage / "published.tar"),
                total_s=time.monotonic() - start,
                score_s=score_s,
            )
        )
        + "\n"
    )


def validate(args):
    stage = Path(os.environ["PS_STAGE_DIR"])
    timing = json.loads((stage / "timing.json").read_text())
    assert sha256(stage / "published.tar") == timing["archive_sha256"]
    with tarfile.open(stage / "published.tar") as z:
        result = json.load(z.extractfile("publish/score.json"))
        files = [x.name for x in z.getmembers() if x.isfile()]
        assert (
            result["frames"] == result["source_raw_rgb_crosschecked"] == args.frames
            and result["producer_job"] == args.producer_job
        )
        assert result["source_self_abs"] < 1e-5 and len(result["rows"]) == args.frames * 2
        valid = {(i, p) for i in range(args.frames) for p in ("near", "far")}
        assert {(r["tick"], r["role"]) for r in result["rows"]} == valid
        assert all(math.isfinite(r["body_dists"]) and r["area"] > 0 for r in result["rows"])
        scores = np.mean([r["body_dists"] for r in result["rows"]])
        assert abs(scores - result["pooled"]["dists"]) < 1e-8
        if args.kind in ("full", "rung"):
            packet = json.load(z.extractfile("publish/review/manifest.json"))
            assert len(files) == 14 and set(map(int, packet["ticks"])) == set(harvest.TICKS)
            assert (
                packet["finish_job"] == args.producer_job
                and packet["joint_rally_bytes"] == result["bytes"]
            )
        else:
            assert len(files) == 1 and (
                args.kind == "cold-neural"
                or args.frames < 240
                or abs(scores - result["cross_machine_calibration"]) < 0.0002
            )
    estimate = (
        1.5 * (timing["total_s"] - timing["score_s"] + timing["score_s"] * 240 / args.frames) + 60
    )
    assert (args.frames == 24 and estimate <= args.full_cap) or (
        args.frames == 240 and timing["total_s"] < args.full_cap
    )
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            dict(
                passed=True,
                checks=dict(
                    complete=True,
                    source_frames_verified=args.frames,
                    full_source_support=True,
                    producer=args.producer_job,
                    scored_person_ticks=args.frames * 2,
                    source_self=True,
                    review_samples=12 if args.kind in ("full", "rung") else 0,
                    forecast_s=estimate,
                ),
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", action="store_true")
    for name in ("source", "weights", "features", "data", "source-pixels"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--kind", choices=("calibration", "cold-neural", "full", "rung"), required=True)
    p.add_argument("--frames", type=int, choices=(24, 240), required=True)
    p.add_argument("--crf", type=int, choices=(50, 60, 70), default=50)
    p.add_argument("--producer-job", required=True)
    p.add_argument("--full-cap", type=int, default=1600)
    a = p.parse_args()
    validate(a) if a.validate else record(a)
