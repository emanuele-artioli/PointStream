"""Measure one originally registered external 240-tick system or codec at its native cadence."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
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

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs/experiments/test-joint-extension-20261122/native"
sys.path.insert(0, str(ROOT / "docs/experiments/test-crop-joint-20261122"))
import score_native as native  # noqa: E402

sys.path.insert(0, str(CARD))
import samples  # noqa: E402


def select_arm(a, binding, bank):
    if a.arm.startswith("av1-"):
        filename = a.arm.removeprefix("av1-") + ".ivf"
        r = json.loads((bank / (a.event + "-rate.json")).read_text())
        assert r["source_sha256"] == binding["source_sha256"] and r["fps"] == binding["fps"]
        row = next(x for x in r["rows"] if x["native_file"] == filename)
        return dict(
            name=a.arm,
            kind="ivf_tar",
            width=row["width"],
            height=row["height"],
            bytes=row["bytes"],
            archive=str(a.data),
            member="publish/" + filename,
            sha256=row["native_sha256"],
        )
    if a.arm.startswith("dcvc-480-q"):
        qp = int(a.arm.removeprefix("dcvc-480-q"))
        assert 0 <= qp <= 63
        with tarfile.open(a.data) as z:
            n = json.load(z.extractfile("publish/result.json"))
            report = json.load(z.extractfile("publish/reports.json"))
        suffix = "-neural.json" if qp == 24 else f"-neural-q{qp}.json"
        card = json.loads((bank / (a.event + suffix)).read_text())
        assert (
            n["source_sha256"] == card["source_sha256"] == binding["source_sha256"]
            and n["frames"] == 240
        )
        assert Fraction(*n["native_fps"]) == Fraction(binding["fps"])
        rec = n["native"]
        assert (
            rec["accepted"]
            and rec["qp"] == qp
            and rec["size"] == [480, 272]
            and rec["container_sha256"] == card["native"]["container_sha256"]
        )
        passes = [p for group in report["decode"].values() for p in group["passes"]]
        assert len(passes) == 4 and all(
            p["yuv420_sha256"] == n["yuv_frames"] and p["frame_sha256"] == n["yuv_frames"]
            for p in passes
        )
        return dict(
            name=a.arm,
            kind="yuv_tar",
            width=480,
            height=272,
            bytes=rec["bytes"],
            archive=str(a.data),
            member="publish/decoded.yuv",
            frame_sha256=n["yuv_frames"],
            stream_sha256=rec["container_sha256"],
            decoded_sha256=n["raw_sha256"],
        )
    assert a.arm.startswith("joint-crf") and a.sender is not None
    sender = json.loads(a.sender.read_text())
    crf = int(a.arm.removeprefix("joint-crf"))
    row = next(r for r in sender["rates"] if r["crf"] == crf)
    assert (
        sender["source_sha256"] == binding["source_sha256"]
        and sender["native_fps"] == binding["fps"]
    )
    if a.receipt is None:
        receipt = a.data / "publish/finish.json"
        path = a.data / "publish/frames.npy"
        rr = json.loads(receipt.read_text())
        assert (
            rr["ticks"] == list(range(240))
            and rr["yuv_sha256"] == row["decoded_sha256"]
            and rr["kernels"]
        )
        assert rr["isolation"]["interfaces"] == ["lo"] and rr["isolation"]["mutation_denied"]
        kind = "rgb_npy"
        fingerprint = rr["frames_file_sha256"]
    else:
        receipt = a.receipt
        path = a.data
        rr = json.loads(receipt.read_text())
        assert rr["ticks"] == list(range(240)) and len(rr["rows"]) == 240
        kind = "rgb_zseq"
        fingerprint = rr["frames_sha256"]
    assert (
        rr["wire"]["joint_sha256"] == row["joint_sha256"]
        and rr["wire"]["cache_sha256"] == row["init_sha256"]
    )
    return dict(
        name=a.arm,
        kind=kind,
        width=1920,
        height=1080,
        bytes=row["joint_bytes"],
        path=str(path),
        sha256=fingerprint,
        receipt=str(receipt),
        receipt_sha256=sha256(receipt),
    )


def validate(a, stage, dest):
    score = json.loads((stage / "score.json").read_text())
    assert score["frames"] == a.frames and score["source_self_abs"] < 1e-5
    assert len(score["rows"]) == 2 * a.frames and len(score["source_grid_sha256"]) == 64
    assert {(x["tick"], x["role"]) for x in score["rows"]} == {
        (tick, role) for tick in range(a.frames) for role in ("near", "far")
    }
    assert all(np.isfinite(x["body_dists"]) and x["area"] > 0 for x in score["rows"])
    assert abs(score["pooled"]["dists"] - np.mean([x["body_dists"] for x in score["rows"]])) < 1e-8
    assert score["name"] == a.arm and score["event"] == a.event and score["producer"] == a.producer
    assert abs(score["kbps"] - score["bytes"] * float(Fraction(score["native_fps"])) / 30000) < 1e-8
    if score["samples_sha256"] is not None:
        assert sha256(stage / "samples.tar") == score["samples_sha256"]
    assert score["total_s"] < a.full_cap
    forecast = (
        1.5 * ((score["total_s"] - score["score_s"]) + score["score_s"] * 240 / a.frames) + 60
    )
    assert a.frames == 240 or forecast < a.full_cap
    dest.write_text(
        json.dumps(
            dict(
                passed=True,
                checks=dict(
                    frames=a.frames,
                    roles=2,
                    forecast_s=forecast,
                    complete=True,
                    source_grid_sha256=score["source_grid_sha256"],
                    score_sha256=sha256(stage / "score.json"),
                ),
            )
        )
        + "\n"
    )


def main(a):
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        validate(a, stage, Path(os.environ["PS_VALIDATION_PATH"]))
        return
    before = time.monotonic()
    assert a.frames == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert len(os.sched_getaffinity(0)) == 1 and (
        not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    assert cv2.getNumThreads() == 1 and sha256(a.weights) == native.WEIGHT
    b = json.loads((CARD / (a.event + "-binding.json")).read_text())
    assert b["source_sha256"] == sha256(a.source) and b["source_features_sha256"] == sha256(
        a.features
    )
    receipts = CARD / (a.event + "-crop-receipt.json")
    assert sha256(receipts) == b["crop_receipt_sha256"]
    arm = select_arm(a, b, CARD.parents[1] / "cohort-expansion-20261122/test-physical")
    supp, sig = native.get_support(
        a.features, receipt=receipts, tracks=[b["roles"][v] for v in ("near", "far")]
    )
    cfgfile = CARD / (a.event + "-review-ticks.json")
    cfg = json.loads(cfgfile.read_text()) if cfgfile.is_file() else None
    samples_sha = None
    if arm["kind"] in ("rgb_npy", "rgb_zseq") and cfg is not None:
        samples_sha = samples.export(a, arm, native, b, cfg, stage / "samples.tar")
    model = quality.load_dists(a.weights, "cpu")
    opts = argparse.Namespace(
        scratch=Path(os.environ["PS_SCRATCH_DIR"]),
        source=a.source,
        frames=a.frames,
        fps=Fraction(b["fps"]),
        full_cap=a.full_cap,
    )
    if cfg is not None:
        opts.review_ticks = set(cfg["ticks"])
    start = time.monotonic()
    result = native.score(opts, arm, model, supp, sig)
    result.update(
        event=a.event,
        producer=a.producer,
        samples_sha256=samples_sha,
        score_s=time.monotonic() - start,
        source_binding_sha256=sha256(CARD / (a.event + "-binding.json")),
        total_s=time.monotonic() - before,
    )
    (stage / "score.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    if a.frames == 240:
        validate(a, stage, stage / "full_validation.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", choices=("match127", "match105", "match116"), required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--frames", type=int, choices=(24, 240), required=True)
    p.add_argument("--full-cap", type=int, default=1600)
    p.add_argument("--validate", action="store_true")
    for k in ("source", "features", "weights", "data", "sender", "receipt"):
        p.add_argument("--" + k, type=Path, required=k not in ("sender", "receipt"))
    if sys.argv[1:] == ["--validate"]:
        path = Path(os.environ["PS_STAGE_DIR"]) / "dispatch.json"
        argv = json.loads(path.read_text())["command"]
        assert argv[1:3] == ["-m", "experiments.tennis.test_extension_native_score"]
        args = p.parse_args(argv[3:])
        args.validate = True
        main(args)
    else:
        main(p.parse_args())
