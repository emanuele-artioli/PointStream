"""Reconstruct the original Shanghai cleaned court and verify its identity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
from src.pointstream import ball

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
sys.path.insert(0, str(CARD))
import build_joint as build  # noqa: E402


def main(a):
    stage = Path(os.environ["PS_STAGE_DIR"])
    original = json.loads((CARD / "match102-sender.json").read_text())
    receipt = json.loads((CARD / "match102-result.json").read_text())
    if a.validate:
        r = json.loads((stage / "clean-parity.json").read_text())
        if r["frames"] == 3:
            assert r["donor"]["registrations"] == original["donor"]["smoke"]
        else:
            for key in (
                "smoke",
                "donors",
                "holes",
                "painted_area",
                "mask_sha256",
                "clean_yuv_sha256",
            ):
                assert r["donor"][key] == original["donor"][key], key
            assert build.sha(stage / "clean.yuv") == original["donor"]["clean_yuv_sha256"]
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(
            json.dumps(
                {
                    "passed": True,
                    "checks": {
                        "frames": r["frames"],
                        "registration_parity": True,
                        "clean_yuv_sha256": r["donor"].get("clean_yuv_sha256"),
                    },
                }
            )
        )
        return
    assert a.frames == {"smoke": 3, "full": 240}[os.environ["PS_STAGE"]]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(1)
    assert build.sha(a.source) == receipt["source_sha256"]
    assert build.sha(a.features) == receipt["prior_feature_archive_sha256"]
    a.out = Path(os.environ["PS_SCRATCH_DIR"]) / "shanghai-clean"
    a.out.mkdir()
    start = time.monotonic()
    payload = a.ball.read_bytes()
    positions, appearance = ball.decode_frames(payload, 240, 960, 544)
    assert ball.encode_frames(positions, appearance, 960, 544) == payload
    source = build.raster_source(a)
    masks = build.spatial_masks(a, receipt, positions)
    donor = build.donor_base(a, source, masks, smoke_only=a.frames == 3)
    record = {
        "frames": a.frames,
        "donor": donor,
        "opencv": cv2.__version__,
        "elapsed_s": time.monotonic() - start,
    }
    (stage / "clean-parity.json").write_text(json.dumps(record, indent=2) + "\n")
    if a.frames == 240:
        shutil.copyfile(a.out / "clean.yuv", stage / "clean.yuv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    p.add_argument("--frames", type=int)
    for name in ("source", "features", "ball"):
        p.add_argument("--" + name, type=Path)
    main(p.parse_args())
