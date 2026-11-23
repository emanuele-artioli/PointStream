"""Reproduce and retain the authenticated Sydney clean sender raster on one CPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import cv2

from experiments.tennis.test_extension_joint_sender import CARD, bind
from src.codecs import svtav1
from src.pointstream import ball

REFERENCE = CARD / "match127-sender-result.json"
FRAME_SIZE = 960 * 544 * 3 // 2
FRAMES = 240
DONOR_FIELDS = ("smoke", "donors", "holes", "painted_area", "mask_sha256", "clean_yuv_sha256")


def validate(stage: Path, original: dict) -> dict:
    record = json.loads((stage / "clean-parity.json").read_text())
    assert record["source_sha256"] == original["source_sha256"]
    assert record["source_opencv"] == original["source_opencv"]
    assert record["ffmpeg"]["sha256"] == original["tools"]["ffmpeg"]["sha256"]
    donor = record["donor"]
    if record["frames"] == 3:
        assert donor["registrations"] == original["donor"]["smoke"]
        assert donor["forecast_seconds"] < 300
    else:
        assert record["frames"] == FRAMES
        assert all(donor[k] == original["donor"][k] for k in DONOR_FIELDS)
        clean = stage / "clean.yuv"
        assert clean.stat().st_size == FRAME_SIZE * FRAMES
        assert bind.digest(clean) == original["donor"]["clean_yuv_sha256"]
    return {
        "passed": True,
        "checks": {
            "source_sha256": record["source_sha256"],
            "frames": record["frames"],
            "clean_yuv_sha256": donor.get("clean_yuv_sha256"),
            "registration_parity": True,
        },
    }


def main(args):
    stage = Path(os.environ["PS_STAGE_DIR"])
    original = json.loads(REFERENCE.read_text())
    if args.validate:
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(
            json.dumps(validate(stage, original), indent=2) + "\n"
        )
        return
    started = time.monotonic()
    assert args.frames == {"smoke": 3, "full": FRAMES}[os.environ["PS_STAGE"]]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(1)
    assert cv2.__version__ == original["source_opencv"]
    ffmpeg = svtav1.tool("ffmpeg")
    assert ffmpeg["sha256"] == original["tools"]["ffmpeg"]["sha256"]
    for attribute, name in (
        ("binding", "match127-binding.json"),
        ("crop", "match127-crop-receipt.json"),
    ):
        setattr(args, attribute, CARD / name)
    args.roles = CARD.parent / "match127-source-roles.json"
    args.ball_receipt = (
        CARD.parents[1] / "cohort-expansion-20261122/test-physical/match127-sender-retry.json"
    )
    config = bind.case(args)
    args.out = Path(os.environ["PS_SCRATCH_DIR"]) / "clean-parity"
    args.out.mkdir()
    names, codes, references, crop = bind.build_joint.signal(args, config)
    payload = args.ball.read_bytes()
    positions, appearance = ball.decode_frames(payload, FRAMES, 960, 544)
    assert ball.encode_frames(positions, appearance, 960, 544) == payload
    assert len(names) == len(references) == 2 and codes
    source = bind.build_joint.raster_source(args)
    masks = bind.build_joint.spatial_masks(args, crop, positions)
    donor = bind.build_joint.donor_base(args, source, masks, smoke_only=args.frames == 3)
    if args.frames == FRAMES:
        assert all(donor[k] == original["donor"][k] for k in DONOR_FIELDS)
        shutil.copyfile(args.out / "clean.yuv", stage / "clean.yuv")
    record = {
        "source_sha256": bind.digest(args.source),
        "feature_sha256": bind.digest(args.features),
        "pose_sha256": bind.digest(args.poses),
        "trajectory_sha256": bind.digest(args.ball),
        "frames": args.frames,
        "source_opencv": cv2.__version__,
        "ffmpeg": ffmpeg,
        "donor": donor,
        "wall_seconds": time.monotonic() - started,
    }
    (stage / "clean-parity.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.frames == FRAMES:
        (stage / "full-validation.json").write_text(
            json.dumps(validate(stage, original), indent=2) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--frames", type=int, choices=(3, FRAMES))
    for key in ("source", "features", "poses", "ball"):
        parser.add_argument("--" + key, type=Path)
    main(parser.parse_args())
