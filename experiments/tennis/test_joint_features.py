"""Run the production player selector on a prespecified held-out native window."""

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import tarfile
import tempfile

import numpy as np

from experiments.tennis import k4b_features as features
from experiments.tennis.k4b_pose import player_masks
from experiments.tennis.k6 import window_players
from experiments.visor.h1 import match_detections
from src.segmentation.masks import ClipMasks, decompress


def manifest(args, stage):
    clip = dict(
        id="test/" + args.event,
        frames=args.length,
        fps=float(Fraction(args.fps)),
        width=1920,
        height=1080,
        sha256=args.digest,
        review_frames=[120],
    )
    path = stage / "test-source-manifest.json"
    path.write_text(
        json.dumps(dict(schema=1, contains_human_marks=False, clips=[clip]), indent=2) + "\n"
    )
    return path


def court_record(args):
    with tarfile.open(args.court_archive) as archive:
        data = json.load(archive.extractfile("publish/result.json"))
    r = data["events"][args.event]
    assert (
        data["frames"] == r["frames"] == 240
        and r["source_sha256"] == args.digest
        and r["court_eligible"]
    )
    return {x["frame"]: np.asarray(x["homography"], np.float64) for x in r["rows"] if x["fits"]}


def audit(root, fits):
    feature = json.loads((root / "features.json").read_text())
    clip = ClipMasks.load(root / "sam-masks.rle")
    poses = json.loads(decompress((root / "pose.json.zst").read_bytes()))
    assert feature["frame_count"] == len(clip.frames) == len(poses) == 240
    ids = window_players(feature, clip, 0, fits)["players"]
    counts = {key: dict(mask=0, paired=0, missing=[]) for key in ids}
    for tick, frame in enumerate(clip.frames):
        masks = player_masks(frame, feature["aliases"], ids)
        boxes = np.asarray(poses[tick]["boxes"], np.float64).reshape(-1, 4)
        associated = match_detections(masks, boxes * 0.5) if len(boxes) else {}
        for key in ids:
            seen = key in masks and masks[key].any()
            valid = seen and key in associated
            counts[key]["mask"] += int(seen)
            counts[key]["paired"] += int(valid)
            if not valid:
                counts[key]["missing"].append(tick)
    return dict(
        ids=ids,
        tracks=counts,
        admitted=len(ids) == 2 and all(v["paired"] == 240 for v in counts.values()),
    )


def main(args):
    stage = Path(os.environ["PS_STAGE_DIR"])
    if args.limit != 1:
        raise ValueError("One native window is fixed for each event")
    if args.validate:
        summary = json.loads((stage / "k4b-features.json").read_text())
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temp:
            with tarfile.open(stage / "published.tar") as archive:
                archive.extractall(temp, filter="data")
            pub = Path(temp) / "publish"
            report = features.validate(pub, summary, stage / "test-source-manifest.json")
            record = json.loads((pub / "admission.json").read_text())
            report["checks"].update(
                source=record["source_sha256"] == args.digest,
                coverage=record["frames"] == 240 and len(record["tracks"]) <= 2,
                internal=all(
                    v["paired"] + len(v["missing"]) == 240 for v in record["tracks"].values()
                ),
                cost=os.environ["PS_STAGE"] != "smoke" or summary["seconds"] * 1.5 + 160 < 1200,
            )
            ok = all(report["checks"].values())
            report["passed"] = ok
            Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report) + "\n")
            if not ok:
                raise ValueError(report["checks"])
        return
    tag = "test/" + args.event + "@120"
    args.manifest = manifest(args, stage)
    args.windows = [tag]
    args.sources = [args.source]
    args.max_windows = 1
    args.ffmpeg = "ffmpeg"
    fits = court_record(args)
    if features.run(args):
        raise RuntimeError("source feature extraction failed")
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    r = audit(pub / tag.replace("/", "__"), fits)
    r.update(event=args.event, source_sha256=args.digest, phase=os.environ["PS_STAGE"], frames=240)
    (pub / "admission.json").write_text(json.dumps(r, indent=2) + "\n")
    (stage / "admission.json").write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "event", "digest", "fps", "court-archive", "sam-checkpoint", "pose"):
        p.add_argument("--" + k)
    p.add_argument("--length", type=int)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--validate", action="store_true")
    main(p.parse_args())
