"""Build one external paid native stream on campus, retaining its full raw rasters only in scratch."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2

from experiments.tennis.shanghai_ball_rate_crf import TOOLS
from src.codecs import svtav1
from src.pointstream import ball

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs/experiments/test-joint-extension-20261122/native"
sys.path.insert(0, str(CARD))
import bind_case as bind  # noqa: E402


def check(stage, receipt):
    assert receipt["fitted"] == [240, 240] and receipt["source_verified"]
    if receipt["steps"] == 3:
        assert [r[0] for r in receipt["smoke"]["registrations"]] == [0, 120, 239]
        assert min(r[2] for r in receipt["smoke"]["registrations"]) >= 50
        assert receipt["smoke"]["forecast_seconds"] < 900
    else:
        assert len(receipt["rungs"]) == 3 and len({r["crf"] for r in receipt["rungs"]}) == 3
        for row in receipt["rungs"]:
            for fn, digest in row["published"].items():
                assert bind.digest(stage / fn) == digest
            assert row["covered_ticks"] == 240
    return dict(
        passed=True,
        checks=dict(
            real_native_source=True,
            checked_role_tracks=True,
            source_crops=2,
            distinct_native_registrations=3,
            output_steps=receipt["steps"],
        ),
    )


def main(a):
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        r = json.loads((stage / "sender.json").read_text())
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(check(stage, r)) + "\n")
        return
    assert a.steps == {"smoke": 3, "full": 240}[os.environ["PS_STAGE"]]
    start = time.monotonic()
    assert len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(0)
    assert cv2.getNumThreads() == 1
    tools = {name: svtav1.tool(name) for name in TOOLS}
    assert {k: r["sha256"] for k, r in tools.items()} == TOOLS
    if a.event is not None:
        a.binding = CARD / (a.event + "-binding.json")
        a.crop = CARD / (a.event + "-crop-receipt.json")
        a.roles = CARD.parent / (a.event + "-source-roles.json")
        a.ball_receipt = (
            CARD.parents[1]
            / "cohort-expansion-20261122/test-physical"
            / (a.event + "-sender-retry.json")
        )
    cfg = bind.case(a)
    a.out = Path(os.environ["PS_SCRATCH_DIR"]) / "sender"
    receipt = dict(
        steps=a.steps,
        source_verified=True,
        source=bind.digest(a.source),
        fps=str(cfg["fps"]),
        event=cfg["event"],
        fitted=[240, 240],
        tools=tools,
    )
    if a.steps == 3:
        a.out.mkdir()
        names, poses, refs, crop = bind.build_joint.signal(a, cfg)
        raw = a.ball.read_bytes()
        pos, app = ball.decode_frames(raw, 240, 960, 544)
        assert (
            raw == ball.encode_frames(pos, app, 960, 544) and len(names) == len(refs) == 2 and poses
        )
        source = bind.build_joint.raster_source(a)
        masks = bind.build_joint.spatial_masks(a, crop, pos)
        receipt["smoke"] = bind.build_joint.donor_base(a, source, masks, smoke_only=True)
    else:
        bind.build_joint.main(a, cfg)
        original = json.loads((a.out / "result.json").read_text())
        receipt["selected_crf"] = original["selected_crf"]
        receipt["rungs"] = []
        for r in original["rates"]:
            published = {}
            for name, key in (
                ("rally.psj", "joint_sha256"),
                ("initial.psr", "init_sha256"),
                ("players.pscp", "carrier_sha256"),
                ("players.manifest", "manifest_sha256"),
            ):
                dest = stage / f"crf{r['crf']}-{name}"
                shutil.copyfile(a.out / f"crf{r['crf']}" / name, dest)
                published[dest.name] = bind.digest(dest)
                assert published[dest.name] == r[key]
            receipt["rungs"].append(dict(crf=r["crf"], covered_ticks=240, published=published))
        shutil.copyfile(a.out / "result.json", stage / "sender-result.json")
    receipt["wall_seconds"] = time.monotonic() - start
    assert receipt["wall_seconds"] < (500 if a.steps == 3 else 1600)
    (stage / "sender.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if a.steps == 240:
        (stage / "full-validation.json").write_text(json.dumps(check(stage, receipt)) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--event", choices=("match127", "match105", "match116"))
    for key in ("features", "source", "poses", "ball", "ball-receipt", "binding", "crop", "roles"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--steps", type=int, choices=(3, 240))
    main(p.parse_args())
