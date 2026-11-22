"""Measure native source-flow fidelity for an external comparator already scored spatially."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import time

from experiments.tennis import test_extension_native_score as spatial
from experiments.tennis.player_requalification import sha256


def validate(a, stage, out):
    r = json.loads((stage / "motion.json").read_text())
    t = json.loads((stage / "timing.json").read_text())
    rows = r["rows"][a.arm]
    (arm,) = r["arms"]
    assert arm["name"] == a.arm and arm["bytes"] > 0
    assert r["recorded_ticks"] == a.frames and r["summary"][a.arm]["n"] == len(rows)
    assert 0 < len(rows) <= 2 * (a.frames - 1)
    assert len({(v["role"], v["tick"]) for v in rows}) == len(rows)
    assert all(0 < v["tick"] < a.frames and v["pixels"] > 0 for v in rows)
    assert len(r["source_grid_sha256"]) == len(r["source_mask_rle_sha256"]) == 64
    assert t["wall_s"] < a.full_cap and r["cpu_seconds"] <= t["wall_s"] * 1.1
    forecast = 1.5 * (t["wall_s"] - r["cpu_seconds"] + r["cpu_seconds"] * 240 / a.frames) + 25
    assert a.frames == 240 or forecast < a.full_cap
    out.write_text(
        json.dumps(
            dict(
                passed=True,
                checks=dict(
                    frames=a.frames,
                    transitions=len(rows),
                    source_grid_sha256=r["source_grid_sha256"],
                    forecast_s=forecast,
                    motion_sha256=sha256(stage / "motion.json"),
                ),
            )
        )
        + "\n"
    )


def run(a):
    import temporal_native as temporal

    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        validate(a, stage, Path(os.environ["PS_VALIDATION_PATH"]))
        return
    start = time.monotonic()
    assert len(os.sched_getaffinity(0)) == 1
    assert a.frames == {"smoke": 12, "full": 240}[os.environ["PS_STAGE"]]
    a.scratch = Path(os.environ["PS_SCRATCH_DIR"])
    b = json.loads((spatial.CARD / (a.event + "-binding.json")).read_text())
    assert (
        sha256(a.features) == b["source_features_sha256"] and sha256(a.source) == b["source_sha256"]
    )
    rec = spatial.CARD / (a.event + "-crop-receipt.json")
    assert sha256(rec) == b["crop_receipt_sha256"]
    a.support = spatial.native.get_support(
        a.features, receipt=rec, tracks=[b["roles"][k] for k in ("near", "far")]
    )
    bank = spatial.CARD.parents[1] / "cohort-expansion-20261122/test-physical"
    arm = spatial.select_arm(a, b, bank)
    a.arms = a.scratch / "arm.json"
    a.arms.write_text(json.dumps([arm]) + "\n")
    a.fps = Fraction(b["fps"])
    a.out = stage / "motion.json"
    temporal.main(a)
    (stage / "timing.json").write_text(json.dumps(dict(wall_s=time.monotonic() - start)) + "\n")
    if a.frames == 240:
        validate(a, stage, stage / "full_validation.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("arm", "event", "producer", "report-sha256"):
        p.add_argument("--" + k, required=True)
    for k in ("data", "source", "features"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--frames", type=int, choices=(12, 240), required=True)
    p.add_argument("--full-cap", type=int, default=700)
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        cmd = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "dispatch.json").read_text())[
            "command"
        ]
        assert cmd[1:3] == ["-m", "experiments.tennis.test_extension_motion_control"]
        run(p.parse_args(["--validate", *cmd[3:]]))
    else:
        run(p.parse_args())
