"""Time the paid Shanghai base and ball canvas, verifying the full compositor's native pixel digests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
sys.path.insert(0, str(CARD))
import ball_receiver_clock as clock  # noqa: E402


def validate(a, stage, dest):
    result = json.loads((stage / "clock.json").read_text())
    witness = json.loads((CARD / "match102-finish-summary.json").read_text())
    assert (
        result["frames"] == a.frames
        and result["opencv_threads"] == result["native_decoder_threads"] == 1
    )
    assert (
        result["joint_sha256"] == witness["wire"]["joint_sha256"]
        and result["cache_sha256"] == witness["wire"]["cache_sha256"]
    )
    assert result["paid_canvas_sha256"] == witness["base_frame_sha256"][: a.frames]
    assert (
        result["pixels_changed_ticks"] == result["sender_present"] and result["sender_present"] > 0
    )
    assert len(result["passes"]) == 3
    forecast = 1.5 * result["wall_s"] * 240 / a.frames + 15
    assert a.frames == 240 or forecast < 280
    dest.write_text(
        json.dumps(
            dict(
                passed=True,
                checks=dict(
                    real_paid_receiver=True,
                    frames=a.frames,
                    rgb_matches=a.frames,
                    estimate_s=forecast,
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
    assert (
        len(os.sched_getaffinity(0)) == 1
        and a.frames == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    )
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    paid = scratch / "paid"
    paid.mkdir()
    for key, fn in (("joint", "rally.psj"), ("cache", "initial.psr"), ("carrier", "players.pscp")):
        (paid / fn).symlink_to(getattr(a, key))
    model = next(
        r.split(":", 1)[1].strip()
        for r in Path("/proc/cpuinfo").read_text().splitlines()
        if r.startswith("model name")
    )
    clock.main(
        argparse.Namespace(
            frames=a.frames, paid=paid, scratch=scratch, out=stage / "clock.json", cpu_model=model
        )
    )
    if a.frames == 240:
        validate(a, stage, stage / "full_validation.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, choices=(24, 240), required=True)
    p.add_argument("--validate", action="store_true")
    for name in ("joint", "cache", "carrier"):
        p.add_argument("--" + name, type=Path, required=True)
    if sys.argv[1:] == ["--validate"]:
        path = Path(os.environ["PS_STAGE_DIR"]) / "dispatch.json"
        argv = json.loads(path.read_text())["command"]
        assert argv[1:3] == ["-m", "experiments.tennis.test_joint_ball_clock"]
        a = p.parse_args(argv[3:])
        a.validate = True
        main(a)
    else:
        main(p.parse_args())
