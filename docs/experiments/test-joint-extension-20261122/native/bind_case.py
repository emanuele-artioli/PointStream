"""Bind the rank-frozen, source-qualified extension to the existing full Shanghai sender."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

from experiments.tennis import common_player_packet as packet

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parents[1] / "test-crop-joint-20261122"
sys.path.insert(0, str(SANDBOX))
import build_joint  # noqa: E402


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def case(a):
    cfg = json.loads(a.binding.read_text())
    crop = json.loads(a.crop.read_text())
    assert cfg["event"] in ("match127", "match105", "match116") and cfg["event"] == crop["event"]
    rates = {
        "match127": (Fraction(60), packet.RECIPE_60),
        "match105": (Fraction(25), packet.RECIPE_25),
        "match116": (Fraction(25), packet.RECIPE_25),
    }
    fps, recipe = rates[cfg["event"]]
    official = json.loads(
        (
            HERE.parents[1]
            / "cohort-expansion-20261122/test-physical"
            / (cfg["event"] + "-neural.json")
        ).read_text()
    )
    assert (
        official["source_sha256"] == cfg["source_sha256"]
        and Fraction(*official["native_fps"]) == fps
    )
    assert (
        Fraction(cfg["fps"]) == fps and cfg["native_width"] == 1920 and cfg["native_height"] == 1080
    )
    assert cfg["source_sha256"] == crop["source_sha256"] == digest(a.source)
    assert (
        cfg["source_features_sha256"] == crop["prior_feature_archive_sha256"] == digest(a.features)
    )
    assert cfg["pose_archive_sha256"] == crop["output_archive_sha256"] == digest(a.poses)
    assert cfg["crop_receipt_sha256"] == digest(a.crop) and cfg[
        "source_role_receipt_sha256"
    ] == digest(a.roles)
    role_receipt = json.loads(a.roles.read_text())
    assert (
        role_receipt["source_sha256"] == cfg["source_sha256"]
        and role_receipt["features_sha256"] == cfg["source_features_sha256"]
    )
    assert set(cfg["roles"]) == {"far", "near"}
    assert set(cfg["roles"].values()) == set(crop["selected_ids"]) == set(role_receipt["selected"])
    assert crop["frames"] == 240 and len(cfg["prior_ranks_failed"]) == (
        "match127",
        "match105",
        "match116",
    ).index(cfg["event"])
    assert crop["fitted_both"] is True
    for role, track in cfg["roles"].items():
        assert cfg["paired_source_coverage"][role] == 240 and crop["tracks"][track]["fitted"] == 240
    actual_ball = json.loads(a.ball_receipt.read_text())
    assert actual_ball["source_sha256"] == crop["source_sha256"] and actual_ball["fps"] == str(fps)
    assert (
        actual_ball["payload_sha256"] == cfg["ball_sha256"] == digest(a.ball)
        and actual_ball["frames"] == 240
    )
    return dict(
        event=cfg["event"],
        fps=fps,
        recipe=recipe,
        receipt=a.crop,
        roles=cfg["roles"],
        units={k: cfg["event"] + "_" + k for k in ("far", "near")},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "out",
        "features",
        "source",
        "poses",
        "ball",
        "ball-receipt",
        "binding",
        "crop",
        "roles",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    build_joint.main(a, case(a))
