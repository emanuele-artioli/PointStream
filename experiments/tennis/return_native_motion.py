"""Evaluate the verified native returning display under the established source-only motion metric."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiments.tennis.foreground_campus_score import arm
from experiments.tennis.player_requalification import sha256

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"


def read(p):
    return json.loads(p.read_text())


def body_supports(path, original, digest):
    import cv2

    record = path / "supports.json"
    assert sha256(record) == digest
    src = read(record)
    assert src["source_sha256"] == original and src["ticks"] == list(range(240))
    maps = {}
    for row in src["rows"]:
        t, role = row["tick"], row["role"]
        dest = path / row["mask"]
        assert sha256(dest) == row["sha256"]
        mask = cv2.imread(str(dest), cv2.IMREAD_GRAYSCALE) > 0
        x0, y0, x1, y1 = row["xyxy"]
        assert mask.shape == (y1 - y0, x1 - x0) and mask.sum() == row["area"] and row["area"] > 0
        assert (t, role) not in maps
        maps[t, role] = (row["xyxy"], mask)
    assert len(maps) == 480
    return [[maps[t, "near"], maps[t, "far"]] for t in range(240)], digest


def validate(root, report):
    from math import isfinite

    value = read(root / "motion.json")
    n = report["frames"]
    assert value["recorded_ticks"] == n
    rows = value["rows"]
    assert set(rows) == set(report["priced"])
    basis = None
    for name, points in rows.items():
        assert len(points) == 2 * (n - 1)
        fixed = [(r["tick"], r["role"], r["pixels"], r["source_delta"]) for r in points]
        if basis is None:
            basis = fixed
        else:
            assert basis == fixed
        assert all(isfinite(r["error"]) and r["pixels"] > 0 for r in points)
        assert isfinite(value["summary"][name]["error"]) and report["priced"][name] > 0
    assert value["source_grid_sha256"] == report["source_grid_sha256"]
    forecast = report["forecast_seconds"]
    checks = dict(
        all_transitions=True,
        source_grid=True,
        charged_inputs=True,
        identical_supports=True,
        near_far=True,
        cost=report["elapsed_seconds"] < 595 and (n == 240 or forecast < 595),
    )
    (root / "motion_validation.json").write_text(
        json.dumps(dict(passed=all(checks.values()), checks=checks)) + "\n"
    )
    if not all(checks.values()):
        raise ValueError(checks)


def run(a):
    sys.path.insert(0, str(CARD))
    import temporal_native as temporal

    start = time.monotonic()
    src = read(a.grid)
    assert a.frames == {"smoke": 8, "full": 240}[os.environ["PS_STAGE"]] and src["fps"] == 25
    assert (
        sha256(a.source) == src["source_sha256"] and sha256(a.canonical) == src["canonical_sha256"]
    )
    support = body_supports(a.supports, src["source_sha256"], src["supports_sha256"])
    info = read(a.contract)
    assert src["source_sha256"] == info["source_sha256"]
    ordinary = dict(
        name="SVT-AV1",
        kind="ivf_stream",
        width=640,
        height=384,
        path=str(a.ivf),
        **info["ordinary"],
    )
    models = [ordinary]
    neural = arm(
        argparse.Namespace(
            kind="neural",
            data=a.neural,
            label="DCVC-UF",
            producer=info["neural_job"],
            source_sha=src["source_sha256"],
            fps=25,
            qp=36,
            paid=info["neural_bytes"],
        )
    )
    assert (
        neural["bytes"] == info["neural_bytes"]
        and neural["sha256"] == info["neural_raw_sha256"]
        and neural["container_sha256"] == info["neural_stream_sha256"]
    )
    models.append(dict(neural, kind="yuv", decoded_sha256=neural["sha256"]))
    paid = arm(
        argparse.Namespace(
            kind="finish",
            data=a.witness,
            label="PointStream",
            producer=info["receiver_job"],
            paid=info["joint_bytes"],
            cache=info["cache_bytes"],
            expected_joint=info["joint_sha256"],
            fps=25,
        )
    )
    assert (
        paid["sha256"] == info["joint_raw_sha256"]
        and paid["receipt_sha256"] == info["finish_receipt_sha256"]
    )
    models.append(paid)
    scratch, stage = Path(os.environ["PS_SCRATCH_DIR"]), Path(os.environ["PS_STAGE_DIR"])
    inputs = scratch / "arms.json"
    inputs.write_text(json.dumps(models) + "\n")
    setup = time.monotonic() - start
    temporal.main(
        argparse.Namespace(
            source=a.source,
            source_pixels=a.canonical,
            features=None,
            support=support,
            fps=Fraction(25),
            arms=inputs,
            scratch=scratch,
            frames=a.frames,
            out=stage / "motion.json",
            source_hashes=src["source_rgb_hashes"],
        )
    )
    elapsed = time.monotonic() - start
    rec = dict(
        frames=a.frames,
        source_sha256=src["source_sha256"],
        state=info["state"],
        input_contract=sha256(a.contract),
        source_grid_sha256=hashlib.sha256(
            "".join(src["source_rgb_hashes"][: a.frames]).encode("ascii")
        ).hexdigest(),
        elapsed_seconds=elapsed,
        forecast_seconds=1.5 * (setup + (elapsed - setup) * 240 / a.frames) + 30,
        priced={r["name"]: r["bytes"] for r in models},
        tool=sha256(CARD / "temporal_native.py"),
    )
    (stage / "timing.json").write_text(json.dumps(rec, indent=2) + "\n")
    validate(stage, rec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "canonical", "grid", "supports", "ivf", "neural", "witness", "contract"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    if a.validate:
        out = Path(os.environ["PS_STAGE_DIR"])
        validate(out, read(out / "timing.json"))
        Path(os.environ["PS_VALIDATION_PATH"]).write_bytes(
            (out / "motion_validation.json").read_bytes()
        )
    else:
        run(a)
