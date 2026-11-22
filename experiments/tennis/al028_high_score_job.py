"""Score the separately paid high-rate Al028 base using the sealed receiver."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace

from experiments.tennis import common_player_packet as packet
from experiments.tennis import w8_payload as wire

FINISH = "ca9c335e804ec51003ca13e17c9895bb2c231ef79d30e16342d81050f0e679d0"
HIGH = "55bdedb1840e343a40acea72c80712a3e5165c15731e5ed039f20d57324e42f9"
CACHE = "91bd3a3fd9b18db301348c98e5af1536b859f2e51dde71e7de56c3beee1c1af2"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(a):
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    if a.frames != n:
        raise ValueError("wrong stage size")
    t = time.monotonic()
    high = a.extra / "ps-nov22-al028-matched-39771/joint-30.5"
    ordinary = a.extra / "ps-nov22-al028-matched-39771/ordinary-37.75"
    ordinary640 = a.extra / "ps-nov22-al028-640-bill-47725/q55"
    joint = high / "rally.psj"
    if (
        sha(joint) != HIGH
        or sha(high / "stream.ivf")
        != "85744d7d3c77badb06de7d528b38721725ad09c24928a20bb927766c0105ee4c"
        or sha(ordinary / "stream.ivf")
        != "bac24432bcec62bd5a491839c18c4b162dba369f84827ed055e2cbe01716abd4"
        or sha(ordinary640 / "stream.ivf")
        != "c8d5dea39e2e63579c33b691acb6506064916394d5644a2ef6d7acd4cb6b2aab"
    ):
        raise ValueError("paid high-rate stream identity")
    blob = joint.read_bytes()
    profile = packet._embedded_profile(blob)
    parts, _ = wire.unpack(blob, profile)
    old = (a.assets / "ps-nov22-al028-exact-budget/joint-53.5/rally.psj").read_bytes()
    old_profile = packet._embedded_profile(old)
    prior, _ = wire.unpack(old, old_profile)
    if (
        profile["cache_sha256"] != CACHE
        or profile["player"] != old_profile["player"]
        or parts["poses"] != prior["poses"]
        or parts["trajectory"] != prior["trajectory"]
        or parts["base"] != (high / "stream.ivf").read_bytes()
    ):
        raise ValueError("high channel conditioning or cache differs")
    os.environ["AL028_ASSETS"] = str(a.assets)
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/joint-replication-feasibility-20261122/al028_joint_recompose.py"
    )
    spec = importlib.util.spec_from_file_location("al028_high_score", path)
    if spec is None or spec.loader is None:
        raise ValueError("scorer missing")
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    score.BASES = {
        11: score.BASES[11],
        39: (
            high / "decoded.yuv",
            320,
            192,
            "36f5dd4444f90bdd83b26de0ef9da2ce6e4708d6cc5938df83482112f916f160",
        ),
    }
    score.BASELINES = {
        "av1_320_q37_75": (
            ordinary / "decoded.yuv",
            320,
            192,
            40.2475,
            "6223ebe351ea60873971e1ff379ac40ec7ed638770db22ec60eeaeedcdce0766",
        ),
        "av1_640_q55": (
            ordinary640 / "decoded.yuv",
            640,
            384,
            39.76583333333333,
            "7062f3d4c11b0bfda9b9fb34e845825538f3fde5b3c53d67b32969aaa6feaa8f",
        ),
    }
    out = Path(os.environ["PS_SCRATCH_DIR"]) / "al028-high-score"
    score.main(
        SimpleNamespace(
            finish=a.finish / "publish",
            finish_sha=FINISH,
            gen4=a.player1 / "publish",
            gen6=a.player2 / "publish",
            dcvc=None,
            dcvc_sha=None,
            out=out,
            frames=n,
            save_rates=[],
        )
    )
    result = json.loads((out / "result.json").read_text())
    result["scope"] = (
        "Separate high-rate paired-base recomposition; same sealed source-denied players, 25 fps, no human labels."
    )
    result["joint_warm_kbps"] = {"11": 11.133333333333333, "39": 39.751666666666665}
    result["high_joint_sha256"] = HIGH
    result["worker_elapsed_seconds"] = time.monotonic() - t
    stage = Path(os.environ["PS_STAGE_DIR"])
    (stage / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if n == 240:
        with tarfile.open(stage / "private-review.tar", "x") as tar:
            tar.add(out / "private-review", arcname="private-review")
    (stage / "output-hashes.json").write_text(
        json.dumps(
            {
                p.name: sha(p)
                for p in stage.iterdir()
                if p.name in ("result.json", "private-review.tar")
            },
            indent=2,
        )
        + "\n"
    )


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "result.json").read_text())
    h = json.loads((stage / "output-hashes.json").read_text())
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    checks = {
        "frames": r["frames"] == n,
        "witness_and_joint": len(r["rgb_stream_sha256"]) == 2 and r["high_joint_sha256"] == HIGH,
        "finisher": r["finish_receipt_sha256"] == FINISH,
        "result_hash": h["result.json"] == sha(stage / "result.json"),
    }
    forecast = 1.5 * r["worker_elapsed_seconds"] * 10 + 120
    if n == 24:
        checks["forecast"] = forecast < 1800
    else:
        checks["review"] = len(r["private_review_png_sha256"]) == 12 * 7 and h[
            "private-review.tar"
        ] == sha(stage / "private-review.tar")
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": all(checks.values()), "checks": checks, "forecast_seconds": forecast})
        + "\n"
    )
    if not all(checks.values()):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    for x in ("assets", "extra", "finish", "player1", "player2"):
        p.add_argument("--" + x, type=Path)
    p.add_argument("--frames", type=int, choices=(24, 240))
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)


if __name__ == "__main__":
    main()
