"""Recompose the exact CRF54 delivery with the original Al028 receiver on gpu3 CPU."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace

from experiments.tennis import common_player_packet, w8_payload
from experiments.tennis.al028_high_score_job import CACHE, FINISH, sha

WIRE = "1093859b4bcf14d0ebe1f1abe08fdf518acd21b763ed2b23327b2826c9f47020"
BASE = "d190d55ff0b24a052baa3ae46d08d4f968418d93ff24d472870befaa6c7823b7"
YUV = "ebbe171ef7a763abe276e5b2f62d1ebccbf2ea0af7aeb053b66e4e84960b9218"


def validate(stage, n):
    record = json.loads((stage / "result.json").read_text())
    checks = [
        record["frames"] == n,
        record["wire_sha256"] == WIRE,
        record["wire_bytes"] == 24257,
        record["finish_receipt_sha256"] == FINISH,
        set(record["rgb_stream_sha256"]) == {"11", "54"},
        record["published_frames"] == (1 if n == 24 else 12),
    ]
    if n == 24:
        checks.append(record["forecast_seconds"] < 265)
    else:
        checks.append(sha(stage / "samples.tar") == record["samples_sha256"])
    path = os.getenv("PS_VALIDATION_PATH")
    if path:
        Path(path).write_text(json.dumps({"passed": all(checks), "checks": checks}) + "\n")
    if not all(checks):
        raise ValueError("native validation or cost bounds failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "extra", "finish", "player1", "player2"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--validate", action="store_true")
    a = parser.parse_args()
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        validate(stage, n)
        return
    if n != a.frames:
        raise ValueError("wrong stage")
    started = time.monotonic()
    data = (a.extra / "rally.psj").read_bytes()
    original = (a.assets / "ps-nov22-al028-exact-budget/joint-53.5/rally.psj").read_bytes()
    profile = common_player_packet._embedded_profile(data)
    old_profile = common_player_packet._embedded_profile(original)
    parts, _ = w8_payload.unpack(data, profile)
    prior, _ = w8_payload.unpack(original, old_profile)
    if (
        len(data) != 24257
        or sha(a.extra / "rally.psj") != WIRE
        or sha(a.extra / "stream.ivf") != BASE
        or parts["base"] != (a.extra / "stream.ivf").read_bytes()
        or profile != old_profile
        or profile["cache_sha256"] != CACHE
        or any(parts[k] != prior[k] for k in ("poses", "trajectory"))
    ):
        raise ValueError("paid stream or receiver state differs")
    os.environ["AL028_ASSETS"] = str(a.assets)
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/joint-replication-feasibility-20261122/al028_joint_recompose.py"
    )
    spec = importlib.util.spec_from_file_location("al028_recompose_strict", source)
    if spec is None or spec.loader is None:
        raise ValueError("missing receiver")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.BASES = {11: runner.BASES[11], 54: (a.extra / "decoded.yuv", 320, 192, YUV)}
    runner.BASELINES = {}
    out = Path(os.environ["PS_SCRATCH_DIR"]) / "strict54"
    runner.main(
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
    r = json.loads((out / "result.json").read_text())
    r["scope"] = (
        "Two exact native displays from the same receiver; human-label metrics scored separately."
    )
    r["joint_warm_kbps"] = {"11": 11.133333333333333, "54": 24257 / 1200}
    r["wire_sha256"] = WIRE
    r["wire_bytes"] = len(data)
    samples = list(sorted((out / "private-review").glob("*/joint-54.png")))
    r["published_frames"] = len(samples)
    if n == 240:
        with tarfile.open(stage / "samples.tar", "x") as tf:
            for name in samples:
                tf.add(name, arcname=name.parent.name + ".png")
        r["samples_sha256"] = sha(stage / "samples.tar")
    r["elapsed_seconds_including_audit"] = time.monotonic() - started
    r["forecast_seconds"] = 1.5 * 10 * r["elapsed_seconds"] + 60
    (stage / "result.json").write_text(json.dumps(r, indent=2) + "\n")
    validate(stage, n)


if __name__ == "__main__":
    main()
