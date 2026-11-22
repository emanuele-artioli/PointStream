"""Judge the 24-frame returned-appearance pilot on campus without exporting generated pixels."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.tennis.player_requalification import sha256

HERE = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(p.read_text())


def check(a, stage, report, dest):
    rows = report["measure"]["rows"]
    receipt = report["measure"]["crop_receipts"]
    assert len(a.job) == len(set(a.job)) == 2
    checks = dict(
        ticks=[(r["tick"], r["player"]) for r in rows]
        == [(t, i) for t in range(a.frames) for i in [0, 1]],
        localizations=all(r["pixels"] > 0 and 0 < r["coverage"] <= 1 for r in rows),
        authentic_seeds=len(receipt) == 4
        and all(
            {q["job"] for q in receipt if q["index"] == i} == {a.job[i]}
            and len({q["seed"] for q in receipt if q["index"] == i}) == 1
            for i in [0, 1]
        ),
        output_provenance=len({q["crop"] for q in receipt}) == 4,
        source_witness=report["measure"]["source_sha256"] == read(a.grid)["source_sha256"],
        forecast=report["seconds"] < 1500
        and (stage == "full" or report["seconds"] * 1.5 * 24 / a.frames + 75 < 1500),
    )
    res = dict(passed=all(checks.values()), checks=checks)
    dest.write_text(json.dumps(res) + "\n")
    assert res["passed"], res


def run(a):
    begin = time.monotonic()
    stage = os.environ["PS_STAGE"]
    assert a.frames == {"smoke": 3, "full": 24}[stage]
    out = Path(os.environ["PS_STAGE_DIR"])
    base = Path(os.environ["PS_SCRATCH_DIR"])
    grid = read(a.grid)
    assert (
        sha256(a.canonical) == grid["canonical_sha256"]
        and sha256(a.source) == grid["source_sha256"]
    )
    meta = read(a.bank)
    archive = base / "banks"
    inputs = [a.first_smoke, a.first_full, a.second_smoke, a.second_full]
    for (i, stage_name), src in zip(
        [(0, "smoke"), (0, "full"), (1, "smoke"), (1, "full")], inputs, strict=True
    ):
        job = a.job[i]
        r = meta[job][stage_name]
        where = archive / job / stage_name
        where.mkdir(parents=True)
        assert sha256(src) == r["timing"]["archive_sha256"]
        (where / "published.tar").symlink_to(src)
        for field, name in [("timing", "worker_timing.json"), ("execution", "execution.json")]:
            (where / name).write_text(json.dumps(r[field]) + "\n")
    records = []
    for old in read(a.paid / "arms.json"):
        b = a.paid / old["stream_name"]
        f = base / (old["name"] + ".yuv")
        assert sha256(b) == old["stream_sha256"] and b.stat().st_size == old["bytes"]
        subprocess.run(
            ["dav1d", "-i", str(b), "-o", str(f), "--muxer", "yuv", "--threads", "1"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        assert sha256(f) == old["sha256"]
        records.append(dict(old, path=str(f)))
    dest = base / "anchors.json"
    dest.write_text(json.dumps(records))
    module = importlib.util.spec_from_file_location(
        "return_score", HERE / "docs/experiments/al031-complete-receiver-20261122/prefix_score.py"
    )
    assert module and module.loader
    m = importlib.util.module_from_spec(module)
    module.loader.exec_module(m)
    result = base / "prefix-scores.json"
    m.run(
        argparse.Namespace(
            source=a.source,
            canonical=a.canonical,
            supports=a.supports / "supports.json",
            weights=a.weights,
            anchors=dest,
            archives=archive,
            fresh=a.paid / "fresh",
            prior=a.paid / "prior",
            out=result,
            jobs=a.job,
            budget=a.budget,
            threads=1,
            frames=a.frames,
        )
    )
    report = dict(
        measure=read(result),
        seconds=time.monotonic() - begin,
        canonical_sha=grid["canonical_sha256"],
    )
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    check(a, stage, report, out / "prefix_validation.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in [
        "source",
        "supports",
        "weights",
        "canonical",
        "grid",
        "bank",
        "paid",
        "first-smoke",
        "first-full",
        "second-smoke",
        "second-full",
    ]:
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--job", action="append", required=True)
    p.add_argument("--frames", choices=[3, 24], type=int, required=True)
    p.add_argument("--budget", required=True, type=float)
    if sys.argv[1:] == ["--validate"]:
        s = Path(os.environ["PS_STAGE_DIR"])
        c = read(s / "dispatch.json")["command"]
        assert c[1:3] == ["-m", "experiments.tennis.return_cache_prefix_score"]
        check(
            p.parse_args(c[3:]),
            os.environ["PS_STAGE"],
            read(s / "result.json"),
            Path(os.environ["PS_VALIDATION_PATH"]),
        )
    else:
        run(p.parse_args())
