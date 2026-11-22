"""Bind signed receiver outputs, evaluation pixels, and source masks for compact campus scoring."""

import argparse
import json
from pathlib import Path

from experiments.tennis.player_requalification import sha256

HERE = Path(__file__).resolve().parent
AL = HERE.parent / "al031-complete-receiver-20261122"
SH = HERE.parent / "test-crop-joint-20261122"
FLEET = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def load(p):
    return json.loads(p.read_text())


def make(a):
    al = a.clip == "al031"
    card = AL if al else HERE
    orig = card / "neural-q24.json" if al else HERE / f"{a.clip}-neural-q24-validated.json"
    if a.clip == "000":
        orig = HERE / "000-neural-outer-q22.json"
    source = next(r for r in load(orig)["staged_inputs"] if r["name"] == "source")
    grid = card / (a.clip + "-campus-evaluation-inputs.json")
    g = load(grid)
    template = load(SH / "match102-campus-score-calibration-dispatch.json")
    weights = next(v for v in template["staged_inputs"] if v["name"] == "weights")
    remote = Path(g["canonical_path"]).parent
    assert g["source_sha256"] == source["sha256"]
    inputs = [source, weights]

    def add(name, path, sha, extract=False):
        inputs.append(dict(name=name, path=str(path), sha256=sha, extract=extract))

    add("canonical", g["canonical_path"], g["canonical_sha256"])
    add("grid", remote / "grid.json", sha256(grid))
    add("supports", g["supports_tar_path"], g["supports_tar_sha256"], True)
    options = [
        "--clip",
        a.clip,
        "--label",
        a.label,
        "--kind",
        a.kind,
        "--source",
        "{staged:source}",
        "--source-sha",
        source["sha256"],
        "--canonical",
        "{staged:canonical}",
        "--canonical-receipt",
        "{staged:grid}",
        "--supports",
        "{staged:supports}",
        "--weights",
        "{staged:weights}",
        "--data",
        "{staged:data}",
        "--frames",
        "{frames}",
        "--producer",
        a.producer,
        "--fps",
        str(g["fps"]),
        "--paid",
        str(a.paid),
    ]
    root = FLEET / a.producer / "full"
    add(
        "data",
        root / ("frames.zseq" if a.kind == "rung" else "published.tar"),
        a.sha,
        a.kind != "rung",
    )
    if a.kind == "rung":
        add("rung-receipt", root / "rung.json", a.receipt_sha)
        options += ["--receipt", "{staged:rung-receipt}"]
    if a.kind in ["finish", "rung"]:
        r = load(card / (a.prefix + "-prices.json" if al else f"{a.clip}-packets.json"))
        wire = next(x for x in r["rates"] if x["crf"] == a.crf)
        assert wire["joint_bytes"] == a.paid
        options += ["--expected-joint", wire["joint_sha256"], "--cache", str(wire["init_bytes"])]
    else:
        options += ["--qp", str(a.qp)]
    if al:
        sender = next(
            v["job"]
            for v in map(json.loads, (AL / "jobs.jsonl").read_text().splitlines())
            if v["stage"] == "ball-assembly"
        )
        src = FLEET / sender / "full"
        br = AL / "source-ball-receipt.json"
        add("ball", src / "ball.payload", load(br)["payload_sha256"])
        add("ball-receipt", src / "sender.json", sha256(br))
        options += ["--ball-payload", "{staged:ball}", "--ball-receipt", "{staged:ball-receipt}"]
    if a.expected_dists is not None:
        options += ["--expected-dists", str(a.expected_dists)]
    template.update(
        entrypoint=["-m", "experiments.tennis.foreground_campus_score"],
        validator=["{python}", "-m", "experiments.tennis.foreground_campus_score", "--validate"],
        arguments=options,
        staged_inputs=inputs,
        inputs=[{k: q[k] for k in ["path", "sha256"]} for q in inputs],
        deadline="2026-11-22T21:01:00Z",
        hosts=["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
    )
    (card / a.out).write_text(json.dumps(template, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--kind", choices=["neural", "finish", "rung"], required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--paid", required=True, type=int)
    p.add_argument("--receipt-sha")
    p.add_argument("--prefix", choices=["fresh", "prior"])
    p.add_argument("--crf", type=int)
    p.add_argument("--qp", type=int)
    p.add_argument("--expected-dists", type=float)
    p.add_argument("--out", required=True)
    make(p.parse_args())
