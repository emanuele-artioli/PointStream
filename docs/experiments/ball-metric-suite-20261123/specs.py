"""Generate one match's rescore spec from its completed original fleet receipt."""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "cohort-expansion-20261122" / "test-physical"
MODULE = "experiments.tennis.cohort_ball_metric_suite"


def make(status, event, output):
    assert status["status"] == "complete" and status["gate"]["passed"]
    prior = PRIOR / (event + "-score.json")
    primary = json.loads(prior.read_text())
    assert primary["frames"] == 240 and primary["match"] == event
    keys = ("source", "metadata", "rate", "labels", "backbone")
    inputs = []
    for name, rec in zip(keys, status["gate"]["inputs"], strict=True):
        inputs.append(dict(rec, name=name, extract=False))
    assert inputs[0]["sha256"] == primary["source_sha256"]
    assert inputs[2]["sha256"] == primary["rate_archive_sha256"]
    assert inputs[3]["sha256"] == primary["labels_sha256"]
    inputs.extend((
        dict(name="score", path=status["directory"] + "/full/scores.json",
             sha256=hashlib.sha256(prior.read_bytes()).hexdigest(), extract=False),
        dict(name="lpips", path="/home/itec/emanuele/Datasets/pointstream-data/visor/"
             "h3-2026-10-10/inputs/lpips/alexnet-owt-7be5be79.pth",
             sha256="7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
             extract=False),
    ))
    base = json.loads((HERE.parent / "shanghai-joint-low-anchors-20261122" /
                       "validated-1920.json").read_text())
    spec = dict(
        schema=1, entrypoint=["-m", MODULE],
        arguments=[a for r in inputs for a in ("--" + r["name"], "{staged:" + r["name"] + "}")] +
        ["--frames", "{frames}"],
        validator=["{python}", "-m", MODULE, "--validate"],
        hosts=[status["host_alias"]], environment=base["environment"], staged_inputs=inputs,
        inputs=[{k: i[k] for k in ("path", "sha256")} for i in inputs],
        verify_staged_inputs=True, device="cpu", gpu_models=[], gpu_memory_mib=0, cpu_threads=1,
        single_cpu_campaign=True, local_storage_gib=12, required_commands=[],
        smoke=dict(seconds=90, representative_basis="Six stratified real ball crops per arm. "
                   "Full native decoded-pixel witness, model identities and prior DISTS parity."),
        full=dict(seconds=315), validator_seconds=15, budget_seconds=420, stall_seconds=315,
        deadline="2026-11-23T17:45:00Z", scale={"frames": {"smoke": 24, "full": 240}},
    )
    with output.open("x") as stream:
        stream.write(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("status", type=Path)
    p.add_argument("match")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    make(json.loads(args.status.read_text()), args.match, args.output)
