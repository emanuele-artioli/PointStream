"""Use only frozen prices, source identities and the primary scorer’s native host."""

import argparse
import hashlib
import json
from pathlib import Path


def build(primary, status, receipt, grid):
    member = next(r for r in grid["members"] if r["match"] == receipt["match"])
    assert (
        member["rate_archive_sha256"] == receipt["rate_archive_sha256"]
        and status["status"] == "complete"
        and status["gate"]["passed"]
    )
    inputs = list(primary["staged_inputs"])
    assert {i["name"] for i in inputs} == {"source", "metadata", "rate", "labels", "backbone"}
    addr = status["directory"] + "/full/scores.json"
    inp = dict(
        name="score",
        path=addr,
        sha256=hashlib.sha256(
            (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
        ).hexdigest(),
        extract=False,
    )
    inputs.append(inp)
    module = "experiments.tennis.cohort_ball_curve"
    args = [v for i in inputs for v in ("--" + i["name"], "{staged:" + i["name"] + "}")]
    return dict(
        primary,
        entrypoint=["-m", module],
        arguments=args
        + [
            "--rate-sha",
            member["rate_archive_sha256"],
            "--labels-sha",
            receipt["labels_sha256"],
            "--frames",
            "{frames}",
        ],
        hosts=[status["host_alias"]],
        validator=["{python}", "-m", module, "--validate"],
        staged_inputs=inputs,
        inputs=[{k: i[k] for k in ("path", "sha256")} for i in inputs],
        smoke=dict(
            seconds=450,
            representative_basis="Verified full-packet decode and complete source identity for all physical rates; at most six evenly spaced visible source marks through all DISTS paths, with per-source cost forecast.",
        ),
        full=dict(seconds=2300),
        budget_seconds=2930,
        stall_seconds=2320,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("primary", "status", "receipt", "grid", "out"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    r = build(*(json.loads(k.read_text()) for k in (a.primary, a.status, a.receipt, a.grid)))
    with a.out.open("x") as f:
        json.dump(r, f, indent=2)
        f.write("\n")
