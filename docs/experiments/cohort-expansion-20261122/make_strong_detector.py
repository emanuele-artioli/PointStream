"""Prepare original-host lossless native rendering or a CUDA readout for the selected affordable anchor."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path


def create(a):
    spec = json.loads(a.template.read_text())
    status = json.loads(a.qualification_status.read_text())
    q = json.loads(a.qualification.read_text())
    assert status["status"] == "complete" and status["gate"]["passed"] and q["source_qualified"]
    strong = a.mode == "judge"
    names = (
        ("score", "labels", "wasb_code", "wasb_weights")
        if strong
        else ("score", "smoke_score", "rate", "metadata")
    )
    inputs = [r for r in spec["staged_inputs"] if r["name"] in names]
    assert len(inputs) == len(names)
    inputs.append(
        dict(
            name="qualification",
            path=status["directory"] + "/full/judge.json",
            extract=False,
            sha256=hashlib.sha256(a.qualification.read_bytes()).hexdigest(),
        )
    )
    extra = []
    if strong:
        d = json.loads(a.display_status.read_text())
        sha = a.display_sha.read_text().strip()
        assert d["status"] == "complete" and d["gate"]["passed"] and len(sha) == 64
        inputs.append(
            dict(
                name="display",
                path=d["directory"] + "/full/published.tar",
                extract=False,
                sha256=sha,
            )
        )
        extra = ["--display-sha", sha]
        spec["contention"] = {
            "policy": "continue",
            "basis": "Pixel and localization quality on immutable inputs is independent of runtime.",
            "resume_attempts": 0,
        }
        spec["gpu_sharing"] = {
            "mode": "shared",
            "basis": "Frozen objective measurements on the validated Ada judge; wall time is excluded from throughput comparisons.",
            "margin_mib": 8192,
        }
    module = "experiments.tennis.cohort_affordable_detector"
    flags = [
        v
        for item in inputs
        for v in ("--" + item["name"].replace("_", "-"), "{staged:" + item["name"] + "}")
    ]
    full_seconds = 540 if strong else 1200
    spec.update(
        entrypoint=["-m", module],
        arguments=["--mode", a.mode, *flags, *extra, "--frames", "{frames}"],
        inputs=[{k: r[k] for k in ("path", "sha256")} for r in inputs],
        staged_inputs=inputs,
        validator=["{python}", "-m", module, "--validate"],
        smoke=dict(
            seconds=180 if strong else 230,
            representative_basis="24 true native ticks and detector/model or codec/receiver pixel witness; full source-qualified match fixed in advance.",
        ),
        full=dict(seconds=full_seconds),
        stall_seconds=full_seconds + 10,
        budget_seconds=full_seconds + (290 if strong else 430),
        validator_seconds=90,
    )
    with a.out.open("x") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "judge"))
    for name in ("template", "qualification_status", "qualification", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--display-status", type=Path)
    p.add_argument("--display-sha", type=Path)
    create(p.parse_args())
