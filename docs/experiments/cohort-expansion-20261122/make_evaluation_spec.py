"""Bind a sparse-label evaluation to the exact host and pixels of an accepted TEST score."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def create(a):
    upstream = json.loads(a.score_status.read_text())
    full = json.loads(a.score.read_text())
    smoke = json.loads(a.smoke_score.read_text())
    assert upstream["status"] == "complete" and upstream["gate"]["passed"]
    assert (
        full["match"] == smoke["match"] == a.match
        and full["frames"] == 240
        and smoke["frames"] == 24
    )
    base = json.loads(a.score_spec.read_text())
    assert base["entrypoint"][-1] == "experiments.tennis.cohort_ball_score"
    run = INBOX / upstream["job_id"]
    inputs = [i for i in base["staged_inputs"] if i["name"] in ("source", "metadata", "labels")]

    def add(name, path, sha):
        assert len(sha) == 64
        inputs.append(dict(name=name, path=str(path), sha256=sha, extract=False))

    add("score", run / "full/scores.json", hashlib.sha256(a.score.read_bytes()).hexdigest())
    add(
        "smoke_score",
        run / "smoke/scores.json",
        hashlib.sha256(a.smoke_score.read_bytes()).hexdigest(),
    )
    if a.kind == "judge":
        module = "experiments.tennis.cohort_ball_judge"
        inp = next(i for i in base["staged_inputs"] if i["name"] == "rate")
        assert inp["sha256"] == full["rate_archive_sha256"]
        inputs.append(inp)
        template = json.loads(
            (HERE.parent / "shanghai-prospective-20261122/primary-score-gpu3-spec.json").read_text()
        )
        inputs.extend(
            i for i in template["staged_inputs"] if i["name"] in ("wasb_code", "wasb_weights")
        )
        assert len(inputs) == 8
        settings = ["--rate-sha", inp["sha256"]]
        full_seconds = 2300
    else:
        assert a.kind == "neuroscore"
        module = "experiments.tennis.cohort_neural_score"
        neural = json.loads(a.neural_status.read_text())
        assert neural["status"] == "complete" and neural["gate"]["passed"]
        sha = a.neural_archive_sha.read_text().strip()
        add("neural", INBOX / neural["job_id"] / "full/published.tar", sha)
        inputs.append(next(i for i in base["staged_inputs"] if i["name"] == "backbone"))
        settings = ["--neural-sha", sha]
        if a.qp != 24:
            settings.extend(["--qp", str(a.qp)])
        full_seconds = 1500
    names = [
        v
        for i in inputs
        for v in ("--" + i["name"].replace("_", "-"), "{staged:" + i["name"] + "}")
    ]
    r = dict(
        base,
        entrypoint=["-m", module],
        arguments=names
        + settings
        + ["--labels-sha", full["labels_sha256"], "--frames", "{frames}"],
        validator=["{python}", "-m", module, "--validate"],
        staged_inputs=inputs,
        inputs=[{k: i[k] for k in ("path", "sha256")} for i in inputs],
        hosts=[upstream["host_alias"]],
        smoke={
            "seconds": 400,
            "representative_basis": "Complete 24-tick native source, pixel hash binding and metric/model path with independently scored source marks.",
        },
        full={"seconds": full_seconds},
        budget_seconds=full_seconds + 600,
        stall_seconds=full_seconds + 15,
    )
    if getattr(a, "cuda", False):
        assert a.kind == "judge"
        gpu = json.loads(
            (HERE.parent / "ball-matched-low-20261122/refit-judge-spec.json").read_text()
        )
        assert gpu["single_cpu_campaign"]
        r["single_cpu_campaign"] = True
        r.update(
            device="gpu",
            hosts=gpu["hosts"],
            gpu_models=gpu["gpu_models"],
            gpu_memory_mib=gpu["gpu_memory_mib"],
            gpu_sharing={"mode": "exclusive"},
            smoke=dict(
                seconds=115,
                representative_basis="Real full-pixel identities, previously validated WASB CUDA model, all three arms on the first 24 source ticks and conservative cost check.",
            ),
            full=dict(seconds=310),
            stall_seconds=320,
            validator_seconds=35,
            budget_seconds=510,
        )
        r["arguments"] += ["--cuda"]
        r["validator"] += ["--cuda"]
    with a.out.open("x") as f:
        json.dump(r, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("judge", "neuroscore"))
    p.add_argument("match")
    for k in ("score-spec", "score-status", "score", "smoke-score", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("neural-status", "neural-archive-sha"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--qp", type=int, choices=(12, 24, 36, 48), default=24)
    create(p.parse_args())
