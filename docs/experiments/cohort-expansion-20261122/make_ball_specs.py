"""Create a downstream TEST request only from completed, checksum-identified upstream outputs."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from make_sender_specs import main as generate_sender_specs

HERE = Path(__file__).resolve().parent
INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def create(kind, match, job, up, out, archive_sha=None):
    base = json.loads(
        (HERE.parent / "shanghai-prospective-20261122/primary-score-gpu3-spec.json").read_text()
    )
    with TemporaryDirectory() as directory:
        local = Path(directory) / "specs"
        generate_sender_specs(local)
        sender_spec = json.loads((local / f"{match}-sender-spec.json").read_text())
    inputs = [x for x in sender_spec["staged_inputs"] if x["name"] in ("source", "metadata")]
    selection = next(
        x
        for x in json.loads((HERE / "staged.json").read_text())["selection"]
        if x["match"] == match
    )
    receipt = json.loads(up.read_text())
    assert receipt["source_sha256"] == selection["source"]["sha256"] and receipt["frames"] == 240

    def stage(name, path, sha):
        if len(sha) != 64:
            raise ValueError("invalid receipt digest")
        inputs.append(dict(name=name, path=str(path), sha256=sha, extract=False))

    site = INBOX / job / "full"
    if kind == "rate":
        stage("sender", site / "sender.json", hashlib.sha256(up.read_bytes()).hexdigest())
        stage("payload", site / "ball.payload", receipt["payload_sha256"])
        module = "experiments.tennis.cohort_ball_rate"
        args = [
            v
            for name in ("source", "metadata", "sender", "payload")
            for v in ("--" + name, "{staged:" + name + "}")
        ]
        seconds = 2300
        smoke_seconds = 400
    else:
        assert kind == "score" and archive_sha and receipt["match"] == match
        stage("rate", site / "published.tar", archive_sha)
        label = selection["labels"]
        stage("labels", label["path"], label["sha256"])
        inputs += [x for x in base["staged_inputs"] if x["name"] == "backbone"]
        module = "experiments.tennis.cohort_ball_score"
        args = [
            v
            for name in ("source", "metadata", "rate", "labels", "backbone")
            for v in ("--" + name, "{staged:" + name + "}")
        ]
        args += ["--rate-sha", archive_sha, "--labels-sha", label["sha256"]]
        seconds = 2200
        smoke_seconds = 400
    result = dict(
        sender_spec,
        entrypoint=["-m", module],
        arguments=args + ["--frames", "{frames}"],
        staged_inputs=inputs,
        inputs=[{key: x[key] for key in ("path", "sha256")} for x in inputs],
        validator=["{python}", "-m", module, "--validate"],
        smoke=dict(
            seconds=smoke_seconds,
            representative_basis="Real 24-frame prefix at this clip's native cadence, both geometry and complete packet paths, result identity and bounded extrapolation.",
        ),
        full=dict(seconds=seconds),
        budget_seconds=seconds + smoke_seconds + 180,
        stall_seconds=seconds + 10,
    )
    with out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("rate", "score"))
    p.add_argument("match")
    p.add_argument("job")
    p.add_argument("receipt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--archive-sha")
    a = p.parse_args()
    create(a.kind, a.match, a.job, a.receipt, a.out, a.archive_sha)
