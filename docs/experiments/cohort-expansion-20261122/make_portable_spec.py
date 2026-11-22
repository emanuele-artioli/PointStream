"""Stage byte-identical primary RGB displays across CPU architectures for the CUDA judge."""

import argparse
import json
from pathlib import Path


def build(a):
    spec = json.loads(a.judge_spec.read_text())
    assert spec["device"] == "gpu" and spec["single_cpu_campaign"]
    if a.action == "freeze":
        old = json.loads(a.score_status.read_text())
        assert old["status"] == "complete" and old["gate"]["passed"]
        inputs = [
            i
            for i in spec["staged_inputs"]
            if i["name"] in ("source", "metadata", "rate", "score", "smoke_score")
        ]
        assert len(inputs) == 5
        args = [
            v
            for i in inputs
            for v in ("--" + i["name"].replace("_", "-"), "{staged:" + i["name"] + "}")
        ]
        rate = next(i["sha256"] for i in inputs if i["name"] == "rate")
        module = "experiments.tennis.cohort_ball_display"
        spec.update(
            device="cpu",
            hosts=[old["host_alias"]],
            gpu_models=[],
            gpu_memory_mib=0,
            entrypoint=["-m", module],
            arguments=args + ["--rate-sha", rate, "--frames", "{frames}"],
            validator=["{python}", "-m", module, "--validate"],
            staged_inputs=inputs,
            inputs=[{k: i[k] for k in ("path", "sha256")} for i in inputs],
            smoke=dict(
                seconds=300,
                representative_basis="Lossless read-back, full RGB checksums for all three real 24-frame displays on the scoring host, and throughput forecast.",
            ),
            full=dict(seconds=1400),
            validator_seconds=160,
            budget_seconds=2050,
            stall_seconds=1450,
        )
        spec.pop("gpu_sharing")
    else:
        assert a.action == "judge"
        spec["contention"] = {
            "policy": "continue",
            "basis": "Pixel and localization quality on immutable inputs is independent of runtime.",
            "resume_attempts": 0,
        }
        spec["gpu_sharing"] = {
            "mode": "shared",
            "basis": "The fixed-pixel WASB objective and frozen source labels do not depend on wall-clock timing. This GPU model has passed the native CUDA and CPU-pixel verification.",
            "margin_mib": 8192,
        }
        prior = json.loads(a.display_status.read_text())
        assert prior["status"] == "complete" and prior["gate"]["passed"]
        sha = a.display_archive_sha.read_text().strip()
        assert len(sha) == 64
        inp = dict(
            name="frozen_display",
            path=prior["directory"] + "/full/published.tar",
            sha256=sha,
            extract=False,
        )
        spec["staged_inputs"].append(inp)
        spec["inputs"].append({k: inp[k] for k in ("path", "sha256")})
        spec["arguments"] += ["--frozen-display", "{staged:frozen_display}", "--frozen-sha", sha]
    with a.out.open("x") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "judge"))
    p.add_argument("judge_spec", type=Path)
    p.add_argument("out", type=Path)
    for k in ("score-status", "display-status", "display-archive-sha"):
        p.add_argument("--" + k, type=Path)
    build(p.parse_args())
