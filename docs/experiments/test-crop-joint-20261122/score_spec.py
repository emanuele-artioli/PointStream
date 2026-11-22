"""Prepare a one-core campus scoring job that returns only twelve PNGs and a numerical receipt."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def make(a):
    template = json.loads(
        (HERE.parent / "cohort-expansion-20261122/test-physical/match102-neural.json").read_text()
    )
    assert template["native_fps"] == [30000, 1001]
    original = json.loads((HERE / "match102-spec.json").read_text())
    scored = json.loads(
        (
            HERE.parent / "shanghai-prospective-20261122/neural-score-half-label-spec.json"
        ).read_text()
    )
    inputs = [
        {**r, "extract": False}
        for r in original["staged_inputs"]
        if r["name"] in ("source", "features")
    ]
    weights = next(r for r in scored["staged_inputs"] if r["name"] == "backbone")
    inputs.append(dict(weights, name="weights"))
    dest = "frames.zseq" if a.kind == "rung" else "published.tar"
    inputs.append(
        dict(
            name="data",
            path=str(INBOX / a.producer / "full" / dest),
            sha256=a.sha,
            extract=a.kind == "full",
        )
    )
    options = [
        "--data",
        "{staged:data}",
        "--kind",
        a.kind,
        "--producer-job",
        a.producer,
        "--crf",
        str(a.crf),
        "--source",
        "{staged:source}",
        "--features",
        "{staged:features}",
        "--weights",
        "{staged:weights}",
        "--full-cap",
        "1600",
        "--frames",
        "{frames}",
    ]
    if a.kind == "rung":
        assert a.receipt_sha and len(a.receipt_sha) == 64
        inputs.append(
            dict(
                name="receipt",
                path=str(INBOX / a.producer / "full/rung.json"),
                sha256=a.receipt_sha,
                extract=False,
            )
        )
        options += ["--receipt", "{staged:receipt}"]
    module = "experiments.tennis.test_joint_native_score"
    spec = dict(
        schema=1,
        device="cpu",
        entrypoint=["-m", module],
        validator=["{python}", "-m", module, "--validate", *options],
        arguments=options,
        staged_inputs=inputs,
        inputs=[{k: r[k] for k in ("path", "sha256")} for r in inputs],
        environment=scored["environment"],
        required_commands=[],
        deadline="2026-11-22T21:01:00Z",
        verify_staged_inputs=True,
        hosts=["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
        gpu_models=[],
        gpu_memory_mib=0,
        cpu_threads=1,
        single_cpu_campaign=True,
        local_storage_gib=28,
        smoke=dict(
            seconds=420,
            representative_basis="24 native frames with two actual player support masks per tick, independent source pixel digest, CPU DISTS and optional audited PNG extraction. Whole-job forecast includes fixed validation costs.",
        ),
        full=dict(seconds=1600),
        validator_seconds=150,
        budget_seconds=2430,
        stall_seconds=1660,
        scale=dict(frames=dict(smoke=24, full=240)),
    )
    # Fleet validators receive expanded runtime arguments via their own declaration.
    p = HERE / (
        "match102-campus-score-"
        + a.kind
        + ("-" + str(a.crf) if a.kind != "calibration" else "")
        + ".json"
    )
    p.write_text(json.dumps(spec, indent=2) + "\n")
    print(p.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("calibration", "full", "rung"), required=True)
    p.add_argument("--crf", type=int, default=50)
    p.add_argument("--producer", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--receipt-sha")
    make(p.parse_args())
