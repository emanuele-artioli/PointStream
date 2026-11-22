"""Bind each frozen TEST case to its already closed K4 publication before crop inference."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(a):
    assert a.event in ("match108", "match102", "match120")
    old = HERE.parent / "test-joint-admission-20261122"
    d = json.loads((old / (a.event + "-features.json")).read_text())
    if a.archive:
        assert a.job
        with tarfile.open(a.archive) as archive:
            output = json.load(archive.extractfile("publish/admission.json"))
        receipt = dict(
            archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),
            job=a.job,
            full=output,
        )
    else:
        receipt = json.loads((old / (a.event + "-outcome.json")).read_text())
    data = {v["name"]: v for v in d["staged_inputs"]}
    digest = data["source"]["sha256"]
    fs = receipt["archive_sha256"]
    job = receipt["job"]
    got = receipt["full"]
    assert got["event"] == a.event and got["source_sha256"] == digest
    staged = [
        data["source"],
        data["yolo11x_pose"],
        dict(
            name="features",
            extract=True,
            path=f"/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/{job}/full/published.tar",
            sha256=fs,
        ),
    ]
    module = "experiments.tennis.test_crop_pose"
    d.update(
        entrypoint=["-m", module],
        staged_inputs=staged,
        inputs=[{k: v for k, v in row.items() if k in ("path", "sha256")} for row in staged],
        arguments=[
            "--event",
            a.event,
            "--source",
            "{staged:source}",
            "--source-sha",
            digest,
            "--features",
            "{staged:features}",
            "--feature-sha",
            fs,
            "--pose",
            "{staged:yolo11x_pose}",
            "--frames",
            "{frames}",
        ],
        validator=["{python}", "-m", module, "--validate"],
        scale={"frames": {"smoke": 24, "full": 240}},
        smoke=dict(
            seconds=230,
            representative_basis="All 240 source masks determine the original K6 scale rule; two athletes and the first 24 native source ticks run the registered estimator.",
        ),
        full=dict(seconds=655),
        budget_seconds=1080,
        cpu_threads=3,
        gpu_memory_mib=12000,
        local_storage_gib=15,
        deadline="2026-11-22T15:01:00Z",
        validator_seconds=60,
    )
    (HERE / (a.event + "-spec.json")).write_text(json.dumps(d, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("event")
    p.add_argument("--archive", type=Path)
    p.add_argument("--job")
    main(p.parse_args())
