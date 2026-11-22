"""Index a completed low-rate control and sample its verified native decoded frames."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
SAMPLE = HERE.parent / "shanghai-joint-review-20261122/sample_native_anchors.py"
SOURCE = HERE.parent / "test-crop-joint-20261122/match102-source-review-frames.json"


def collect(label, output):
    ledgers = [HERE / "validated-jobs.jsonl", HERE / "small-jobs.jsonl"]
    registered = [
        json.loads(line)
        for ledger in ledgers
        if ledger.exists()
        for line in ledger.read_text().splitlines()
    ]
    entries = [row for row in registered if row["label"] == label]
    assert len(entries) == 1
    job = entries[0]["job"]
    status = json.loads(subprocess.check_output(["scripts/ps-fleet", "status", job]))
    assert status["job_id"] == job and status["status"] == "complete"
    assert status["gate"]["spec_sha256"] == entries[0]["spec_sha256"]
    prefix = status["directory"] + "/full/"
    ssh = ["ssh", "-o", "BatchMode=yes", "gpu6"]
    receipt = subprocess.check_output([*ssh, "cat", prefix + "anchors.json"])
    native = json.loads(receipt)
    source = json.loads(SOURCE.read_text())
    assert native["source_sha256"] == source["source_sha256"]
    assert native["frames"] == 240 and native["fps"] == "30000/1001"
    archive = prefix + "published.tar"
    digest = subprocess.check_output([*ssh, "sha256sum", archive]).decode().split()[0]
    selected = {}
    for name, r in native["selected"].items():
        assert r["ceiling"] == {"warm": 13555, "cold": 151035}[name]
        if r["affordable"]:
            assert r["bytes"] <= r["ceiling"]
            selected[name] = dict(
                r, file="publish/" + r["file"], decoded_file="publish/" + r["decoded_file"]
            )
        else:
            selected[name] = {"affordable": False}
    mode = "roi" if label.startswith("ROI") else "ordinary"
    row = dict(
        job_id=job,
        state="complete",
        mode=mode,
        width=native["width"],
        height=native["height"],
        archive=archive,
        archive_sha256=digest,
        selected=selected,
        report_sha256=hashlib.sha256(receipt).hexdigest(),
    )
    output.mkdir(parents=True, exist_ok=True)
    for p, payload in (
        (HERE / (label + "-receipt.json"), receipt),
        (HERE / (label + "-index.json"), (json.dumps(row, indent=2) + "\n").encode()),
    ):
        if p.exists():
            assert p.read_bytes() == payload
        else:
            p.write_bytes(payload)
    dest = output / (label + "-sample.tar")
    if dest.exists() or not any(v["affordable"] for v in selected.values()):
        return
    part = dest.with_suffix(".part")
    ticks = sorted(map(int, source["ticks"]))
    with part.open("xb") as f:
        subprocess.run(
            [*ssh, "python3 -c " + shlex.quote(SAMPLE.read_text())],
            input=json.dumps(dict(rows=[row], ticks=ticks)).encode(),
            stdout=f,
            check=True,
            timeout=600,
        )
    part.rename(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("labels", nargs="+")
    args = p.parse_args()
    for label in args.labels:
        collect(label, args.output)
