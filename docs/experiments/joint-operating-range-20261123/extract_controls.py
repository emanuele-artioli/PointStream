"""Export only frozen RGB scoring instants from an authenticated SVT control job."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
SCORER = HERE.parent / "shanghai-joint-review-20261122"
TICKS = [9, 31, 49, 72, 95, 112, 135, 152, 175, 198, 215, 238]
ARCHIVE_ROOT = "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox"


def run(args):
    status = json.loads(args.status.read_text())
    assert status["status"] == "complete" and status["gate"]["passed"]
    job = status["job_id"]
    key = f"{ARCHIVE_ROOT}/{job}/full"
    assert status["directory"] == f"{ARCHIVE_ROOT}/{job}"
    host = status["host_alias"]
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]
    hash_cmd = f"sha256sum {shlex.quote(key + '/published.tar')}"
    receipt_cmd = f"cat {shlex.quote(key + '/anchors.json')}"
    archive_hash = subprocess.check_output(ssh + [hash_cmd], timeout=90, text=True).split()[0]
    receipt = subprocess.check_output(ssh + [receipt_cmd], timeout=30)
    report = json.loads(receipt)
    review = json.loads((args.review / "review.json").read_text())
    assert report["source_sha256"] == review["source_video_sha256"]
    assert report["frames"] == 240 and report["fps"] == "30000/1001"
    assert [f["frame_index"] for f in review["frames"]] == TICKS
    selected = {}
    for name, r in report["selected"].items():
        assert r["affordable"] and r["bytes"] <= r["ceiling"]
        if report["mode"] == "foreground-roi":
            assert r["identical_qp_control"]["non_inert"]
        selected[name] = dict(r, file="publish/" + r["file"],
                              decoded_file="publish/" + r["decoded_file"])
    row = dict(state="complete", job_id=job, archive=f"{key}/published.tar",
               archive_sha256=archive_hash, width=report["width"], height=report["height"],
               mode=report["mode"], selected=selected)
    config = dict(rows=[row], ticks=TICKS, deduplicate=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "anchors.json").write_bytes(receipt)
    (args.output / "sample-source.json").write_text(json.dumps(config, indent=2) + "\n")
    sample_code = (SCORER / "sample_native_anchors.py").read_text()
    archive = args.output / "selected-native.tar"
    with archive.open("xb") as output:
        subprocess.run(ssh + ["python3 -c " + shlex.quote(sample_code)],
                       input=json.dumps(config).encode(), stdout=output, check=True, timeout=240)
    spec = importlib.util.spec_from_file_location("anchor_frames", SCORER / "anchor_frames.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_witness = json.loads(args.source_witness.read_text())
    module.render(archive, review, args.output / "arms", source_witness)
    for arm in sorted((args.output / "arms").iterdir()):
        score = args.output / f"{arm.name}.json"
        subprocess.run([args.python, str(SCORER / "score_receivers.py"),
                        "--review", str(args.review), "--arm", str(arm / "arm.json"),
                        "--automatic-official-ball", str(args.ball), "--out", str(score)],
                       check=True, timeout=240)
    with args.ball.open(encoding="utf-8-sig") as marks:
        visible = {int(r["Frame"]) for r in csv.DictReader(marks) if r["Visibility"] == "1"}
    assert set(TICKS) <= visible


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("status", "review", "source-witness", "ball", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--python", required=True)
    run(p.parse_args())
