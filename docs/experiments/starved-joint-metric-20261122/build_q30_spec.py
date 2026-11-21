"""Bind an accepted Q30 producer archive to the scoped regional replay."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive.startswith("/home/itec/emanuele/Datasets/pointstream-data/"):
        raise ValueError("Q30 archive must be in the campus data root")
    if not re.fullmatch(r"[a-f0-9]{64}", args.archive_sha256):
        raise ValueError("Q30 archive digest invalid")
    spec = json.loads(Path("docs/experiments/starved-joint-metric-20261122/dcvc-q24-spec.json").read_text())
    spec["entrypoint"] = ["-m", "experiments.tennis.starved_joint_dcvc_q30"]
    spec["validator"] = ["{python}", "-m", "experiments.tennis.starved_joint_dcvc_q30", "--validate"]
    old = next(row for row in spec["staged_inputs"] if row["name"] == "old")
    previous = old["path"]
    old["path"], old["sha256"] = args.archive, args.archive_sha256
    for item in spec["inputs"]:
        if item["path"] == previous:
            item["path"], item["sha256"] = args.archive, args.archive_sha256
    spec["arguments"] += ["--extension-sha", args.archive_sha256]
    spec["smoke"]["seconds"] = 120
    spec["smoke"]["representative_basis"] = "Decode all 240 accepted Q30 frames and compare both passes to original per-frame hashes. Score one native 1080p tick and compare exact plain and ROI controls; preserve private stroke frames."
    spec["full"]["seconds"] = 220
    spec["budget_seconds"] = 400
    with args.output.open("x") as handle:
        json.dump(spec, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
