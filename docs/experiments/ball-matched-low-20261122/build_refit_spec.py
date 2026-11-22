"""Build a source-bound CPU refit spec after the rate-matching artifact exists."""
import argparse
import json
import re
from pathlib import Path

BASE = "/home/itec/emanuele/Datasets/pointstream-data"
CPU_JOB = "20261121T215556Z-5d39ce00"
SOURCE = BASE + "/tennis/k5d-20261104-inputs/tracknet-dataset.zip"
ORIGINAL = BASE + "/jobs/fleet/inbox/20261113T173340Z-64160888/full/published.tar"
ENV = BASE + "/environments/pointstream-20261009T214002Z.tar.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matched-sha256", required=True)
    p.add_argument("--deadline", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[a-f0-9]{64}", args.matched_sha256):
        raise ValueError("Missing accepted matched-packet checksum")
    staged = [
        {"name": "source", "path": SOURCE, "sha256": "bb45372d011ef4092d78173169c8ad6280d7dbdb9998efaf2b1c72a63ca1dd18", "extract": False},
        {"name": "original", "path": ORIGINAL, "sha256": "e939b3e79e5fedb534ae135148045eb490d9d7e9aca3c92c5bb0edbf1de53e0d", "extract": False},
        {"name": "matched", "path": BASE + f"/jobs/fleet/inbox/{CPU_JOB}/full/published.tar", "sha256": args.matched_sha256, "extract": False},
    ]
    spec = {
        "schema": 1, "entrypoint": ["-m", "experiments.tennis.ball_matched_low_refit"],
        "arguments": ["--source", "{staged:source}", "--original", "{staged:original}",
                      "--matched", "{staged:matched}", "--frames", "{frames}"],
        "staged_inputs": staged,
        "inputs": [{"path": x["path"], "sha256": x["sha256"]} for x in staged],
        "environment": {"path": ENV, "sha256": "32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661"},
        "required_commands": ["ffmpeg"],
        "validator": ["{python}", "-m", "experiments.tennis.ball_matched_low_refit", "--validate"],
        "deadline": args.deadline, "verify_staged_inputs": True, "stall_seconds": 600,
        "device": "cpu", "hosts": ["gpu5", "gpu6"], "gpu_models": [], "gpu_memory_mib": 0,
        "cpu_threads": 1, "single_cpu_campaign": True, "local_storage_gib": 30,
        "smoke": {"seconds": 240, "representative_basis": "24 real frames, source-only archived BallTrack predictions, two actual base decodes and production streak fitting; project full fit with margin."},
        "full": {"seconds": 720}, "validator_seconds": 60, "budget_seconds": 1040,
        "scale": {"frames": {"smoke": 24, "full": 292}},
    }
    with args.output.open("x") as out:
        json.dump(spec, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
