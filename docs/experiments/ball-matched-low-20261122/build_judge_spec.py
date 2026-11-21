"""Construct the approved fixed-checkpoint WASB pilot after CPU artifact verification."""
import argparse
import json
import re
from pathlib import Path

BASE = "/home/itec/emanuele/Datasets/pointstream-data"
CPU_JOB = "20261121T215556Z-5d39ce00"
ENV = BASE + "/environments/pointstream-20261009T214002Z.tar.gz"
ENV_SHA = "32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661"
CODE = BASE + "/tennis/k5b-2026-10-20/inputs/WASB-SBDT-923462c.tar"
CODE_SHA = "657fb5e0c06bbe621f10bdf6d8937f5cb7f8cbef1c4a369e30d9d87f54e87a55"
WEIGHTS = BASE + "/tennis/k5b-2026-10-20/inputs/wasb-tennis-weights.tar"
WEIGHTS_SHA = "4129d4210145dd029e393028deef717a10b59a603f93e5081c1058b8f178f5e6"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-archive-sha256", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{64}", args.packet_archive_sha256):
        raise ValueError("Missing artifact checksum")
    staged = [
        {"name": "packets", "path": BASE + f"/jobs/fleet/inbox/{CPU_JOB}/full/published.tar",
         "sha256": args.packet_archive_sha256, "extract": False},
        {"name": "wasb_code", "path": CODE, "sha256": CODE_SHA, "extract": True},
        {"name": "wasb_weights", "path": WEIGHTS, "sha256": WEIGHTS_SHA, "extract": True},
    ]
    spec = {
        "schema": 1, "entrypoint": ["-m", "experiments.tennis.ball_matched_low_judge"],
        "arguments": ["--packets", "{staged:packets}", "--wasb-code", "{staged:wasb_code}",
                      "--wasb-weights", "{staged:wasb_weights}", "--frames", "{frames}"],
        "staged_inputs": staged,
        "inputs": [{"path": x["path"], "sha256": x["sha256"]} for x in staged],
        "environment": {"path": ENV, "sha256": ENV_SHA},
        "required_commands": ["ffmpeg"],
        "validator": ["{python}", "-m", "experiments.tennis.ball_matched_low_judge", "--validate"],
        "deadline": args.deadline, "verify_staged_inputs": True, "stall_seconds": 600,
        "hosts": ["gpu5", "gpu6"], "gpu_models": ["RTX 6000 Ada"], "gpu_memory_mib": 10000,
        "cpu_threads": 1, "single_cpu_campaign": True, "local_storage_gib": 30,
        "contention": {"policy": "stop", "resume_attempts": 0},
        "smoke": {"seconds": 240, "representative_basis": "24 frames per actual decoded packet and same WASB model on CUDA; require full inference forecast to fit the full stage."},
        "full": {"seconds": 850}, "validator_seconds": 60, "budget_seconds": 1200,
        "scale": {"frames": {"smoke": 24, "full": 292}},
    }
    with args.output.open("x") as out:
        json.dump(spec, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
