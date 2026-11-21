"""Build a W7 plain-control CPU fleet specification without submitting it."""
import argparse
import json
from datetime import datetime
from pathlib import Path

SOURCE = "/home/itec/emanuele/Datasets/pointstream-data/tennis/w7-tier1-20261030/sources/alcaraz_hurkacz__028.mp4"
SOURCE_SHA = "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
ENV = "/home/itec/emanuele/Datasets/pointstream-data/environments/pointstream-20261009T214002Z.tar.gz"
ENV_SHA = "32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True, help="ISO 8601 timestamp with timezone")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    parsed = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        parser.error("Deadline must include a timezone")
    spec = {
        "schema": 1,
        "entrypoint": ["-m", "experiments.tennis.starved_native_fleet"],
        "arguments": ["--source", "{staged:source}", "--frames", "{frames}", "--scale", "4"],
        "staged_inputs": [{"name": "source", "path": SOURCE, "sha256": SOURCE_SHA, "extract": False}],
        "inputs": [{"path": SOURCE, "sha256": SOURCE_SHA}],
        "environment": {"path": ENV, "sha256": ENV_SHA},
        "required_commands": ["ffmpeg", "ffprobe", "SvtAv1EncApp", "dav1d"],
        "validator": ["{python}", "-m", "experiments.tennis.starved_native_fleet", "--validate"],
        "deadline": args.deadline,
        "verify_staged_inputs": True,
        "stall_seconds": 1000,
        "contention": {"policy": "stop", "resume_attempts": 0},
        "device": "cpu",
        "hosts": ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
        "gpu_models": [],
        "gpu_memory_mib": 0,
        "cpu_threads": 1,
        "single_cpu_campaign": True,
        "local_storage_gib": 30,
        "smoke": {
            "seconds": 300,
            "representative_basis": "24 source frames, same pinned scale-four SVT-AV1 encode and dav1d decode; independent decode validation and conservative full projection",
        },
        "full": {"seconds": 900},
        "validator_seconds": 120,
        "budget_seconds": 1700,
        "scale": {"frames": {"smoke": 24, "full": 240}},
    }
    with args.output.open("x") as output:
        json.dump(spec, output, indent=2)
        output.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
