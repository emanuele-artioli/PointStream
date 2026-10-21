"""Prepare immutable K8 inputs and a CPU fleet specification (no source data writes).

Run on the shared filesystem, with --dest under Datasets/pointstream-data.
The source video is hard-linked; its sha256 is recorded. Reference photographs
are inventoried as provenance, not misrepresented as a geometric court model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.datasets.k4_inputs import ARCHIVE, RAW_MATCH, file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--full-frames", type=int, default=6000)
    args = parser.parse_args()
    dest = args.dest.resolve()
    dest.relative_to((args.datasets / "pointstream-data").resolve())
    dest.mkdir(parents=True, exist_ok=False)
    video = args.datasets / ARCHIVE / RAW_MATCH
    linked = dest / "nadal_kyrgios.mp4"
    os.link(video, linked)
    identity = {"video_path": str(video), "video_sha256": file_sha256(video),
                "reference_path": str(args.datasets / "Tennis_courts/reference"),
                "reference_description": "966 court photographs, no geometric model; user authorized regulation geometry",
                "court_geometry": {"length_m": 23.77, "doubles_width_m": 10.97, "singles_width_m": 8.23,
                                   "service_from_net_m": 6.40, "net_excluded": True,
                                   "source": "https://www.itftennis.com/en/about-us/organisation/tennis-glossary/"}}
    content = json.dumps(identity, indent=2, sort_keys=True) + "\n"
    inputs = dest / f"inputs-{hashlib.sha256(content.encode()).hexdigest()[:12]}.json"
    inputs.write_text(content)
    spec = {"schema": 1, "entrypoint": ["-m", "experiments.tennis.k8"],
            "arguments": ["run", "--video", "{staged:video}", "--inputs", "{staged:inputs}",
                          "--frames", "{frames}", "--workers", "8", "--threshold", "0.25"],
            "scale": {"frames": {"smoke": 1500, "full": args.full_frames}},
            "environment": {"path": str(args.environment), "sha256": args.environment_sha256},
            "inputs": [{"path": str(inputs), "sha256": file_sha256(inputs)}],
            "staged_inputs": [{"name": "inputs", "path": str(inputs), "sha256": file_sha256(inputs)},
                              {"name": "video", "path": str(linked), "sha256": identity["video_sha256"]}],
            "smoke": {"seconds": 600, "representative_basis": "First 30 seconds at full 1080p50 through ffmpeg scene cuts, every-frame YUV420 conversion, white-line hypotheses, RANSAC, 1080p chamfer, shot rule, review JPEGs and HTML. Includes broadcast transitions; validator requires real court registrations."},
            "full": {"seconds": 2700}, "validator": ["{python}", "-m", "experiments.tennis.k8", "validate"],
            "validator_seconds": 60, "budget_seconds": 3500,
            "deadline": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(),
            "stall_seconds": 600, "cpu_threads": 16, "device": "cpu", "gpu_memory_mib": 0, "gpu_models": [],
            "hosts": [f"gpu{i}" for i in range(1, 7)], "local_storage_gib": 30,
            "required_commands": ["ffmpeg", "ffprobe"]}
    (dest / "job.json").write_text(json.dumps(spec, indent=2) + "\n")
    for part in range(4):
        full = json.loads(json.dumps(spec))
        full["arguments"].extend(["--part", "{part}", "--parts", "4"])
        full["scale"] = {"frames": {"smoke": 1500, "full": 0}, "part": {"smoke": -1, "full": part}}
        full["stall_seconds"] = 1800
        (dest / f"k8-full-{part}.json").write_text(json.dumps(full, indent=2) + "\n")
    print(json.dumps({"spec": str(dest / "job.json"), "inputs": identity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
