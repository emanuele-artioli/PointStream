"""Construct a bounded /002 finisher only after both genuine seals exist."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--second-archive-sha", required=True)
    parser.add_argument("--second-receipt-sha", required=True)
    parser.add_argument("--second-job", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{8}", args.second_job):
        raise ValueError("invalid second generation job identity")
    if not all(
        re.fullmatch(r"[a-f0-9]{64}", value)
        for value in (args.second_archive_sha, args.second_receipt_sha)
    ):
        raise ValueError("invalid second generation digest")
    root = "/home/itec/emanuele/Datasets/pointstream-data/"
    source = Path("docs/experiments/receiver-replay-20261118/retry-1600utc/finish4.spec.json")
    spec = json.loads(source.read_text())
    carrier = root + "tennis/cumulative-joint-20261122/return-cached-player-v5/receiver.pscp"
    manifest = root + "tennis/cumulative-joint-20261122/return-cached-player-v5/receiver.manifest"
    carrier_sha = "a0ec63326642130adb5d765a3e7c30aaf564ad5bbf7cd17d56594ffd82dc5b06"
    values = {
        "carrier": (carrier, carrier_sha),
        "manifest": (manifest, "250c8aa159a3915fdce6a4c5ee4d436b3dddcf14886b6ee39468b105723539b5"),
        "player1": (
            root + "jobs/fleet/inbox/20261122T002538Z-607ae615/full/published.tar",
            "9b56dbaf14d261a9c0dfff4dbabb246a913fbf988d4fac8e9b45a7e993270e72",
        ),
        "player2": (
            root + "jobs/fleet/inbox/" + args.second_job + "/full/published.tar",
            args.second_archive_sha,
        ),
    }
    for item in spec["staged_inputs"]:
        if item["name"] in values:
            item["path"], item["sha256"] = values[item["name"]]
    spec["inputs"] = [
        {"path": item["path"], "sha256": item["sha256"]} for item in spec["staged_inputs"]
    ]
    original = spec["arguments"]
    first_deny = original.index("--deny")
    denial_args = original[first_deny : original.index("--full-cap")]
    for path in [
        root + "tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__002.mp4",
        root + "tennis/cumulative-joint-20261122/nadal-kyrgios-broadcast.mp4",
        root + "tennis/cross-shot-reference-20261122/reuse/conditioning.tar",
        root
        + "tennis/cross-shot-reference-20261122/reference-channel-lossless-webp-v1/initial.psr",
    ]:
        denial_args += ["--deny", path]
    unit0 = "nadal_kyrgios__002_at120__person_000000000000"
    unit2 = "nadal_kyrgios__002_at120__person_000000000002"
    spec["arguments"] = [
        "run",
        "--carrier",
        "{staged:carrier}",
        "--carrier-sha256",
        carrier_sha,
        "--manifest",
        "{staged:manifest}",
        "--sam",
        "{staged:sam}",
        "--ticks",
        "{ticks}",
        "--generation",
        unit0,
        "{staged:player1}",
        "9dd95be42e30bdff5d176624a4f697dacabeb89cc04fe0b7a9d93c49b661439b",
        "--generation",
        unit2,
        "{staged:player2}",
        args.second_receipt_sha,
        *denial_args,
        "--full-cap",
        "1000",
    ]
    spec["entrypoint"] = ["-m", "experiments.tennis.common_finish_return_job"]
    spec["validator"] = [
        "{python}",
        "-m",
        "experiments.tennis.common_finish_return_job",
        "validate-stage",
    ]
    spec["hosts"] = ["gpu5", "gpu6"]
    spec["gpu_sharing"] = {
        "mode": "shared",
        "margin_mib": 8192,
        "basis": "Correctness-only source-denied finish; timings may be contended.",
    }
    spec["contention"] = {
        "policy": "continue",
        "resume_attempts": 0,
        "basis": "Correctness and pixel identity only; shared timings are not performance evidence.",
    }
    spec["smoke"] = {
        "seconds": 300,
        "representative_basis": "Four authentic receiver frames and masks using both sealed return tracks.",
    }
    spec["full"] = {"seconds": 1000}
    spec["validator_seconds"] = 90
    spec["budget_seconds"] = 1500
    spec["deadline"] = args.deadline
    args.out.write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
