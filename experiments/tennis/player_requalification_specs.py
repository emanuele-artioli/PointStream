"""Prepare bounded paired-profile fleet requests; never submit or authorize them."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.jobs.inbox import validate_spec
from experiments.tennis.k6b_generate import sha256
from experiments.tennis.player_requalification import PROFILES

MODULE = "experiments.tennis.player_requalification"
REMOTE_RECIPE = "/home/itec/emanuele/Datasets/pointstream-data/tennis/player-requalification-20261116/config.json"


def build_specs(config: Path, inventory: Path, remote_recipe: str, deadline: str) -> dict[str, dict[str, Any]]:
    """Freeze identical inputs and numeric settings; release and live admission stay separate."""
    inputs = json.loads(inventory.read_text())
    files = copy.deepcopy(inputs["inputs"])
    if len(files) != 11 or {x["name"] for x in files} != {"conditioning", "base", "match", *(f"w{i}" for i in range(8))}:
        raise ValueError("exact eleven-input inventory required")
    recipe = {"name": "recipe", "path": remote_recipe, "sha256": sha256(config), "extract": False}
    staged = [recipe, *files]
    shared_arguments = ["--config", "{staged:recipe}", "--config-sha256", recipe["sha256"],
                        "--max-units", "{units}"]
    shared_arguments += [arg for row in files for arg in ("--staged", row["name"], "{staged:" + row["name"] + "}")]
    result = {}
    for profile in PROFILES:
        spec = {
            "schema": 1, "hosts": ["gpu5", "gpu6"], "gpu_models": ["RTX 6000 Ada"],
            "gpu_memory_mib": 18000, "cpu_threads": 1, "single_cpu_campaign": True,
            "gpu_sharing": {"mode": "shared", "margin_mib": 8192,
                            "basis": "Model/conditioning correctness only; wall times are contaminated and cannot establish speed."},
            "entrypoint": ["-m", MODULE], "arguments": ["run", "--profile", profile, *shared_arguments],
            "scale": {"units": {"smoke": 1, "full": 2}},
            "inputs": [{"path": x["path"], "sha256": x["sha256"]} for x in staged],
            "staged_inputs": staged, "verify_staged_inputs": True,
            "environment": inputs["environment"], "local_storage_gib": 48,
            "smoke": {"seconds": 600, "representative_basis": "First complete 24-native-frame player context; same production 512x784,30steps,CFG3.5,reference,pose,weights and sampler as full two-player context run."},
            "full": {"seconds": 900}, "budget_seconds": 1800, "deadline": deadline,
            "validator": ["{python}", "-m", MODULE, "validate-stage"], "validator_seconds": 60,
            "required_commands": [], "contention": {"policy": "continue", "resume_attempts": 0,
                "basis": "Correctness and reconstruction checks do not use timing; shared wall times are contaminated."},
        }
        # Pure schema checks do not establish file availability, cache correctness,
        # runtime, worker isolation, scientific validity, or spending authority.
        validate_spec(spec)
        result[profile] = spec
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--remote-recipe", default=REMOTE_RECIPE)
    parser.add_argument("--deadline", required=True, help="Refresh after actual release; timezone required")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = build_specs(args.config, args.inputs, args.remote_recipe, args.deadline)
    args.output.mkdir(exist_ok=False, parents=True)
    for profile, spec in specs.items():
        (args.output / f"{profile}.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "prepared-not-submitted", "profiles": list(specs),
                      "max_job_gpu_seconds": 5400, "release_required": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
