"""Bounded infrastructure campaign: one CUDA operation and explicit validation.

Not codec evidence. Outputs are always written to PS_STAGE_DIR, outside source.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.jobs.monitor import publish_progress, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    output = Path(os.environ["PS_STAGE_DIR"])
    if args.validate:
        result = json.loads((output / "result.json").read_text())
        checks = {
            "cuda": result["device"] == "cuda:0",
            "isolated_gpu": bool(result["visible_gpu"]),
            "computed_result": result["sum"] == 1024 * result["iterations"],
            "representative_input": result["input"] == {"infrastructure_smoke": True},
        }
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), {"passed": all(checks.values()), "checks": checks, "citable": False})
        return 0 if all(checks.values()) else 1
    import torch

    if args.iterations < 1 or args.iterations > 8:
        parser.error("infrastructure smoke iterations must be in 1..8")
    identity = json.loads(args.input.read_text())
    torch.cuda.reset_peak_memory_stats()
    values = torch.zeros(1024, device="cuda:0")
    for completed in range(args.iterations):
        values += 1
        torch.cuda.synchronize()
        publish_progress(os.environ["PS_STAGE"], completed + 1)
    write_json(output / "result.json", {
        "device": str(values.device), "visible_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "iterations": args.iterations, "sum": values.sum().item(), "input": identity,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "torch": torch.__version__,
        "citable": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
