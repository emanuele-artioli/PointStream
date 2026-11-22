"""CPU-only conditioning check in the actual GPU generator environment."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from experiments.tennis import common_receiver_generate as gen

EXPECTED = "731cc34e1fcb7c0874cd5722ab63241a123d31fbcd699e199cee4163cdcbc0dc"
UNITS = [
    "alcaraz_hurkacz__028_at120__person_000000000004",
    "alcaraz_hurkacz__028_at120__person_000000000006",
]


def main(a):
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        data = json.loads((stage / "check.json").read_text())
        ok = (
            data["sha256"] == EXPECTED
            and len(data["units"]) == 2
            and all(x["count"] == data["frames"] for x in data["units"])
        )
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(
            json.dumps({"passed": ok, "checks": {"conditioning": ok}}) + "\n"
        )
        if not ok:
            raise SystemExit(1)
        return
    if hashlib.sha256(a.carrier.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("carrier identity")
    rows = []
    for name, seed, arrays in gen.conditioning(a.carrier, EXPECTED, a.frames, UNITS):
        if arrays["pose"].shape != (a.frames, 784, 512, 3) or not np.array_equal(
            arrays["t"], np.arange(a.frames)
        ):
            raise ValueError("conditioning shape or cadence")
        rows.append(
            {
                "name": name,
                "seed": seed,
                "count": len(arrays["t"]),
                "reference_sha256": hashlib.sha256(np.ascontiguousarray(arrays["reference"])).hexdigest(),
                "pose_sha256": hashlib.sha256(np.ascontiguousarray(arrays["pose"])).hexdigest(),
            }
        )
    data = {
        "sha256": EXPECTED,
        "frames": a.frames,
        "zstd_available": importlib.util.find_spec("zstandard") is not None,
        "units": rows,
        "scope": "CPU environment compatibility; no GPU inference or quality result.",
    }
    (stage / "check.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--carrier", type=Path)
    p.add_argument("--frames", type=int, choices=(24, 44))
    p.add_argument("--validate", action="store_true")
    main(p.parse_args())
