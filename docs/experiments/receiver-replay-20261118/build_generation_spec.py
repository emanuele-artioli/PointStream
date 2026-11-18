"""Materialize one dated fleet request after reference and budget selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.jobs.inbox import validate_spec
from experiments.tennis.player_reference_review import UNITS

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "docs" / "experiments"


def create(
    quality: str,
    unit: str,
    remaining: int,
    deadline: str,
    carrier_directory: str,
    hosts: list[str],
    remaining_cpu: int,
) -> dict:
    registry = json.loads(
        (EXPERIMENTS / "joint-readiness-20261118/linux-carriers-zlib.json").read_text()
    )["carriers"][quality]
    if (
        not 0 < remaining_cpu <= 7200
        or remaining > 4200
        or remaining < 1250
        or unit not in UNITS
        or not hosts
        or set(hosts) - {"gpu5", "gpu6"}
    ):
        raise ValueError("invalid resource reservation, unit or host")
    template = json.loads((EXPERIMENTS / "appearance-resume-20261118/q70.spec.json").read_text())
    protected = json.loads(
        (
            EXPERIMENTS / "w8-inputs/receiver-20261101/runtime-qual-b581-identities-assets.json"
        ).read_text()
    )["protected_inputs"]
    staged = [
        item
        for item in template["staged_inputs"]
        if item["name"] in {"base", *(f"w{i}" for i in range(8))}
    ]
    # The non-source receiver gets only the carrier, manifest, common fit and original weights.
    for key, ext, digest in [
        ("carrier", "pscp", registry["sha256"]),
        ("manifest", "manifest", registry["receiver_manifest_sha256"]),
    ]:
        staged.append(
            {
                "name": key,
                "path": f"{carrier_directory}/q{quality}.{ext}",
                "sha256": digest,
                "extract": False,
            }
        )
    budget = min(2500, remaining - 600, remaining_cpu - 600)
    if budget < 1450:
        raise ValueError(
            "insufficient room for one historical 44/240 generation and mask-finisher floor"
        )
    cap = min(1800, budget - 600)
    args = [
        "run",
        "--carrier",
        "{staged:carrier}",
        "--manifest",
        "{staged:manifest}",
        "--carrier-sha256",
        registry["sha256"],
        "--unit",
        unit,
        "--frames",
        "{ticks}",
        "--full-cap",
        str(cap),
        "--base",
        "{staged:base}",
    ]
    for i in range(8):
        args.extend(["--weight", "{staged:w" + str(i) + "}"])
    for name in ("source", "prepared", "golden", "masks", "fit"):
        args.extend(["--deny", protected[name]["path"]])
    module = "experiments.tennis.common_receiver_job"
    template.update(
        entrypoint=["-m", module],
        arguments=args,
        staged_inputs=staged,
        inputs=[{"path": i["path"], "sha256": i["sha256"]} for i in staged],
        scale={"ticks": {"smoke": 44, "full": 240}},
        smoke={
            "seconds": 600,
            "representative_basis": "One full two-context window from the same paid 240-frame packet; native CUDA and source denial under the full decoder.",
        },
        full={"seconds": cap},
        budget_seconds=budget,
        deadline=deadline,
        hosts=hosts,
        validator=["{python}", "-m", module, "validate-stage"],
        validator_seconds=90,
    )
    validate_spec(template, now=time.time())
    return template


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality", required=True)
    p.add_argument("--unit", choices=UNITS, required=True)
    p.add_argument("--remaining", type=int, required=True)
    p.add_argument(
        "--remaining-cpu",
        type=int,
        required=True,
        help="uncommitted in-job CPU seconds after external CPU reservations",
    )
    p.add_argument("--deadline", required=True)
    p.add_argument("--carrier-directory", required=True)
    p.add_argument("--hosts", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    spec = create(
        args.quality,
        args.unit,
        args.remaining,
        args.deadline,
        args.carrier_directory,
        args.hosts,
        args.remaining_cpu,
    )
    with args.out.open("x") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
