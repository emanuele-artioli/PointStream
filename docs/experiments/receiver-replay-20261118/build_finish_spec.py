"""Construct the final SAM/composition request from accepted per-player seals."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from experiments.jobs.inbox import validate_spec
from experiments.tennis.player_reference_review import UNITS


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "docs/experiments"


def build(
    quality: str,
    directory: str,
    accepted: list[dict],
    remaining: int,
    deadline: str,
    hosts: list[str],
    remaining_cpu: int,
) -> dict:
    if (
        not 480 <= remaining <= 4200
        or not 480 <= remaining_cpu <= 7200
        or not hosts
        or set(hosts) - {"gpu5", "gpu6"}
    ):
        raise ValueError("invalid final resource reservation or host")
    module = "experiments.tennis.common_finish_job"
    q = json.loads((EXPERIMENTS / "joint-readiness-20261118/linux-carriers-zlib.json").read_text())[
        "carriers"
    ][quality]
    src = json.loads((EXPERIMENTS / "w7-specs/tier2-eval/receiver-nad000.json").read_text())
    template = json.loads((EXPERIMENTS / "appearance-resume-20261118/q70.spec.json").read_text())
    protected = json.loads(
        (
            EXPERIMENTS / "w8-inputs/receiver-20261101/runtime-qual-b581-identities-assets.json"
        ).read_text()
    )["protected_inputs"]
    items = [x for x in src["staged_inputs"] if x["name"] == "sam"]
    module_args = [
        "run",
        "--carrier",
        "{staged:carrier}",
        "--carrier-sha256",
        q["sha256"],
        "--manifest",
        "{staged:manifest}",
        "--sam",
        "{staged:sam}",
        "--ticks",
        "{ticks}",
    ]
    for key, ext, sha in [
        ("carrier", "pscp", q["sha256"]),
        ("manifest", "manifest", q["receiver_manifest_sha256"]),
    ]:
        items.append(
            {"name": key, "path": f"{directory}/q{quality}.{ext}", "sha256": sha, "extract": False}
        )
    if {r["unit"] for r in accepted} != set(UNITS) or len(accepted) != 2:
        raise ValueError("two distinct generated players required")
    for i, row in enumerate(accepted):
        key = f"player{i + 1}"
        if (
            row["carrier_sha256"] != q["sha256"]
            or row["frames"] != 240
            or not row["passed"]
            or not re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{8}", row["job_id"])
            or any(
                not re.fullmatch("[0-9a-f]{64}", row[k])
                for k in ("receipt_sha256", "archive_sha256")
            )
        ):
            raise ValueError("upstream generation seal mismatch")
        items.append(
            {
                "name": key,
                "extract": True,
                "sha256": row["archive_sha256"],
                "path": f"/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/{row['job_id']}/full/published.tar",
            }
        )
        module_args += ["--generation", row["unit"], "{staged:" + key + "}", row["receipt_sha256"]]
    for name in ("source", "prepared", "golden", "masks", "fit"):
        module_args += ["--deny", protected[name]["path"]]
    budget = min(2450, remaining, remaining_cpu)
    smoke = min(400, max(240, budget // 3))
    cap = budget - smoke
    module_args += ["--full-cap", str(cap)]
    template.update(
        entrypoint=["-m", module],
        arguments=module_args,
        staged_inputs=items,
        inputs=[{"path": x["path"], "sha256": x["sha256"]} for x in items],
        hosts=hosts,
        environment=src["environment"],
        scale={"ticks": {"smoke": 4, "full": 240}},
        smoke={
            "seconds": smoke,
            "representative_basis": "Both authentic 240-tick generations, decoded native base, and per-tick SAM at 0, 11, 23 and 43.",
        },
        full={"seconds": cap},
        validator=["{python}", "-m", module, "validate-stage"],
        validator_seconds=90,
        budget_seconds=budget,
        deadline=deadline,
        local_storage_gib=64,
    )
    validate_spec(template, now=time.time())
    return template


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality", required=True)
    p.add_argument("--carrier-directory", required=True)
    p.add_argument("--accepted", nargs=2, type=Path, required=True)
    p.add_argument("--remaining", type=int, required=True)
    p.add_argument(
        "--remaining-cpu",
        type=int,
        required=True,
        help="uncommitted in-job CPU seconds after external CPU reservations",
    )
    p.add_argument("--deadline", required=True)
    p.add_argument("--hosts", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = build(
        a.quality,
        a.carrier_directory,
        [json.loads(s.read_text()) for s in a.accepted],
        a.remaining,
        a.deadline,
        a.hosts,
        a.remaining_cpu,
    )
    with a.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
