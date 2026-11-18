"""Create the next exclusive PNG stage from retained upstream seals and charged lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))
from build_finish_spec import build  # noqa: E402
from build_generation_spec import create  # noqa: E402
from experiments.jobs.inbox import validate_spec  # noqa: E402
from experiments.tennis.player_reference_review import UNITS  # noqa: E402


def next_spec(stage: str, spent: int, accepted: list[dict], now: datetime) -> dict:
    # Nonoverlapping grants include validator, input staging and output publication.
    # Reservations survive a cheap early completion and accrue to subsequent stages.
    available = 5700 - spent
    if spent < 435 or not 0 < available <= 5265:
        raise ValueError("invalid cumulative charge")
    required = {"player2": [], "player1": [UNITS[0]], "finish": list(UNITS)}[stage]
    if [r["unit"] for r in accepted] != required or any(
        not r["passed"] or r["frames"] != 240 for r in accepted
    ):
        raise ValueError("missing full predecessor")
    directory = json.loads((HERE.parent / "selected-publication.json").read_text())["directory"]
    hosts = ["gpu5", "gpu6"]
    # Forty minutes for admission. The fleet expires a pending job once its
    # remaining UTC window cannot cover both stage limits. Waiting uses no GPU.
    deadline = now + timedelta(minutes=40)
    if stage != "finish":
        future = 2265 + 800 if stage == "player2" else 800
        budget = min(2400, available - future)
        smoke = 400
        full = min(1800, budget - smoke)
        if full < 1640:
            raise ValueError("historical exclusive player upper estimate no longer fits")
        end = (deadline + timedelta(seconds=smoke + full)).isoformat()
        # Keep input identities and one-core isolation from the registered spec.
        spec = create(
            "png", UNITS[0] if stage == "player2" else UNITS[1], budget + 600, end, directory, hosts, budget + 600
        )
        args = spec["arguments"]
        args[args.index("--full-cap") + 1] = str(full)
        spec["smoke"]["seconds"] = smoke
        spec["full"]["seconds"] = full
        spec["budget_seconds"] = budget
    else:
        if available < 800:
            raise ValueError("protected finisher resource is missing")
        budget = min(2450, available)
        end = (deadline + timedelta(seconds=budget)).isoformat()
        spec = build("png", directory, accepted, budget, end, hosts, budget)
    spec["gpu_sharing"] = {"mode": "exclusive"}
    spec["contention"] = {"policy": "stop", "resume_attempts": 0}
    validate_spec(spec, now=time.time())
    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["player2", "player1", "finish"])
    parser.add_argument("--spent", required=True, type=int)
    parser.add_argument("--accepted", type=Path, nargs="*", default=[])
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args()
    spec = next_spec(
        a.stage,
        a.spent,
        [json.loads(p.read_text()) for p in a.accepted],
        datetime.now(timezone.utc),
    )
    with a.out.open("x") as output:
        output.write(json.dumps(spec, indent=2) + "\n")
