"""Refresh only the execution deadline of the frozen Sydney full-receiver specifications."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prepare(deadline, output):
    end = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if end.utcoffset() is None or end <= datetime.now(timezone.utc):
        raise ValueError("The next execution window needs a future deadline with a UTC offset")
    updated = {}
    for role in ("near", "far"):
        name = f"match127-generate-{role}.json"
        spec = json.loads((HERE / name).read_text())
        args = spec["arguments"]
        assert args[args.index("--unit") + 1] == f"match127_{role}"
        assert spec["scale"] == {"ticks": {"smoke": 64, "full": 240}}
        assert spec["gpu_sharing"] == {"mode": "exclusive"}
        assert spec["full"]["seconds"] == 2680
        assert spec["budget_seconds"] == 3450
        spec["deadline"] = end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        updated[name] = spec
    output.mkdir(parents=True, exist_ok=False)
    for name, spec in updated.items():
        (output / name).write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.deadline, args.output)
