"""Build a registered 480x272 DCVC-UF control for all twelve official TEST sources."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def create(directory, qp=24):
    template = json.loads(
        (HERE.parent / "shanghai-prospective-20261122/neural-q24-half-spec.json").read_text()
    )
    for key in ("arguments", "validator"):
        template[key][template[key].index("--qp") + 1] = str(qp)
    suffix = "-neural" if qp == 24 else f"-neural-q{qp}"
    for case in json.loads((HERE / "staged.json").read_text())["selection"]:
        source = dict(
            name="source",
            path=case["source"]["path"],
            sha256=case["source"]["sha256"],
            extract=False,
        )
        staged = [source, *template["staged_inputs"][1:]]
        flags = [
            "--source-sha",
            source["sha256"],
            "--fps",
            case["video"]["avg_frame_rate"],
            "--full-limit",
            "300",
        ]
        spec = dict(
            template,
            staged_inputs=staged,
            inputs=[{k: s[k] for k in ("path", "sha256")} for s in staged],
            arguments=template["arguments"] + flags,
            validator=template["validator"] + flags,
            deadline="2026-11-22T21:01:00Z",
            smoke={**template["smoke"], "seconds": 120},
            full={"seconds": 300},
            validator_seconds=40,
            stall_seconds=310,
            budget_seconds=540,
        )
        with (directory / (case["match"] + suffix + ".json")).open("x") as f:
            json.dump(spec, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--qp", type=int, choices=range(64), default=24)
    args = p.parse_args()
    create(args.directory, args.qp)
