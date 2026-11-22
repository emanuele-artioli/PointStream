"""Cost one byte-guided Sydney DCVC-UF stream using the already verified four-pass worker."""

from argparse import ArgumentParser
import json
from pathlib import Path

if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("template", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("width", type=int, choices=(480, 960))
    p.add_argument("qp", type=int, choices=range(64))
    a = p.parse_args()
    spec = json.loads(a.template.read_text())
    assert spec["entrypoint"] == ["-m", "experiments.tennis.shanghai_neural_control"] and any(
        i["name"] == "source"
        and i["sha256"] == "8ebca8b693202d8bf7afaee515dde4d723a9fb83250115e6eff080038b0bbf50"
        for i in spec["staged_inputs"]
    )
    limit = {480: 420, 960: 900}[a.width]
    for key in ("arguments", "validator"):
        args = spec[key]
        for flag, value in [("--qp", a.qp), ("--scale", 1920 // a.width), ("--full-limit", limit)]:
            args[args.index(flag) + 1] = str(value)
    spec.update(
        smoke=dict(spec["smoke"], seconds=160),
        full={"seconds": limit},
        budget_seconds=limit + 255,
        stall_seconds=limit + 10,
    )
    with a.out.open("x") as stream:
        json.dump(spec, stream, indent=2)
        stream.write("\n")
