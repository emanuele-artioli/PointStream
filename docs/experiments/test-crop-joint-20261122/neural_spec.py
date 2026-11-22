"""Derive a bounded integer-Q comparator from the existing native four-decode control."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(a):
    sp = json.loads(a.template.read_text())
    assert sp["entrypoint"] == ["-m", "experiments.tennis.shanghai_neural_control"]
    assert (
        sp["budget_seconds"] == 540
        and sp["staged_inputs"][0]["sha256"]
        == json.loads((HERE / "match102-result.json").read_text())["source_sha256"]
    )
    for key in ("arguments", "validator"):
        vals = sp[key]
        vals[vals.index("--qp") + 1] = str(a.qp)
        vals[vals.index("--scale") + 1] = str(a.scale)
        assert vals[vals.index("--fps") + 1] == "30000/1001"
    spec = HERE / f"match102-neural-s{a.scale}-q{a.qp}.json"
    spec.write_text(json.dumps(sp, indent=2) + "\n")
    print(spec.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--qp", type=int, choices=range(64), required=True)
    p.add_argument("--scale", type=int, choices=(2, 4), required=True)
    main(p.parse_args())
