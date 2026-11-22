"""Prepare a warm, source-derived, joint-foreground ROI anchor. No human labels are staged."""

import argparse
import json
from pathlib import Path

FEATURE_SHA = "994101c8d3466080777c262d92b9f15953c5636cc4addae06f75ad478d3594ee"
BALL_SHA = "3ad726835c0e51c11daa80d9b54630a1e52494227680b434254e583950426918"
INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def build(base, out):
    d = json.loads(base.read_text())
    for name, path, sha in [
        ("roi_features", INBOX / "20261122T095722Z-14da4d11/full/published.tar", FEATURE_SHA),
        ("ball_payload", INBOX / "20261122T094101Z-ad793d52/full/ball.payload", BALL_SHA),
    ]:
        d["staged_inputs"].append(dict(name=name, path=str(path), sha256=sha, extract=False))
        d["inputs"].append(dict(path=str(path), sha256=sha))
        d["arguments"] += ["--" + name.replace("_", "-"), "{staged:" + name + "}"]
    d["smoke"]["representative_basis"] = (
        "Both source player masks and paid ball positions on 24 real native frames, genuine +40 outside ROI map, byte-only warm-cap bisection, same-QP ordinary twin and decoded luma difference on any affordable result."
    )
    with out.open("x") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    build(a.base, a.out)
