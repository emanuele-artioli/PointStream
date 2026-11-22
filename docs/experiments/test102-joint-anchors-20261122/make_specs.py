"""Bind joint budget controls to the verified Shanghai source and existing native bank."""

import argparse
import json
from pathlib import Path


def create(source_spec, output):
    base = json.loads(source_spec.read_text())
    inputs = [x for x in base["staged_inputs"] if x["name"] in ("source", "metadata", "rate")]
    for r in inputs:
        if r["name"] == "rate":
            r["name"] = "earlier"
            earlier = r["sha256"]
    module = "experiments.tennis.test102_spatial_anchors"
    for width in (320, 480, 640, 960, 1920):
        args = [v for i in inputs for v in ("--" + i["name"], "{staged:" + i["name"] + "}")]
        spec = dict(
            base,
            entrypoint=["-m", module],
            arguments=args
            + ["--earlier-sha", earlier, "--width", str(width), "--frames", "{frames}"],
            validator=["{python}", "-m", module, "--validate"],
            hosts=["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
            staged_inputs=inputs,
            inputs=[{k: i[k] for k in ("path", "sha256")} for i in inputs],
            smoke=dict(
                seconds=480,
                representative_basis="First 24 original native ticks, two complete target ceilings, physical CRF byte bisection with genuine SVT encoder, dav1d, and YUV bitwise receipt.",
            ),
            full=dict(seconds=2100),
            validator_seconds=160,
            budget_seconds=2900,
            stall_seconds=2160,
        )
        with (output / ("match102-anchor" + str(width) + ".json")).open("x") as f:
            json.dump(spec, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_spec", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    create(a.source_spec, a.output)
