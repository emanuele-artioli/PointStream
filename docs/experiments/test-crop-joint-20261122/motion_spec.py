"""Bind the source-verified, 4-way native motion calculation to its original Shanghai witness."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def make(job, witness_sha):
    assert len(witness_sha) == 64
    spec = json.loads((HERE / "match102-campus-score-calibration-rgb.json").read_text())
    anchors = json.loads((HERE.parent / "test102-joint-anchors-20261122/manifest.json").read_text())
    staged = [r for r in spec["staged_inputs"] if r["name"] in ("source", "features", "pixels")]
    staged += [dict(r, name="neural") for r in spec["staged_inputs"] if r["name"] == "data"]
    for mode, width in (("ordinary", 320), ("roi", 640)):
        row = next(r for r in anchors["rows"] if r["mode"] == mode and r["width"] == width)
        staged.append(
            dict(name=mode, path=row["archive"], sha256=row["archive_sha256"], extract=False)
        )
    staged.append(
        dict(
            name="witness",
            path=str(INBOX / job / "full/published.tar"),
            sha256=witness_sha,
            extract=True,
        )
    )
    module = "experiments.tennis.test_joint_native_motion"
    args = []
    for item in staged:
        key = item["name"]
        args += ["--" + key, "{staged:" + key + "}"]
    args += ["--frames", "{frames}"]
    spec.update(
        entrypoint=["-m", module],
        validator=["{python}", "-m", module, "--validate"],
        arguments=args,
        staged_inputs=staged,
        inputs=[{k: r[k] for k in ("path", "sha256")} for r in staged],
        smoke=dict(
            seconds=90,
            representative_basis="Five exact native frames and four authenticated decoded arms, repeated source flow and two matched athlete masks, estimated full CPU time under the ten-minute total budget.",
        ),
        full=dict(seconds=455),
        validator_seconds=20,
        budget_seconds=600,
        stall_seconds=480,
        scale=dict(frames=dict(smoke=5, full=240)),
    )
    (HERE / "match102-campus-motion.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--producer", required=True)
    p.add_argument("--sha", required=True)
    a = p.parse_args()
    make(a.producer, a.sha)
