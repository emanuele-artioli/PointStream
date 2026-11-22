"""Bind a Sydney control to its authenticated source, frozen 240-tick bank and measured ceilings."""

from argparse import ArgumentParser
import json
from pathlib import Path

INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")
PROVENANCE = {
    "roi_features": (
        "20261122T113333Z-9f0e3e92/full/published.tar",
        "7c7b678cf671e22092efe5284a19cd113bca49ee125af558d844c886fc31c19d",
    ),
    "ball_payload": (
        "20261122T102817Z-df6c7026/full/ball.payload",
        "d85b4b68e26d93d9b5e477785ed4b58c003aad044d9654aeaf74e52d49255757",
    ),
}


def create(a):
    base = json.loads(a.score_spec.read_text())
    inputs = [dict(i) for i in base["staged_inputs"] if i["name"] in ("source", "metadata", "rate")]
    bank = next(i for i in inputs if i["name"] == "rate")
    bank["name"] = "earlier"
    assert (
        next(i["sha256"] for i in inputs if i["name"] == "source")
        == "8ebca8b693202d8bf7afaee515dde4d723a9fb83250115e6eff080038b0bbf50"
    )
    labels = []
    limits = []
    for v in a.ceilings:
        label, amount = v.split("=")
        assert label.isalnum() and int(amount) > 0 and label not in labels
        labels.append(label)
        limits += ["--ceiling", v]
    assert labels and set(a.roi_ceiling or []).issubset(labels)
    if a.roi_ceiling:
        for name, (path, sha) in PROVENANCE.items():
            inputs.append(dict(name=name, path=str(INBOX / path), sha256=sha, extract=False))
    common = ["--match", "match127", "--fps", "60", *limits]
    common.extend(["--cadence-stride", str(a.cadence_stride)])
    if a.roi_ceiling:
        common.extend(["--roi-offset", str(a.roi_offset)])
    for k in a.roi_ceiling or []:
        common += ["--roi-ceiling", k]
    module = "experiments.tennis.test102_spatial_anchors"
    args = [
        v
        for i in inputs
        for v in ("--" + i["name"].replace("_", "-"), "{staged:" + i["name"] + "}")
    ]
    spec = dict(
        base,
        entrypoint=["-m", module],
        arguments=args
        + [
            "--earlier-sha",
            bank["sha256"],
            "--width",
            str(a.width),
            "--frames",
            "{frames}",
            *common,
        ],
        validator=["{python}", "-m", module, "--validate", *common],
        hosts=["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
        staged_inputs=inputs,
        inputs=[{k: i[k] for k in ("path", "sha256")} for i in inputs],
        smoke=dict(
            seconds=500,
            representative_basis="First 24 native Sydney ticks, actual paid byte ceilings, native SVT/dav1d and full hashes. An ROI search also verifies both source identities, the paid ball map, and its actual ordinary twin.",
        ),
        full=dict(seconds=2100),
        validator_seconds=160,
        budget_seconds=2900,
        stall_seconds=2160,
    )
    with a.out.open("x") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("score_spec", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("width", type=int, choices=[160, 192, 256, 320, 480, 640, 960, 1920])
    p.add_argument("ceilings", nargs="+")
    p.add_argument("--roi-ceiling", action="append")
    p.add_argument("--roi-offset", type=int, choices=range(1, 41), default=40)
    p.add_argument("--cadence-stride", choices=(1, 2), type=int, default=1)
    create(p.parse_args())
