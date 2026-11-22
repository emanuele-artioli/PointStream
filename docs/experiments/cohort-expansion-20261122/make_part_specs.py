"""Build an explicitly checkpointed source retry after the whole-stage cost gate fails."""

import argparse, json, hashlib
from pathlib import Path


def build(a):
    prev = json.loads(a.template.read_text())
    base = [
        x
        for x in prev["staged_inputs"]
        if x["name"] in ("source", "metadata", "archive", "weights")
    ]
    module = (
        "experiments.tennis.cohort_ball_sender"
        if a.action == "assemble"
        else "experiments.tennis.cohort_ball_part"
    )
    extra = []
    if a.action == "assemble":
        for key, first in [("a", 0), ("b", 120)]:
            state = json.loads(getattr(a, "status_" + key).read_text())
            receipt = getattr(a, "part_" + key)
            r = json.loads(receipt.read_text())
            assert (
                state["status"] == "complete"
                and state["gate"]["passed"]
                and r["start"] == first
                and r["stop"] == first + 120
            )
            assert r["source_sha256"] == next(i["sha256"] for i in base if i["name"] == "source")
            base.append(
                dict(
                    name="part_" + key,
                    path=state["directory"] + "/full/part.json",
                    sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    extract=False,
                )
            )
        smoke_size = 24
        full_size = 240
    else:
        first = int(a.action)
        extra = ["--first", str(first)]
        smoke_size = 12
        full_size = 120
    args = [
        v for i in base for v in ("--" + i["name"].replace("_", "-"), "{staged:" + i["name"] + "}")
    ]
    r = dict(
        prev,
        entrypoint=["-m", module],
        arguments=args + extra + ["--frames", "{frames}"],
        validator=["{python}", "-m", module, "--validate"],
        hosts=[a.host],
        inputs=[{k: i[k] for k in ("path", "sha256")} for i in base],
        staged_inputs=base,
        scale=dict(frames=dict(smoke=smoke_size, full=full_size)),
        smoke=dict(
            seconds=450,
            representative_basis="Same sender, fixed source/weights; actual 240-frame median and three-frame lookback, repeated split equality for a real 12-tick global interval; complete real 24-tick native fit and pack for assembly.",
        ),
        full=dict(seconds=2400),
        budget_seconds=3000,
        stall_seconds=2430,
    )
    with a.out.open("x") as f:
        json.dump(r, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("0", "120", "assemble"))
    p.add_argument("template", type=Path)
    p.add_argument("host")
    p.add_argument("out", type=Path)
    for k in ("status-a", "part-a", "status-b", "part-b"):
        p.add_argument("--" + k, type=Path)
    build(p.parse_args())
