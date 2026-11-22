"""Compare physical Shanghai deliveries under each exact cached and cold byte ceiling."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text())


def endpoints(path, sender):
    d = load(path)
    assert not d["qa_only"] and d["source_sha256"] == sender["source_sha256"]
    rows = d["paired_ticks"]
    assert len(set(r["tick"] for r in rows)) == len(rows) > 0
    for key in ("players", "ball", "foreground"):
        assert abs(mean(r[key] for r in rows) - d["mean_paired"][key]) < 1e-9
    assert (
        abs(
            (d["mean_paired"]["players"] + d["mean_paired"]["ball"]) / 2
            - d["mean_paired"]["foreground"]
        )
        < 1e-9
    )
    common = (
        d["review_sha256"],
        d["annotation_sha256"],
        d["player_support"],
        d["official_ball_sha256"],
        d["backbone_sha256"],
        tuple(r["tick"] for r in rows),
    )
    desc = dict(
        arm=d["arm"],
        bytes=d["bytes"],
        kbps=d["kbps"],
        sha256=d["bitstream_sha256"],
        scores=d["mean_paired"],
        receipt=path.name,
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return common, desc


def compare(args):
    sender = load(args.sender)
    payments = {str(r["crf"]): r for r in sender["rates"]}
    records = {}
    controls = {"ordinary": [], "roi": [], "neural": []}
    common = None
    for path in args.controls:
        instance, row = endpoints(path, sender)
        if common is None:
            common = instance
        assert instance == common
        words = row["arm"].lower().split()
        family = (
            "neural"
            if "dcvc-uf" in words
            else "roi"
            if "roi" in words
            else "ordinary"
            if "ordinary" in words
            else None
        )
        assert family in controls
        controls[family].append(row)
    for crf, filename in args.joint:
        assert crf not in records
        instance, joint = endpoints(Path(filename), sender)
        assert instance == common
        bill = payments[crf]
        assert joint["bytes"] == bill["joint_bytes"] and joint["sha256"] == bill["joint_sha256"]
        cases = {}
        for cache, extra in (("cached", 0), ("cold", bill["init_bytes"])):
            limit = joint["bytes"] + extra
            competitors = {}
            for name, bank in controls.items():
                feasible = [r for r in bank if r["bytes"] <= limit]
                if feasible:
                    best = min(feasible, key=lambda r: (r["scores"]["foreground"], r["bytes"]))
                    competitors[name] = dict(
                        best=best,
                        affordable=len(feasible),
                        relative_lower_dists=1
                        - joint["scores"]["foreground"] / best["scores"]["foreground"],
                    )
            cases[cache] = dict(bytes=limit, kbps=bill["kbps_" + cache], controls=competitors)
        records[crf] = dict(joint=joint, cases=cases)
    assert records and all(controls.values())
    return dict(
        source=sender["source_sha256"],
        player_support=common[2],
        annotation_sha256=common[1],
        ticks=common[-1],
        controls=controls,
        measured_rungs=records,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sender", type=Path, required=True)
    p.add_argument("--controls", nargs="+", type=Path, required=True)
    p.add_argument("--joint", nargs=2, action="append", metavar=("CRF", "RESULT"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = compare(args)
    with args.out.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
