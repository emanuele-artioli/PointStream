"""Intersect the archived executable point router and human court-shot labels with the tested passage."""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
START, GAP, RETURN, END = 76626, 76866, 87691, 87931


def read(archive, name, bindings):
    data = archive.extractfile("Datasets/" + name).read()
    if hashlib.sha256(data).hexdigest() != bindings[name]:
        raise ValueError("Archive member identity changed")
    return json.loads(data)


def main(path):
    provenance = json.loads((HERE.parent / "continuous-return-20261118/results.json").read_text())
    digests = {Path(k).name: v for k, v in provenance["sha256"].items()}
    with tarfile.open(path) as archive:
        spans = read(archive, "whole-motion-spans.json", digests)
        marks = read(archive, "k8-marks-fb6cfe8a81f9a440.json", digests)["marks"]
        shots = read(archive, "k8-shot-bounds.json", digests)
    phases = [("first", START, GAP), ("gap", GAP, RETURN), ("return", RETURN, END)]
    motion = []
    for r in spans:
        # Frame t is displayed at (t+1)/50 on this original broadcast.
        lo = int((Decimal(str(r["t_start"])) * 50).to_integral_value(rounding="ROUND_CEILING")) - 1
        hi = int((Decimal(str(r["t_end"])) * 50).to_integral_value(rounding="ROUND_CEILING")) - 1
        if hi > START and lo < END:
            motion.append(dict(start=max(lo, START), end=min(hi, END), route=r["scene_class"]))
    assert motion[0]["start"] == START and motion[-1]["end"] == END
    assert all(x["end"] == y["start"] for x, y in zip(motion, motion[1:]))
    human = [
        dict(
            start=max(s["start"], START),
            end=min(s["end"], END),
            id=s["id"],
            label=marks.get(s["id"], "unreviewed"),
        )
        for s in shots
        if s["start"] < END and s["end"] > START
    ]
    assert human[0]["start"] == START and human[-1]["end"] == END
    assert all(x["end"] == y["start"] for x, y in zip(human, human[1:]))
    data = {}
    for key, a, b in phases:
        route, review, cross = Counter(), Counter(), Counter()
        for r in motion:
            t = max(0, min(b, r["end"]) - max(a, r["start"]))
            if t:
                route[r["route"]] += t
        for h in human:
            t = max(0, min(b, h["end"]) - max(a, h["start"]))
            if t:
                review[h["label"]] += t
                for r in motion:
                    k = max(0, min(b, h["end"], r["end"]) - max(a, h["start"], r["start"]))
                    if k:
                        cross[r["route"] + " / " + h["label"]] += k
        assert sum(route.values()) == sum(review.values()) == sum(cross.values()) == b - a
        data[key] = dict(frames=b - a, motion=route, human_court_class=review, overlap=cross)
    result = dict(
        source_sha256="676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66",
        fps=50,
        provenance={
            k: digests[k]
            for k in [
                "whole-motion-spans.json",
                "k8-shot-bounds.json",
                "k8-marks-fb6cfe8a81f9a440.json",
            ]
        },
        motion_revision=provenance["revisions"]["motion"],
        phases=data,
        motion=motion,
        human=human,
    )
    (HERE / "routing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    main(parser.parse_args().archive)
