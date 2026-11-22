"""Count contiguous paired source tracks and separately show complete native shots."""

import argparse
import itertools
import json
from pathlib import Path


def summarize(paths):
    cases = []
    for path in paths:
        record = json.loads(path.read_text())
        assert record["frames"] == 240
        assert len(record["tracks"]) == 2
        missing = set()
        for track in record["tracks"].values():
            assert track["fitted"] + len(set(track["missing"])) == 240
            assert all(0 <= t < 240 for t in track["missing"])
            missing.update(track["missing"])
        support = [t not in missing for t in range(240)]
        intervals = []
        i = 0
        for value, group in itertools.groupby(support):
            length = len(list(group))
            if value:
                intervals.append([i, i + length])
            i += length
        assert i == 240 and sum(b - a for a, b in intervals) == sum(support)
        cases.append(
            dict(
                event=record["event"],
                source_sha256=record["source_sha256"],
                paired_ticks=sum(support),
                longest_native_run=max((b - a for a, b in intervals), default=0),
                eligible_halves=sum(all(support[t : t + 120]) for t in (0, 120)),
                complete=all(support),
                intervals=intervals,
            )
        )
    assert len({r["event"] for r in cases}) == len(cases)
    return dict(
        cases=cases,
        complete_shots=sum(r["complete"] for r in cases),
        frames=240 * len(cases),
        paired_ticks=sum(r["paired_ticks"] for r in cases),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("receipts", nargs="+", type=Path)
    a = p.parse_args()
    r = summarize(a.receipts)
    with a.out.open("x") as f:
        json.dump(r, f, indent=2)
        f.write("\n")
