"""Price native warm player performance at equal quality on genuinely measured anchor curves."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def estimate(a):
    sys.path.insert(0, str(HERE.parent / "med-mixed-receiver-20261122"))
    from rate_summary import summarize
    from replicate import reuse_threshold

    grid = json.loads(a.comparisons.read_text())["results"]
    fresh = json.loads((HERE / "fresh-prices.json").read_text())
    init = {r["init_bytes"] for r in fresh["rates"]}
    assert len(init) == 1
    rows = [r for name, r in grid.items() if name.startswith("fresh-joint")]
    assert len(rows) == 3
    anchors = {
        "ordinary": [r for n, r in grid.items() if n.startswith("ordinary")],
        "neural": [r for n, r in grid.items() if n.startswith("dcvc")],
    }
    output = {
        "source_sha256": fresh["source_sha256"],
        "support": "480 source-model body crops in the second Alcaraz-Hurkacz shot",
        "break_even_if_unchanged": {
            r["arm"]["name"]: {
                mode: reuse_threshold(r, bank, next(iter(init))) for mode, bank in anchors.items()
            }
            for r in rows
        },
        "equal_quality_band": summarize(anchors, rows),
    }
    a.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--comparisons", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    estimate(cli.parse_args())
