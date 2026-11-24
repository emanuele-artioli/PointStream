"""Apply the frozen joint spatial suite to three verified Shanghai controls."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "joint-metric-crosscheck-20261123"))
from score import CASES, run  # noqa: E402


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "ball", "arms", "alex", "receipts", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    controls = ("ordinary-1920-r90", "ordinary-1920-r120", "foreground-roi-640-r120")
    CASES["shanghai-wider-controls"] = {
        "joint": controls[0],
        "arms": {name: name for name in controls},
        "scope": "Three Shanghai higher-rate control streams scored on the exact twelve frozen source instants.",
    }
    run(SimpleNamespace(**vars(args), case="shanghai-wider-controls", ticks=12))
