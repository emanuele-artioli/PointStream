#!/usr/bin/env python3
"""Stop hook: notice when runs have outrun the paper.

The paper is the primary living document, but folding results into it is a
separate step that is easy to postpone until the provenance of a number is
gone. This compares the newest run under `outputs/` against the paper repo's
last commit and says so if the runs are ahead.

Advisory only, and deliberately so: plenty of sessions legitimately produce
runs not worth writing up yet. It never blocks.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
PAPER = REPO / "67a9ea6275d3d9785ce57026"


def newest_mtime(root: Path, limit: int = 4000) -> float:
    """Newest mtime among run summaries, capped so this stays instant."""
    newest = 0.0
    for count, path in enumerate(root.glob("*/run_summary.json")):
        if count >= limit:
            break
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def paper_last_commit() -> float:
    try:
        out = subprocess.run(
            ["git", "-C", str(PAPER), "log", "-1", "--format=%ct"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def main():
    if not OUTPUTS.is_dir() or not PAPER.is_dir():
        return

    runs_at = newest_mtime(OUTPUTS)
    paper_at = paper_last_commit()
    if not runs_at or not paper_at or runs_at <= paper_at:
        return

    hours = (runs_at - paper_at) / 3600
    age = f"{hours:.0f}h" if hours >= 1 else f"{hours * 60:.0f}min"
    print(
        f"Runs are {age} newer than the paper's last commit. "
        f"If this session produced something citable, run /update-paper; "
        f"if not, no action needed.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
