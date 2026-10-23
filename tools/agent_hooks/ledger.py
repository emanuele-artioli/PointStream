"""Flag dispatch-ledger rows in PLAN.md that look stale.

A row is stale when its status still says "queued" but its branch already
has a merged pull request on origin/main. Prints nothing when every row is
consistent.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROW = re.compile(r"^\|\s*(?P<item>[^|]+?)\s*\|[^|]*\|\s*`?(?P<branch>[^|`]+?)`?\s*\|[^|]*\|\s*(?P<status>[^|]+?)\s*\|\s*$")
MERGE = re.compile(r"^Merge pull request #(?P<pr>\d+) from [^/]+/(?P<branch>.+)$")


@dataclass(frozen=True)
class Row:
    item: str
    branch: str
    status: str


def ledger_rows(plan_text: str) -> list[Row]:
    """Rows of the table under the "Dispatch ledger" heading."""
    rows: list[Row] = []
    in_ledger = False
    for line in plan_text.splitlines():
        if line.startswith("#"):
            in_ledger = "dispatch ledger" in line.lower()
            continue
        if not in_ledger:
            continue
        match = ROW.match(line)
        if match and not set(match["item"]) <= {"-"} and match["item"].lower() != "item":
            rows.append(Row(match["item"], match["branch"], match["status"]))
    return rows


def merged_prs(merge_subjects: list[str]) -> dict[str, list[int]]:
    """Branch name -> merged PR numbers, from merge-commit subjects."""
    merged: dict[str, list[int]] = {}
    for subject in merge_subjects:
        match = MERGE.match(subject.strip())
        if match:
            merged.setdefault(match["branch"], []).append(int(match["pr"]))
    return merged


def stale_rows(rows: list[Row], merged: dict[str, list[int]]) -> list[str]:
    out = []
    for row in rows:
        prs = merged.get(row.branch)
        if prs and row.status.lower().startswith("queued"):
            out.append(
                f"- {row.item} (`{row.branch}`): status '{row.status}', but PR #{max(prs)} "
                "from this branch is merged. Update the row."
            )
    return out


def git(root: Path, *args: str, timeout: float = 15) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


def report(root: Path, fetch: bool = True) -> str:
    plan = root / "PLAN.md"
    if not plan.is_file():
        return ""
    if fetch:
        git(root, "fetch", "-q", "origin", "main")
    subjects = git(root, "log", "origin/main", "--merges", "--format=%s", "-n", "500").splitlines()
    stale = stale_rows(ledger_rows(plan.read_text(encoding="utf-8")), merged_prs(subjects))
    if not stale:
        return ""
    return "Dispatch ledger in PLAN.md may be stale (root session owns it):\n" + "\n".join(stale)


def main() -> int:
    text = report(Path.cwd())
    if text:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
