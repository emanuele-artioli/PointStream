"""SessionStart hook: re-ground a session in live state.

After a compaction (source "compact") it prints the branch, uncommitted
files, recent commits and the ledger check, so the session works from the
repository's real state rather than from the summary alone. On startup and
resume it prints only ledger problems, and nothing when there are none.
Output is capped; stdout becomes session context.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.agent_hooks.ledger import git, report

MAX_FILES = 10

AFTER_COMPACT = (
    "Before acting on a fleet job, re-read its ID from your experiments entry and run "
    "`scripts/ps-fleet status JOB_ID`; never resubmit a job the summary mentions."
)


def git_state(root: Path) -> str:
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip() or "unknown"
    upstream = git(root, "rev-list", "--left-right", "--count", "@{upstream}...HEAD").split()
    lines = [f"Branch: {branch}" + (f" (behind {upstream[0]}, ahead {upstream[1]} of upstream)" if len(upstream) == 2 else "")]
    changed = [line for line in git(root, "status", "--porcelain").splitlines() if line.strip()]
    if changed:
        shown = "\n".join(f"  {line}" for line in changed[:MAX_FILES])
        more = f"\n  … and {len(changed) - MAX_FILES} more" if len(changed) > MAX_FILES else ""
        lines.append(f"Uncommitted files ({len(changed)}):\n{shown}{more}")
    else:
        lines.append("Working tree clean.")
    log = git(root, "log", "--oneline", "-n", "3").strip()
    if log:
        lines.append("Last commits:\n" + "\n".join(f"  {line}" for line in log.splitlines()))
    return "\n".join(lines)


def build(source: str, root: Path, fetch: bool = True) -> str:
    ledger = report(root, fetch=fetch)
    if source != "compact":
        return ledger
    parts = ["State after compaction (read live from git):", git_state(root), AFTER_COMPACT]
    if ledger:
        parts.append(ledger)
    return "\n".join(parts)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        payload = {}
    text = build(str(payload.get("source", "")), Path.cwd())
    if text:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
