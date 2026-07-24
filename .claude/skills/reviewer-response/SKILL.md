---
name: reviewer-response
description: Work a reviewer-driven item for the POINTSTREAM ACM TOMM resubmission — scope the required experiment/code/text change, implement it, and update the reviewer checklist. Use when the user references a reviewer comment, asks what's left for the TOMM revision, or wants to close a checklist item.
---

Read `/home/itec/emanuele/.agent-rules/skills/reviewer-response/SKILL.md` and follow it.

POINTSTREAM specifics (all under `67a9ea6275d3d9785ce57026/`):

- Raw reviews: `reviews.md` — the 5 ACM MM reviews that led to rejection;
  resubmission target is ACM TOMM.
- Authoritative checklist: `reviewers_comments.md` (Reviewers / Status /
  Evidence / Owed fields) — re-read it each time, it moves.
- Paper source: `main.tex` (+ `ref.bib`, `figures/`). Fresh submission, no
  `\rev{}`/`\del{}` macros — edit directly. `main_old.tex` and `backup/` are
  reference only, never edit or delete them, never copy a result across
  without re-deriving it from an `outputs/` path.
- Evidence base: `RESEARCH_LOG.md`.
- Marker grep: `grep -n '^% *\(GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' main.tex` —
  several reviewer answers map onto `HOLE(sec:system-design)` and
  `HOLE(sec:evaluation)`.
- Work classification routes to: experiment → `/run-pipeline` +
  `/results-report`; code change → normal repo rules (ruff/mypy/pytest +
  real-input verification, commit before long runs); paper text →
  `paper-editor` agent.
- Fold evidence in via `/update-paper`, clearing the `HOLE` and writing the
  `CLAIM` line in the same edit.
