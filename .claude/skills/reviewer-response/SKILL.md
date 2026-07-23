---
name: reviewer-response
description: Work a reviewer-driven item for the POINTSTREAM ACM TOMM resubmission — scope the required experiment/code/text change, implement it, and update the reviewer checklist. Use when the user references a reviewer comment, asks what's left for the TOMM revision, or wants to close a checklist item.
---

# Working the TOMM reviewer-response checklist

## The paper trail (all in `67a9ea6275d3d9785ce57026/`)

- **Raw reviews:** `reviews.md` — the 5 ACM MM reviews that led to rejection;
  the resubmission target is ACM TOMM.
- **Authoritative checklist:** `reviewers_comments.md` — one entry per theme
  with Reviewers / Status / Evidence / Owed. This is what gets updated when an
  item advances. **Done means the text or experiment is actually in place**,
  never a plan — the previous action matrix drifted precisely because it
  recorded plans and invalidated runs as Done.
- **Paper source:** `main.tex` (+ `ref.bib`, `figures/`), a separate nested git
  repo (Overleaf sync). Fresh submission, not a tracked revision: no
  `\rev{}`/`\del{}` macros, edit directly. `main_old.tex` and `backup/` are
  reference only.
- **Evidence base:** `RESEARCH_LOG.md` — hard rules, standing results with real
  numbers, dead-end and superseded registries.

## Where the leverage is (2026-07-21 — re-read `reviewers_comments.md`, it moves)

- **Cheapest wins, no new experiment needed:** dynamic backgrounds/cuts (R2 —
  fully solved in `src/shared/scene_classification.py`, just unwritten) and
  shadow handling (R3 — answered by the Residual Guarantee, needs a paragraph).
  Both are blocked only on the missing System Design section.
- **Blocked on the generative engine:** generative-model choice (R1/R5),
  temporal coherence proof (R3/R4), demo video (R3). The G2 campaign never
  completed a rung and no architecture has been selected on evidence.
- **Blocked on one unrun experiment:** missing baselines (R2/R5) — the anchors
  exist, PointStream has never been placed on the curve.
- **Untouched, high effort:** second domain (R2/R3/R4/R5 — the most-requested
  single item), MOS study (R2/R4), VVC anchor, SAM2 comparison (R3).

## Workflow

1. Read the specific review section in `reviews.md` and the matching entry in
   `reviewers_comments.md` — its "Owed" line is the scoped task.
2. Check the paper's markers for where the work lands:
   `grep -n '^% *\(GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' main.tex`.
   Several reviewer answers map onto `HOLE(sec:system-design)` and
   `HOLE(sec:evaluation)`, which are whole missing sections.
3. Classify the work: **experiment** (→ `/run-pipeline`, results via
   `/results-report`), **code change** (normal repo rules: ruff/mypy/pytest +
   real-input verification, commit before long runs), or **paper text** (→
   `paper-editor` agent for substantive edits).
4. For paper claims about the implementation, verify against `src/` and
   `outputs/` before writing — and check `RESEARCH_LOG.md`'s superseded
   registry, because more than one previously-cited result has been retracted.
5. When done, update `reviewers_comments.md` in the same pass: Status plus one
   concrete Resolution line (section added, table added, `outputs/<ts>` run).
6. Fold new evidence in via `/update-paper` — clearing the `HOLE` and writing
   the `CLAIM` line in the same edit.
