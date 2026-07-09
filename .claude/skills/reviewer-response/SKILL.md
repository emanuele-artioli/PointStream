---
name: reviewer-response
description: Work a reviewer-driven item for the POINTSTREAM ACM TOMM resubmission — scope the required experiment/code/text change, implement it, and update the action matrix. Use when the user references a reviewer comment, asks what's left for the TOMM revision, or wants to close an action-matrix item.
---

# Working the TOMM reviewer-response checklist

## The paper trail

- **Raw reviews:** `67a9ea6275d3d9785ce57026/reviews.md` — the 5 ACM MM
  reviews that led to rejection; the resubmission target is ACM TOMM.
- **Authoritative checklist:** `reports/6_action_matrix.md` — reviewer
  themes → status (🟢 tackled / 🟡 partial / 🔴 not) → action → effort, plus
  the two-phase execution checklist. This is what gets updated when an item
  closes.
- **Paper source:** `67a9ea6275d3d9785ce57026/main.tex` (+ `ref.bib`,
  `figures/`) — a separate nested git repo (Overleaf sync). This is a
  **fresh submission, not a tracked revision**: there are no `\rev{}`/`\del{}`
  macros; edit the text directly. `main_old.tex` and `backup/` are reference
  material — don't edit or delete them.

## Current snapshot (2026-07-09 — re-read 6_action_matrix.md, it moves)

- 🟢 Tackled, needs paper text: generative-model choice (SPADE vs diffusion,
  R1/R5), temporal coherence mechanisms (R3/R4), dynamic backgrounds/cuts
  (R2).
- 🟡 Partial: temporal segmentation / SAM2 question (R3).
- 🔴 Open, needs experiments: AV1 + learned-codec baselines (R2/R5),
  generalizability beyond tennis (R2–R5), MOS study (R2/R4), shadow handling
  (R3 — answered conceptually by the Residual Guarantee, needs a paragraph).

## Workflow

1. Read the specific review section in `reviews.md` and the matching action
   matrix row — the matrix's "Action Required" is the scoped task.
2. Classify the work: **experiment** (→ `/run-pipeline`, results via
   `/results-report`), **code change** (normal repo rules, tests +
   real-input verification), or **paper text** (→ `paper-editor` agent for
   substantive edits).
3. For paper claims about the implementation, verify against `src/` and the
   `reports/` findings before writing — several reviewer answers (shadows,
   cuts, temporal coherence) are already substantiated in reports 2, 5, 7.
4. When done, update `reports/6_action_matrix.md` in the same pass: flip the
   status emoji, check the execution-checklist box, and add one line saying
   concretely what closed it (section added, table added, outputs/<ts> run).
5. Fold any new experimental evidence into `reports/` via `/update-reports`.
