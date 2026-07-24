---
name: update-paper
description: Fold new POINTSTREAM findings (run results, diagnoses, retractions, dead ends) into the paper (67a9ea6275d3d9785ce57026/main.tex) guided by its GOAL/HOLE/CLAIM markers, and into RESEARCH_LOG.md. Use after runs complete and results are committed and tested, or when a conclusion changes. Replaces the retired update-reports workflow.
---

Read `/home/itec/emanuele/.agent-rules/skills/update-paper/SKILL.md` and follow it.

POINTSTREAM specifics:

- Paper repo: `67a9ea6275d3d9785ce57026/` (nested Overleaf-synced git repo,
  own `CLAUDE.md`), commit it separately from the code repo.
- Manuscript: `67a9ea6275d3d9785ce57026/main.tex`. Marker grep:
  `grep -n '^% *\(STATUS\|GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' main.tex`.
- Research log: `67a9ea6275d3d9785ce57026/RESEARCH_LOG.md` (standing results,
  bug/dead-end/superseded registries).
- Reviewer checklist: `67a9ea6275d3d9785ce57026/reviewers_comments.md`.
- Citability check: `invariant_failures` in `run_summary.json`. A missing key
  predates the check and reads as unverified, not clean — backfill with
  `python -m src.shared.invariants outputs/`.
- Name the config that produced a number: **default config caps at
  `num-frames: 10`** — a 10-frame smoke result must be labeled as one, never
  presented as a full run.
- Invalidated runs move to `outputs/_superseded/<timestamp>_<reason>/` via
  `mv`, never `rm` (the guard-rm hook blocks it anyway).
- Fresh TOMM submission after the ACM MM rejection — no `\rev{}`/`\del{}`
  revision macros, edit text directly.
- Architecture doc to update on a contract change: `ARCHITECTURE.md` (CI
  checks it lists every module, not that the prose is true).
- No local TeX: verify structurally (balanced braces/environments), let
  Overleaf compile after push.
