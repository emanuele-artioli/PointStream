---
name: paper-editor
description: Edits main.tex in the POINTSTREAM paper repo (67a9ea6275d3d9785ce57026/), guided by its GOAL/HOLE/CLAIM markers, keeping claims consistent with the actual src/ implementation and outputs/ evidence, and updating reviewers_comments.md when an edit closes a reviewer item. Use for any substantive edit to the paper text, not just typo fixes.
tools: Read, Edit, Grep, Glob, Bash
model: sonnet
---

Read `/home/itec/emanuele/.agent-rules/agents/paper-editor.agent.md` and follow it.

POINTSTREAM specifics:

- Paper repo: `67a9ea6275d3d9785ce57026/main.tex` (+ `ref.bib`, `figures/`),
  a separate nested git repo (Overleaf sync) one directory up from the code
  repo, with its own `CLAUDE.md`.
- **Fresh ACM TOMM submission after an ACM MM rejection, not a tracked
  revision** — no `\rev{}`/`\del{}` macros, edit text directly.
  `main_old.tex` and `backup/` are historical reference only: never edit or
  delete them, and never copy a result across from `main_old.tex` without
  re-deriving it from a real `outputs/<ts>/run_summary.json`.
- Research log: `67a9ea6275d3d9785ce57026/RESEARCH_LOG.md`. Reviewer
  checklist: `67a9ea6275d3d9785ce57026/reviewers_comments.md`. Raw reviews:
  `67a9ea6275d3d9785ce57026/reviews.md`.
- Marker grep: `grep -n '^% *\(STATUS\|GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' main.tex`.
- Verify implementation claims against `../src/` and real
  `../outputs/<ts>/run_summary.json`, never from memory.
- If a component's contract changed, check `ARCHITECTURE.md` in the code
  repo — CI checks it lists every module.
- No local TeX on this host — verify structurally, let Overleaf compile.
