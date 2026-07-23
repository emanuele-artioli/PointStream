---
name: update-paper
description: Fold new POINTSTREAM findings (run results, diagnoses, retractions, dead ends) into the paper (67a9ea6275d3d9785ce57026/main.tex) guided by its GOAL/HOLE/CLAIM markers, and into RESEARCH_LOG.md. Use after runs complete and results are committed and tested, or when a conclusion changes. Replaces the retired update-reports workflow.
---

# Folding findings into the POINTSTREAM paper

The paper is the primary living document. The `reports/` tree and its
`REPORTS.md` dashboard were consolidated into the paper and `RESEARCH_LOG.md`
on 2026-07-21 and deleted — **do not recreate them**.

Two destinations, and the split matters:

- **`67a9ea6275d3d9785ce57026/main.tex`** — anything a reader of the paper needs.
  Guided by the `STATUS/GOAL/HOLE/NOTE/NEXT/CLAIM(anchor):` comment markers
  (spec in the paper repo's own CLAUDE.md).
- **`67a9ea6275d3d9785ce57026/RESEARCH_LOG.md`** — everything that cannot live
  in the paper: hard methodology rules, standing results with their numbers,
  codec anchors, the bug registry, and the dead-end and superseded registries.

## Before writing anything

**Check `invariant_failures` on every run you are about to cite.** A run with a
non-empty list is not citable — it fell back to a mock source, quality
evaluation did not complete, the payload accounting does not add up, or the
payload came out larger than the source. A run with *no* verdict has never been
checked; backfill first with `python -m src.shared.invariants outputs/`, because
a missing verdict reads as clean.

Then read the dead-end registry in `RESEARCH_LOG.md`. It exists to stop us
re-landing a conclusion that has already been disproved.

## Procedure

1. **Find what the paper is waiting for:**
   ```
   grep -n '^% *\(STATUS\|GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' main.tex
   ```
   A `HOLE(id)` names the exact data an element is missing. Only land a number
   where the paper actually has a hole for it.

2. **Write the text**, and in the same edit clear the `HOLE` and add the
   provenance line: `CLAIM(id): src=outputs/<timestamp> date=YYYY-MM-DD`. A
   `HOLE` may only be cleared by the edit that lands its data — never in advance.

3. **Update `RESEARCH_LOG.md`:** add the finding to standing results, or the
   dead-end registry if something was disproved. If a queued result just landed
   in the paper text, delete it from the queue.

4. **Cross-update the reviewer checklist** (`reviewers_comments.md`) if a
   referee item advanced. Status becomes Done only when the text or the
   experiment is actually in place, never on a plan.

5. **If a component's contract changed,** update `ARCHITECTURE.md` in the same
   session. CI checks that it lists every module, but not that the prose is true.

6. **Verify and commit.** No local TeX on this host, so verification is
   structural — balanced braces and environments in the edited file — plus
   Overleaf compiling after push. Commit the paper repo separately (it is a
   nested Overleaf repo), naming the anchor ids and the run timestamps.

## Rules of evidence

- Every number cites a real path: an `outputs/<timestamp>/` dir and its
  `run_summary.json`. No path, no claim.
- **Name the config that produced it** — `num-frames`, backend, codec settings.
  A 10-frame smoke number must be labelled as one; the default config caps at
  `num-frames: 10`, so this is the easiest way to overstate a result.
- Distinguish a single-run observation from a swept or confirmed result, in the
  text, not just in your head.
- A conclusion that is now wrong is marked superseded in place, never rewritten
  or deleted — the registry is what stops it being rediscovered.
- Invalidated run outputs move to `outputs/_superseded/<timestamp>_<reason>/`
  with `mv`. Never `rm` — the guard-rm hook will stop you anyway.
- This is a fresh TOMM submission after the ACM MM rejection, so there are no
  `\rev{}`/`\del{}` revision macros: edit the text directly.
