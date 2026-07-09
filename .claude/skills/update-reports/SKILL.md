---
name: update-reports
description: Fold new POINTSTREAM findings (run results, diagnoses, fixes, dead ends, decisions) into reports/ and keep the REPORTS.md dashboard current. Use after running experiments, fixing a pipeline issue, or discovering anything a future session or the paper should know.
---

# Updating the POINTSTREAM reports

`reports/REPORTS.md` is the index and dashboard; the numbered reports hold
the detail. Reports 1–7 are the Antigravity-era base (plans/research notes,
each with its own internal structure — extend them in their own style);
reports 8+ follow the standard format defined in REPORTS.md.

## Which report owns what

| Finding about… | Goes to |
|---|---|
| Scene classification, cut/pan detection, racket/wrist heuristics | `2_scene_classification_research.md` |
| Shared-library architecture, imports, dataset pipeline, compat patches | `3_architecture_refactor_research.md` |
| SPADE4Tennis design/training | `1_spade4tennis_plan.md` |
| Animate-Anyone integration, universal dataset format | `4_animate_anyone_integration_research.md` |
| ControlNet conditioning, temporal consistency | `5_genai_temporal_consistency_research.md` |
| Reviewer-facing status changes | `6_action_matrix.md` (status column + checklist) |
| Strategy/paradigm/paper-structure changes | `7_implementation_plan.md` |
| A new area none of the above owns | New `8_<topic>_report.md` per the REPORTS.md template |

When unsure, prefer a new standard-format report over bolting a findings log
onto a legacy plan document.

## Update procedure

1. Append a dated entry to the owning report (for 8+ reports: the
   `### YYYY-MM-DD — <title>` / Problem / Diagnosis-Evidence / Resolution /
   Paper-impact format; for legacy reports: a dated section in their style).
2. Refresh that report's TL;DR/summary so a fresh reader gets current
   conclusions without replaying the log.
3. Refresh `REPORTS.md`: the workstream-status row(s) affected, the
   prioritized next steps (strike completed items with `~~…~~ **done
   (date)**`), and the `*Last updated:*` date.
4. If a previous conclusion is now wrong, mark it in place with
   `**Superseded YYYY-MM-DD:** see <newer entry>` — never rewrite or delete
   history.
5. If run outputs are invalidated (bug in the run, wrong config), move them
   to `outputs/_superseded/<timestamp>_<reason>/` with `mv` — never `rm`.
6. If the finding changes a working convention (how to run, a pin, a
   gotcha), also update CLAUDE.md / the relevant skill, and keep
   `.agents/rules/pointstream.md` + `.github/instructions/…` in sync.

## Rules of evidence

- Every entry cites real numbers and paths: an `outputs/<ts>/` dir,
  `run_summary.json` sizes/timings, a test name, a commit. No numbers, no
  claim.
- Distinguish single-run observations from swept/confirmed results — say
  which one an entry is.
- Note the config that produced a number (`num-frames`, backend, codec
  settings); a 10-frame smoke number must be labeled as such.
