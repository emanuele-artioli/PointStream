# Workstream briefs

**A session reads `AGENTS.md`, `PLAN.md`, and exactly one of these.** Files not
listed under a brief's "owns" belong to another stream: if you need a change
there, say so in your report rather than making it.

## Read this before anything else

Two things are true and both are load-bearing:

1. **No generative engine here produces a usable player yet.** The ControlNets
   were trained with no appearance input at all and cannot reproduce a specific
   person (`PLAN.md` §2.3). Animate-Anyone, correctly driven, reaches 0.570 LPIPS
   on the player region — against 0.582 for a static copy and 0.645 for an
   *unrelated image* (§2.7). Any plan that assumes a working generator is stale.
2. **Two of the three metrics were broken until 2026-08-23** — LPIPS had no
   dynamic range, VMAF had its inputs crossed. **Every engine ranking taken
   before that date is void**, including the roster verdicts in `done/`. Metrics
   now have calibration invariants (`tests/invariants/test_metric_calibration.py`)
   and comparisons carry n and standard error
   (`src.components.metrics.comparison`).

**Before reporting any measurement, use the `verify-measurement` skill.**

## Live

| Brief | Owns | Status |
|---|---|---|
| `BP12-clip-mode-roster.md` | re-rank every engine in clip mode on calibrated metrics | **critical path** |
| `BP13-motivating-headroom.md` | FG + BG headroom; rewrite the motivating example | high |
| `BP14-training-stop-rule.md` | stop a run that cannot clear the bar | before any training |
| `BP15-test-cull.md` | retire the pre-rewrite tree and its 433 tests | housekeeping |
| `DEFERRED.md` | — | real work deliberately not now |

## Done

`done/` holds finished briefs, each ending with a *Delivered* section. History,
not instructions — `done/README.md` indexes them.

**`done/BP5-roster-decision.md`'s roster verdict is void** (measured on
self-reconstruction), and so is anything in `done/` that ranked engines on LPIPS
or VMAF before 2026-08-23.
