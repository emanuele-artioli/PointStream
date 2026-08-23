# Workstream briefs

**A session reads `AGENTS.md`, `PLAN.md`, and exactly one of these.** Files not
listed under a brief's "owns" belong to another stream: if you need a change
there, say so in your report rather than making it.

## Read this before anything else

Two things are true and both are load-bearing:

1. **No generative engine here produces a usable player.** Re-ranked in clip
   mode on calibrated LPIPS over 12 clips (`PLAN.md` §2.10): **every one of the
   eight loses to pasting the keyframe**, at 2.5σ–10.6σ, and the best of them is
   `upscale-refine`, which is not a generative model. The top three are not
   separable from each other. Any plan that assumes a working generator is stale.

   **The cross-appearance test does not rescue this, and is itself withdrawn**
   as a test of whether an engine uses appearance: a pasted keyframe tops that
   scale with no network at all, and would have passed BP10's "≥ +3 dB means
   ReferenceNet works" gate at +4.45 dB. It is kept only as a measure of
   *dependence on the reference*. Deciding whether any engine renders the right
   *person* needs an identity metric, which does not exist here yet.
2. **One appearance channel is switched off, not missing.** The ControlNets
   were trained with per-track captions naming kit colour, and inference
   hardcodes a generic fallback (`PLAN.md` §2.11). Of three registered
   appearance pathways, one is off, one failed for a known architectural reason,
   and one was never trained. "The generators do not use appearance" is too
   coarse a summary to plan from.
3. **Two of the three metrics were broken until 2026-08-23** — LPIPS had no
   dynamic range, VMAF had its inputs crossed. **Every engine ranking taken
   before that date is void**, including the roster verdicts in `done/`. Metrics
   now have calibration invariants (`tests/invariants/test_metric_calibration.py`)
   and comparisons carry n and standard error
   (`src.components.metrics.comparison`).

**Before reporting any measurement, use the `verify-measurement` skill.**

## Live

**`WAVE-2026-08-24.md` says which of these run together and in what order.**
Read it before picking one up; the waves exist so parallel sessions do not
collide, and Wave 3 forks on BP13's number.

| Brief | Owns | Wave |
|---|---|---|
| `BP16-ci-signal.md` | un-red the CI so regressions are visible again | **1** |
| `BP13-motivating-headroom.md` | FG + BG headroom; rewrite the motivating example | **1 — the one that can change the plan** |
| `BP18-appearance-identity-metric.md` | an instrument that separates "the output moved" from "the right body appeared" | **1** |
| `BP17-caption-channel.md` | drive the trained-but-disabled caption channel | 2 |
| `BP15-test-cull.md` | retire the pre-rewrite tree and its 433 tests | 2 — **after** BP16, never folded into it |
| `BP14-training-stop-rule.md` | stop a run that cannot clear the bar | before any training, in any wave |
| `DEFERRED.md` | — | real work deliberately not now |

## Done

`done/` holds finished briefs, each ending with a *Delivered* section. History,
not instructions — `done/README.md` indexes them.

**`BP12-clip-mode-roster.md` is done** (2026-08-23) and still sits here rather
than in `done/` because its closing section is the current roster verdict.
**`BP10-appearance-pathway.md` is void**: its bands classify a paste as a
working ReferenceNet.

**`done/BP5-roster-decision.md`'s roster verdict is void** (measured on
self-reconstruction), and so is anything in `done/` that ranked engines on LPIPS
or VMAF before 2026-08-23.
