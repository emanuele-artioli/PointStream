# Workstream briefs

**A session reads `AGENTS.md`, `PLAN.md`, and exactly one of these.** Files not
listed under a brief's "owns" belong to another stream: if you need a change
there, say so in your report rather than making it.

## Read this before anything else

Two things are true and both are load-bearing:

0. **The premise is measured and it holds** (`PLAN.md` §2.14): on real 4K a
   player is ~1% of the pixels and **17–24%** of the bitrate, and a panorama
   plate plus homographies saves **34–69%** of the background bitrate. AVC,
   HEVC and AV1 agree on the foreground; **VVC is ~0.077 lower and is the
   exception to name**, with a QP-47 confound still to rule out. Report every
   cell; claim no trend at n=2.
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
collide. The Wave-3 fork is now decided (`PLAN.md` §2.14).

| Brief | Owns | Wave |
|---|---|---|
| `BP15-test-cull.md` | finish retiring the pre-rewrite tree | partly done — decoder and `src/shared` still pinned by `eval_checkpoint` and training |
| `BP14-training-stop-rule.md` | stop a run that cannot clear the bar | before any training, in any wave |
| `BP19-conditioning-architecture.md` | IP-Adapter arm, retrain on the coding task, Uni-ControlNet shared backbone, ReferenceNet extension | **now the critical path** — headroom gate passed; still needs `BP14` first |
| `DEFERRED.md` | — | real work deliberately not now |

## Done

`done/` holds finished briefs, each ending with a *Delivered* section. History,
not instructions — `done/README.md` indexes them.

**`BP16`, `BP18`, `BP13`, `BP20`, `BP15` and `BP17` are done** (2026-08-23). `BP20` replaced `BP13`'s synthetic number with a real-4K one and **decided the fork**: the players are ~1% of pixels and 17–24% of the bitrate, so the premise holds (`PLAN.md` §2.14). `BP17` found the caption channel worth nothing measurable (§2.15).

**`BP12-clip-mode-roster.md` is done** (2026-08-23) and still sits here rather
than in `done/` because its closing section is the current roster verdict.
**`BP10-appearance-pathway.md` is void**: its bands classify a paste as a
working ReferenceNet.

**`done/BP5-roster-decision.md`'s roster verdict is void** (measured on
self-reconstruction), and so is anything in `done/` that ranked engines on LPIPS
or VMAF before 2026-08-23.
