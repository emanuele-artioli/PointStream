# Workstream briefs

**A session reads `AGENTS.md`, `PLAN.md`, `plans/ROADMAP.md`, and exactly one
brief.** Files not listed under a brief's "owns" belong to another stream: if you
need a change there, say so in your report rather than making it.

- **`ROADMAP.md`** — what is left, in what order, with the dependency graph.
  **Start there.**
- **`FORK-bp31.md`** — the three papers, one per outcome of the run currently
  in flight, written before that run reports.
- **`DEFERRED.md`** — real work deliberately not now.

---

## Read this before anything else

Four things are true and all of them are load-bearing.

0. **PointStream loses to the codec it is built on.** BD-rate **+90.97%** against
   an av1 anchor at N=2 scenes with the cross-scene stream on
   (`plans/BP31-findings.md` §9), from +116.8% with a single-frame plate
   (`PLAN.md` §2.20). `AGENTS.md` requires the headline claim to land where
   PointStream wins, so **finding that regime is the work.** `FORK-bp31.md`
   holds the pre-written decision rule.

1. **The premise holds, but its codec-comparison leg does not** (`PLAN.md`
   §2.14, corrected at n=8 by `BP21`). Concentration survives: a player is ~1%
   of the pixels and carries a **18.9 ± 5.0x** concentration of bitrate at n=8
   against AVC (`done/BP21-headroom-widen.md`) — quote it with the standard
   error, and note that `PLAN.md` §2.14's headline "15–47x" is the superseded
   n=2 figure, which that section flags itself. The paper cites the n=8 run.
   **The VVC
   exception did not survive** — measured at n=8 the AVC−VVC gap is
   **+0.028 ± 0.015** at common QP (1.8σ). Do not name VVC as the exception; do
   not cite "17%" without its standard error and the two near-zero clips.

2. **No generative engine here produces a usable player.** Re-ranked in clip mode
   on calibrated LPIPS over 12 clips (`PLAN.md` §2.10): **every one of the eight
   loses to pasting the keyframe**, at 2.5σ–10.6σ, and the best of them is
   `upscale-refine`, which is not a generative model. Any plan that assumes a
   working generator is stale. **The cross-appearance test does not rescue this
   and is itself withdrawn** as a test of whether an engine uses appearance — a
   pasted keyframe tops that scale with no network at all.

3. **Two of the three metrics were broken until 2026-08-23** — LPIPS had no
   dynamic range, VMAF had its inputs crossed. **Every engine ranking taken
   before that date is void**, including the roster verdicts in `done/`. Metrics
   now have calibration invariants (`tests/invariants/test_metric_calibration.py`)
   and comparisons carry n and standard error
   (`src.components.metrics.comparison`).

**Before reporting any measurement, use the `verify-measurement` skill.**

---

## Live

Ordering, dependencies and file ownership are in **`ROADMAP.md`**. This table is
the index only.

| Brief | Owns | Wave |
|---|---|---|
| `BP31-paired-ladder-across-scenes.md` + `BP31-findings.md` | the paired ladder over N scenes | **running now**, another session, PR #45 |
| `BP32-rate-budget.md` ⭐ | where the bits go against where the headroom said they could | 9 — do first |
| `BP33-span-amortisation.md` ⭐ | span: the ladder runs 8 frames, the cache holds 48 | 9 — do first |
| `BP34-operating-point.md` | encode/decode time, the operating point, the title | 9 |
| `BP35-perceptual-bdrate.md` | BD-rate on VMAF and LPIPS | 9 |
| `BP36-second-domain.md` | DAVIS and UVG — P0 item 6 | 9 |
| `BP37-required-behaviour.md` | the gate `PLAN.md` §8 describes | 9 |
| `BP38-paper-infrastructure.md` | figures, a DCVC-class anchor, reproducibility, related work | 9 |
| `BP43-background-representation.md` ⭐ | a smaller plate in pixels, and whether it needs sending at all | 9 |
| `BP39-all-off-corner.md` | D5 — the lattice claim is a hardcoded branch | 10, after PR #45 |
| `BP40-background-honesty.md` | the background component's three open faults | 10, after PR #45 |
| `BP41-ablation-lattice.md` | P0 item 4 — the paper's central contribution | 10 |

## Parked, with the reason

Not dead, not scheduled. `ROADMAP.md` §5 carries the reasons in full.

| Brief | Parked because |
|---|---|
| `BP19-conditioning-architecture.md` | every engine loses to a paste; a training campaign costs the most and buys the least |
| `BP28-offset-crossover.md` | its crossover happens where both arms are as good as a photo of the wrong player; its useful half is folded into `BP41`'s temporal axis |

## Historical

`done/` holds finished briefs, each ending with a *Delivered* section — history,
not instructions. `done/README.md` indexes them. `wave3-report.md`,
`wave5-report.md` and the reports named `BP24-*`, `BP29-*`, `BP30-*` are records
of completed work and are cited from `PLAN.md`; they stay where they are so the
citations keep resolving.

**`done/BP5-roster-decision.md`'s roster verdict is void** (measured on
self-reconstruction), and so is anything in `done/` that ranked engines on LPIPS
or VMAF before 2026-08-23. **`BP10-appearance-pathway.md` is void**: its bands
classify a paste as a working ReferenceNet.

`prompts/` holds the paste-in prompts for sessions on other tools or in other
windows. Several are also the *record* of a wave's report and are cited from
code and findings — check for inbound references before deleting one.
