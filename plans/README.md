# Workstream briefs

**A session reads `AGENTS.md`, `PLAN.md`, `plans/ROADMAP.md`, and exactly one
brief.** Files not listed under a brief's "owns" belong to another stream: if you
need a change there, say so in your report rather than making it.

- **`ROADMAP.md`** — what is left, in what order, with the dependency graph.
  **Start there.**
- **`TERMINOLOGY.md`** — plain-language names used in new reports and the paper.
- **`SESSION-REPORT.md`** — how to assign work to Codex/Claude,
  Cursor/Antigravity, and what a completed session must return.
- **`done/FORK-bp31.md`** — historical pre-result decision record for BP31. It is
  superseded for scheduling by `ROADMAP.md`.
- **`DEFERRED.md`** — real work deliberately not now.

---

## Read this before anything else

Four things are true and all of them are load-bearing.

0. **PointStream currently loses to the codec it is built on.** BD-rate **+90.97%** against
   an av1 anchor at N=2 scenes with the cross-scene stream on
   (`plans/done/BP31-findings.md` §9), from +116.8% with a single-frame plate
   (`PLAN.md` §2.20). `AGENTS.md` requires the headline claim to land where
   PointStream wins, so **finding that regime is the work.** `done/FORK-bp31.md`
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

| Brief / roadmap ID | Owns | Order |
|---|---|---|
| `done/BP31-paired-ladder-across-scenes.md` + `done/BP31-findings.md` | completed multi-scene baseline and current negative result | done, PR #45 merged |
| M1/E1 / `BP45-ultra-low-rate-search.md` | metric direction, ultra-low AV1/VVC range and first search | first |
| B1 / `done/BP44-canonical-background-canvas.md` | canonical background canvas and context resets | done, PR #52 merged |
| D1 / `BP46-long-tennis-scenes.md` | long eligible first-domain inputs | first (D1 complete) |
| M2 / `BP32-rate-budget.md` + `BP33-span-amortisation.md` | byte ledger and long-scene slope | after B1 |
| E1 | first-domain low-rate × scene-length search | after M1/B1/D1 |
| E2 | six-video confirmation | after a candidate win |
| A1 / `BP41-ablation-lattice.md` | core component ablation matrix | after E2 |
| L1 / `BP38-paper-infrastructure.md` | DCVC-RT baseline | after E2 |
| G1 / `BP36-second-domain.md` | independent second domain | after E2 |
| T1 / `BP34-operating-point.md` | profile and speed optimization | after E2 |

## Parked, with the reason

Not dead, not scheduled. `ROADMAP.md` §5 carries the reasons in full.

| Brief | Parked because |
|---|---|
| `done/BP19-conditioning-architecture.md` | every engine loses to a paste; a training campaign costs the most and buys the least |
| `done/BP28-offset-crossover.md` | its crossover happens where both arms are as good as a photo of the wrong player; its useful half is folded into `BP41`'s temporal axis |

## Historical

`done/` holds finished briefs, each ending with a *Delivered* section — history,
not instructions. `done/README.md` indexes them. Completed reports named `BP24-*`,
`BP29-*`, `BP30-*`, `BP44-*`, `wave3-report.md`, `wave5-report.md` reside in `done/`.

**`done/BP5-roster-decision.md`'s roster verdict is void** (measured on
self-reconstruction), and so is anything in `done/` that ranked engines on LPIPS
or VMAF before 2026-08-23. **`BP10-appearance-pathway.md` is void**: its bands
classify a paste as a working ReferenceNet.

`prompts/` holds the paste-in prompts for sessions on other tools or in other
windows. Several are also the *record* of a wave's report and are cited from
code and findings — check for inbound references before deleting one.
