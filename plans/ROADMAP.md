# ROADMAP — everything still standing between here and the TOMM submission

**Written 2026-09-02.** This is the master index. `PLAN.md` says what the system
is and what has been measured; this file says **what is left, in what order, and
what each piece depends on**. A session picks one work item, reads its brief, and
does not read the rest of the tree.

`plans/README.md` classifies every brief as live / parked / historical. This file
is the ordering and the dependency graph.

---

## 0. The one paragraph a new session needs

PointStream is measured, end to end, against the codec it is built on, and **it
loses**: BD-rate **+90.97%** against an av1 anchor at N=2 scenes with the
cross-scene stream on (`plans/BP31-findings.md` §9), down from +109.72% without
it — at **8 frames per scene**, and §0b below is why that qualifier now matters
more than the number. `AGENTS.md` is explicit that a paper whose central result is "we lose
everywhere" is not a submission, so **finding the regime where an object-centric
codec wins is the work**, not a hoped-for outcome. Three things follow, and they
are the shape of this whole roadmap:

1. **One session is already on the winning-regime hunt** (BP31, worktree
   `pointstream-w9-a`, PR #45). Nothing here duplicates it.
2. **The gap between what the motivation measured and what the system delivers
   is ~149 BD-rate points and has never been reconciled.** That reconciliation
   (`BP32`) is arithmetic over data already on disk, and it is the cheapest
   thing in this file that can redirect the whole project. **Do it first.**
3. **Everything else divides cleanly** into work that is true whatever BP31
   finds (waves 9 and 10 below) and work whose shape depends on it
   (`plans/FORK-bp31.md`).

---

## 0b. What the span run changed, 2026-09-02 — read this before picking an item

`plans/BP31-findings.md` §12 ran `BP33` before the extraction campaign. The
mechanism held, **the brief's prediction did not**, and the run exposed something
that reorders this roadmap.

Span amortises the plate — and it amortises **the anchor's intra keyframe just as
well**. Over a doubling the ratio moved 2.17x → 2.01x, about 7%, and the fit says
it flattens near 1.9–2.0x from there. Both arms have a fixed per-scene cost; the
brief modelled only one of them.

Separating fixed from marginal is what the two span points buy, and the anchor's
marginal is a difference quotient, so it does not depend on its keyframe count:

| | fixed, amortised by span | **marginal, per frame** |
|---|---:|---:|
| av1 anchor | ~382,000 B (intra) | **4,757 B** |
| PointStream | 768,277 B (plate) | **9,319 B** (residual + crops + metadata) |

> **PointStream's marginal cost is ~1.96x the anchor's.** Its per-frame payload
> is about twice what av1 spends coding an entire inter frame — one that contains
> the players PointStream is transmitting separately.

**"The plate is 88–91% of the payload" is a span-8 artifact.** At 8 frames the
plate is 80% of it, at 16 it is 72%, and it keeps falling. Run at a span the
cache already supports and the dominant cost is **the residual and the crops**.

**What this reorders.**

- **Plate levers are worth less than `PLAN.md` §7 P0 item 8 assumes.** `BP43`'s
  resolution sweep and `BP40`'s codec and canvas work all act on a term span is
  driving toward irrelevance. Still worth doing — the plate is real at any finite
  span — but they are no longer the item that decides the paper.
- **`BP41` rises sharply.** The residual and appearance axes are now the ones on
  the dominant cost, and the lattice is what prices them.
- **One cheap run comes first, and it falsifies the above if it is wrong.** Span
  24/32/48 under `panorama-full` needs **no component change** (each scene codes
  its own plate, so the canvas-agreement blocker in `BP40` §3b does not apply)
  and gives the third point the two-point fit needs. **And the non-plate split —
  residual against crops against metadata, per frame** — because nobody can act
  on a marginal cost quoted as one number.

Two smaller results from the same run, both now folded into their briefs: canvas
growth is **measured** at x1.038 worst case rather than extrapolated (`BP32`), and
PointStream encodes at **x19.1–19.7** the anchor's wall clock (`BP34`).

---

## 1. The dependency graph

```
                    ┌─────────────────────────────────────────┐
                    │ BP31 (RUNNING, another session, PR #45) │
                    │ the paired ladder over N scenes         │
                    └───────────────────┬─────────────────────┘
                                        │ result
  WAVE 9 — independent, start now       │
  ┌──────────────────────────────┐      │
  │ BP32 rate-budget ledger  ⭐  │──────┼──► feeds BP31's axis choice
  │ BP33 span (8 vs 48 frames)⭐ │──────┘    (tell that session early)
  │ BP34 encode/decode time      │
  │ BP35 perceptual BD-rate      │──────────► feeds BP31's quality axis
  │ BP36 second domain           │
  │ BP37 required-behaviour gate │
  │ BP38 paper infra + DCVC      │
  │ BP43 plate resolution   ⭐   │──────────► feeds BP31's plate levers
  └──────────────┬───────────────┘
                 │
  WAVE 10 — after PR #45 merges (file ownership, not results)
  ┌──────────────▼───────────────┐
  │ BP39 all-off corner (D5)     │
  │ BP40 background honesty +    │
  │      intra_qp + (a)/(b)      │
  │ BP41 the ablation lattice    │
  └──────────────┬───────────────┘
                 │
  WAVE 11 — shape decided by BP31 + BP32: plans/FORK-bp31.md
  ┌──────────────▼───────────────┐
  │ BP42 headline, conclusion,   │
  │      abstract, title         │
  └──────────────────────────────┘
```

**Read `plans/FORK-bp31.md` before planning wave 11.** It carries three
pre-written branches — a regime is found, a boundary is found, nothing wins —
with the paper each one produces. Writing them in advance is the point: it stops
the configuration from being chosen after the numbers are seen and then
presented as though it had been predicted, which `AGENTS.md` names as the one
thing that would sink the paper faster than any negative result.

---

## 2. Wave 9 — start now, no result dependency

Every item here is true whatever BP31 returns. Ownership is file-level and
disjoint; none of these touch `src/runner/stages.py`, `src/components/background/**`,
`experiments/tier/**` or `tests/runner/**`, which PR #45 holds.

| # | Brief | What it settles | Cost | Owns |
|---|---|---|---|---|
| **BP32** ⭐ | `BP32-rate-budget.md` | where the 149 BD-rate points between the measured headroom and the delivered system actually go | hours, mostly arithmetic | `experiments/budget/**`, `outputs/bp32-budget/**` |
| **BP33** ⭐ | `BP33-span-amortisation.md` | whether clip length is the dominant lever — the ladder runs at 8 frames, the cache holds 48 | one sweep | `outputs/bp33-span/**`, brief only until #45 lands |
| **BP34** | `BP34-operating-point.md` | encode/decode time, the operating-point table, and whether the title may keep "Live Video Streaming" | one measured pass | `experiments/timing/**`, `outputs/bp34-timing/**` |
| **BP35** | `BP35-perceptual-bdrate.md` | BD-rate on VMAF and LPIPS, calibrated — the paper argues perceptually and the ladder is Y-PSNR | code + calibration | `src/components/metrics/**`, `tests/components/test_metrics*.py` |
| **BP36** | `BP36-second-domain.md` | the general/DAVIS profile driven end to end — P0 item 6, and the most-requested reviewer item | a day plus data | `src/components/domain/**`, `outputs/bp36-general/**` |
| **BP37** | `BP37-required-behaviour.md` | the gate `PLAN.md` §8 describes, against the list it actually names | half a day | `tests/invariants/**`, `PLAN.md` §8 |
| **BP38** | `BP38-paper-infrastructure.md` | figures, a **DCVC-class anchor** (decided), reproducibility, related-work currency | a day + the anchor | the paper repo, `figures/`, `appendices/` |
| **BP43** ⭐ | `BP43-background-representation.md` | making the plate **smaller in pixels** — never expressible, `BackgroundConfig` has no resolution field | one sweep | `src/components/background/sidecar.py`, `BackgroundConfig` |

**BP32, BP33 and BP43 are marked ⭐ because they can change what BP31 spends its
next campaign on.** Do them first and tell that session the answer; it is about to
extract scene windows for a ten-scene, six-video ladder **at eight frames per
scene**, and if span is the dominant term that campaign is being run at the
wrong operating point.

## 3. Wave 10 — after PR #45 merges

Blocked on **file ownership, not on results**. Each of these edits a file the
BP31 branch is holding.

| # | Brief | What it settles | Blocked by |
|---|---|---|---|
| **BP39** | `BP39-all-off-corner.md` | `DEFERRED.md` D5 — the all-off corner is a hardcoded branch, so the abstract's central lattice claim is delivered by a shortcut | `src/runner/stages.py` |
| **BP40** | `BP40-background-honesty.md` | `BackgroundArtifact.codec` reports a sidecar a streamed run never used; `intra_qp` reaches nothing; levers (a) and (b) cannot compose | `src/components/background/**` |
| **BP41** | `BP41-ablation-lattice.md` | `PLAN.md` §7 P0 item 4 — the core lattice, still un-run, and the paper's central contribution | the corrected ladder harness |

## 4. Wave 11 — fork-dependent

`plans/FORK-bp31.md`. `BP42` (headline claim, Conclusion, abstract, title) is
written there three times over, once per branch.

## 5. Parked, with the reason

Not dead, not scheduled. Each has a brief that still reads correctly; what
changed is the priority, and the reason is recorded so nobody re-derives it.

| Brief | Parked because |
|---|---|
| `BP19-conditioning-architecture.md` | every engine loses to pasting the keyframe (`PLAN.md` §2.10, §2.17), and the roster's own reading is that the value is not in the generator (`ENGINE-ROSTER.md`). A training campaign is the most expensive thing available and buys the least. Revisit only if `BP41` shows the appearance axis moving BD-rate. |
| `BP28-offset-crossover.md` | the crossover it tests happens where both arms have degraded to "photo of a different player". Its useful half — keyframe interval as a **rate** lever — is folded into `BP41`'s temporal axis. |
| `DEFERRED.md` D2 (SAM3) | needs a second conda env with newer torch; P1 item 10 only. |
| `DEFERRED.md` D6 | two pre-rewrite Animate-Anyone tests fail only in the full suite. Re-check under `BP37`: the modules moved to `src/components/generation/`, so "they die with their modules" no longer applies. |

---

## 6. What the paper is still missing, by marker

The camera-ready sweep must return only `CLAIM` lines. Today it returns these.
Each row names the work item that clears it.

| Marker | Where | Cleared by |
|---|---|---|
| `HOLE(sec:conclusion)` | `main.tex` | BP42 |
| `HOLE(abstract)` | `main.tex` | BP42 (needs BP31's number) |
| `NEXT(abstract)` — the title still promises live streaming | `main.tex` | **BP34** |
| `NEXT(paper-wide)` — second domain, MOS study, demo video | `main.tex` | BP36 (domain); MOS is scoped out in `future_work`; demo video = BP38 |
| `HOLE(sec:evaluation)` | `evaluation.tex` | BP31 + BP41 |
| `HOLE(subsec:eval-ladder)` (partial) | `evaluation.tex` | BP31 |
| `NEXT(subsec:eval-ladder)` — BD-rates describe the un-amortised system | `evaluation.tex` | BP31 |
| `HOLE(subsec:eval-lattice)` | `evaluation.tex` | **BP41** |
| `HOLE(subsec:eval-residual)` (partial) | `evaluation.tex` | BP41 |
| `HOLE(subsec:eval-object)` | `evaluation.tex` | BP41 (as a **rate** claim, not an LPIPS one) |
| `HOLE(subsec:eval-general)` | `evaluation.tex` | **BP36** |
| `HOLE(subsec:eval-operating)` | `evaluation.tex` | **BP34** |
| `HOLE(sec:system-design)` — designed and unproven | `system_design.tex` | BP39 (the lattice half), BP41 |
| `NOTE(subsec:lattice)` — no component may be called justified without a BD-rate | `system_design.tex` | BP31, BP41 |
| `HOLE(app:roi)` ×2 | `roi_verification.tex` | BP41's region arm, or scope out in `future_work` |

**Two gaps that no marker records yet**, both found on 2026-09-02 and both owned
by BP38:

- **No learned/neural codec baseline exists anywhere** — not in `src/`, not in
  `experiments/`, and the Related Work section's only learned-coding citation is
  `lu2019dvc` (2019). A 2026 TOMM submission on generative coding will be asked
  why there is no DCVC-class anchor. Either add one or state in the text why the
  conventional ladder is the right comparison for this claim.
- **`avg_vmaf_vs_bitrate.png`, `hls-vmaf.png`, `per_frame_vmaf.png`,
  `cgan_performance.pdf`, `vmaf-lpips_vs_bitrate_dualrow.pdf`, `players.pdf` and
  `PointStreamOverview.pdf` are unreferenced** and date from the ACM MM
  submission whose numbers are retracted. Only `PS-overview.pdf` is `\includegraphics`'d.

---

## 7. What a re-read of the existing work turned up, 2026-09-02

Five things that were already true, are already written down somewhere, and had
not been acted on or connected. Each is assigned above; they are collected here
because together they are the answer to "did we miss something".

1. **Span.** Every ladder has run at 8 frames per scene; the cache holds 48; the
   headroom the system is judged against was measured over those 48. Three
   separate reports call 8 frames "the least favourable amortisation a fixed
   plate cost can get" and none acted on it. → `BP33`.
2. **The ledger has never been drawn.** Headroom says ~23% of av1's rate is
   foreground and 34–69% of the background's is recoverable; the system delivers
   +90.97%. Nobody has attributed the difference. → `BP32`.
3. **The anchor runs at a speed preset.** SVT-AV1 preset 10, and it does *not*
   cancel between the arms the way `DEFERRED.md` D-CODEC-PRESETS assumes,
   because the anchor codes 100% of its pixels through that path and PointStream
   codes only its residual through it. **This is an argument, not a measurement**
   — nobody has run the anchor at another preset, so no number here may be
   restated as a bound until someone does. Note also that the share it turns on
   is span-dependent: PointStream's non-plate payload is ~9% at span 8 and ~28%
   at span 16, so the argument's magnitude shrinks as span grows. `BP32` §3 owns
   the two encodes that would settle it.
4. **BD-rate is PSNR-only by construction.** `MIN_QUALITY_SPAN_DB = 3.0` is in
   decibels — a sliver on a VMAF curve, impossible on an LPIPS one — and nothing
   in the module records which direction quality runs, so an LPIPS BD-rate would
   come back sign-inverted and monotone-looking. The perceptual axis is listed as
   one of four ways to find a winning regime and it cannot currently be computed.
   → `BP35`.
5. **Local `pytest` and `mypy` could not start at all.** Both read config from
   `pyproject.toml`; Python 3.10 has no `tomllib`; `tomli` was missing from the
   env, so both died inside their own argument parsing with an error that looks
   like nothing. CI was the only gate anyone had, and CI's ruff step was
   *narrower* than the local one — it passed explicit paths, which overrides the
   project's file set, and omitted `experiments/`. Both fixed 2026-09-02.

Two more, smaller, already assigned: `PLAN.md` §8 described the invariant suite
as "a three-test stub" when it holds five modules and 1,145 passing lines
(`BP37`), and `src/runner/stages.py:816` carries the same forbidden passthrough
branch as `src/pipeline/reconstruction/reconstruct.py:96` with no test guarding
it (`BP37` widens the guard, `BP39` fixes both).

**And one thing that is not a defect but is the largest single risk:** every
number in this paper is measured on broadcast tennis this project selected,
segmented and cached itself. A regime found only there will be read as a regime
built rather than found. `/home/itec/emanuele/Datasets/UVG/1920x1080` holds
`Jockey` and `ReadySteadyGo` — standard sequences, moving camera, small fast
subject on a large predictable background, which is the claimed regime in a
dataset the project did not curate. → `BP36`.

## 8. Two standing rules added 2026-09-02

**Every result carries size, quality and speed.** Not two of them, and not speed
in a limitations paragraph. `PLAN.md` §5 item 1 already asks for "rate, quality
*and* encode time on the same axes"; `AGENTS.md` now makes it a property of every
reported comparison. Wall clock is already recorded per run — the gap was only
that it never reached the table beside the rate and the quality. Every brief here
has it in its "done when".

**Searching for the winning configuration is the method, not a compromise.** Run
the axes, see the numbers, pick the regime where PointStream wins, and report the
search. `AGENTS.md` was rewritten on 2026-09-02 to say this plainly, because the
earlier wording read as though a configuration chosen after seeing data were
suspect. It is not. The only obligation the search creates is to say which axes
were tried and where the boundary is. Pre-registered bounds stay, for a different
job: catching a broken measurement, not locking a choice.

## 9. Standing hazards this roadmap exists to keep visible

- **The asymmetry.** These checks get applied to disappointing results and
  skipped on exciting ones. When the news is good, add a check rather than
  stopping.
- **Bound before believing, two-sided** where the bound is on the very quantity
  the experiment exists to generalise past.
- **Report the search.** A regime found by search is a finding when the search is
  reported and the claim is scoped to it, and a fabrication when it is presented
  as predicted.
- **Per-video spread, not one averaged number.** BP30 drew two conclusions from
  one video and both inverted at five.
- **One PR per independently revertible change.** Over-splitting burns the
  Copilot review budget; under-splitting keeps `main` stale.
