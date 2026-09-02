# BP41 — The core component ablation matrix, which is the paper's central contribution

The manuscript/configuration may still use “lattice” where it has a precise
structural meaning. New planning and reporting use *component ablation matrix*;
see `plans/TERMINOLOGY.md`.

**`PLAN.md` §7 P0 item 4, still un-run**, and the only P0 item whose absence
removes the paper's *stated* contribution rather than one of its results. The
abstract's central sentence is about the lattice; `sections/system_design.tex`
carries `NOTE(subsec:lattice)` forbidding any claim that a component is
justified until a BD-rate exists for it; `HOLE(subsec:eval-lattice)` says no
lattice sweep has ever been run end to end.

**Owns:** `experiments/lattice/**` (new), `outputs/bp41-lattice/**`,
`config/benchmarks/*.yaml`, `sections/evaluation.tex` §`subsec:eval-lattice` and
§`subsec:eval-object`.

**Blocked on:** Gate B in `plans/ROADMAP.md`, then on
`plans/BP33-span-amortisation.md` choosing a frames-per-scene value. Running the
matrix at the wrong scene length multiplies the wrong number by every axis.

**Read first:** `AGENTS.md` · `PLAN.md` §3 "the ablation lattice", §4, §5, §7 P0
item 4 · `plans/wave5-report.md` (the pose-axis warning) ·
`plans/BP26-config-plumbing.md` in `done/` · `plans/ENGINE-ROSTER.md`.

---

## 0. The definition that makes this measurable

`AGENTS.md` and `PLAN.md` §5 agree: **a component is justified iff enabling it
improves BD-rate against a common anchor.** Not if it shrinks the payload at one
operating point, and not because it looks better. A single-point comparison is
admissible only under dominance, where one arm is better on both axes.

So the deliverable is one table: **one row per component axis, the BD-rate of its
presence against its absence, everything else held fixed.** That table is the
paper.

## 1. The axes, and what is known about each before it runs

`BP26` (2026-08-26) established that **detector, pose, segmenter, appearance,
motion and temporal names now change a run**. Three axes remain unwired
(`BP24`): `codec`, `fallback`, `residual.codec`. Check that list against the code
before planning around it — a flag existing is not a feature working, and this
project has twice found an axis that reached nothing.

| axis | expectation, written before the run | note |
|---|---|---|
| background (off / keyframe / panorama / panorama-stream) | **the largest effect by far.** The plate is 88–91% of the payload | the levers are `BP40`'s |
| residual (absent / coarse / medium / fine) | positive and cheap: +0.9% rate for **+5.40 dB** measured (`BP24` ladder) | the clearest positive in the project |
| appearance (paste / upscale-refine / an engine) | **paste wins.** Every engine loses to it at 2.5σ–10.6σ | this is `subsec:eval-object`, as a *rate* claim |
| temporal policy / keyframe interval | unknown, and this is where `BP28`'s useful half lives | see §3 |
| detector, segmenter | small on rate, possibly large on failure rate | report failures, not just BD-rate |
| pose | **expect ~0 on PSNR** | `plans/wave5-report.md`: the pose axis moved keypoints without moving PSNR, so a lattice quoting only PSNR shows a row of zeros for pose. Quote all three axes (`BP35`) or the row is uninterpretable |

## 2. The object-representation comparison, reframed

`HOLE(subsec:eval-object)` is `PLAN.md` §7 P1 item 9 and is described there as
*the most novel item*. It has been blocked for months on "no working generative
engine", and it does not need one.

`ENGINE-ROSTER.md`'s ranking is on **object-bbox LPIPS on a 12-clip probe**, a
proxy. The lattice makes the real question askable: **what is each object
representation worth in BD-rate inside the full codec?** A paste costs a JPEG
crop per keyframe and nothing thereafter; a generative engine costs a pose
stream and a second of GPU time per frame; `upscale-refine` costs a downscaled
crop. Those are different rate/quality/time points and the roster has never
placed them on that plane.

That is a genuine contribution and it survives every negative result so far:
**"we measured what each object representation is worth in rate, and the cheapest
one wins"** is a finding. It is also the honest version of the alternative
`ENGINE-ROSTER.md` already names — that the value is in the decomposition plus a
cheap appearance channel plus a residual, not in the generator.

## 3. Temporal policy, absorbing what is useful from BP28

`BP28-offset-crossover.md` asks whether a generative engine ever beats a paste at
long offsets. The answer it would get is "yes, where both are as good as a photo
of a different player", which is not usable. **The useful half of the question is
a rate question:** how often must appearance be re-sent, and what does each
choice cost in BD-rate?

The measured input is already in `ENGINE-ROSTER.md`: a paste degrades at
**+0.0458 LPIPS per frame of offset** against a model's +0.0049. That is the
curve that sets the keyframe interval. Sweep the interval as a lattice axis and
read the BD-rate; do not re-run the offset probe.

## 4. Bounds — before the sweep, per axis

- **The background axis dominates**, contributing more BD-rate movement than
  every other axis combined. If it does not, `plans/BP32-rate-budget.md`'s ledger
  is wrong and both need re-reading.
- **The residual axis is positive** — enabling it improves BD-rate. It is the one
  component with a measured favourable trade already; a negative here means the
  sweep is not doing what the coarseness curve did.
- **The pose row reads ~0 on PSNR** and nonzero on LPIPS, or the perceptual axis
  is not being read.
- **No axis produces a BD-rate outside [−50%, +500%]**. Outside that, check the
  corner ran rather than reporting it.
- **Every corner runs.** `PLAN.md` §8 requires that every lattice corner produces
  a runnable pipeline. A sweep that silently skips corners is reporting a
  different lattice from the one the paper describes — count corners attempted
  against corners completed and never read a clean exit as completion
  (`AGENTS.md`: a batch runner that tolerates per-entry failures exits 0 when
  every entry failed).

## 5. Cost control

The lattice is a product of axes and the runs are 4K. Do not run the full
Cartesian product. **One-at-a-time against a fixed reference corner** answers the
"is this component justified" question the paper asks; interactions are a second
pass on the two or three axes that turn out to matter. Batch into one long-lived
process — `experiments/tier/run_ladder.sh` pays the import tax once per axis
instead of once per rung, and that difference was measured at 950 s against
150 s.

## Done when

- One table, one row per axis, BD-rate present-vs-absent against a common anchor,
  on all three quality axes, with n and the per-video spread.
- `HOLE(subsec:eval-lattice)` and `HOLE(subsec:eval-object)` are cleared by the
  edits that land the data, and `NOTE(subsec:lattice)`'s prohibition is lifted for
  each component the table justifies — and only for those.
- Every component the paper describes is either justified by a BD-rate or named
  in the text as measured and not justified. Both are results.
