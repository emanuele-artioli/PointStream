# BP35 — BD-rate on a perceptual axis, because the paper's case is perceptual

**The paper has argued perceptually since the first draft and every BD-rate in it
is Y-PSNR.** PSNR is the metric that undersells exactly what an object-centric
codec does — a plate that is subtly wrong everywhere and a player that is
plausibly rather than identically reconstructed. `plans/prompts/next-session-bp31.md`
lists "quality axis" as one of four untried axes for finding a regime where
PointStream wins, and it is the only one of the four that costs no GPU time.

**Owns:** `src/components/metrics/bd_rate.py`, `src/components/metrics/comparison.py`,
`tests/components/test_metrics*.py`, `tests/invariants/test_metric_calibration.py`,
`outputs/bp35-perceptual/**`. **Does not own** `experiments/tier/**` — deliver the
capability; the BP31 session spends it.

**Read first:** `AGENTS.md` (control the instrument, then the result) ·
`PLAN.md` §2.7 and §6.5 · `plans/BP24-findings.md` §2 · `plans/BP27` ·
`sections/evaluation.tex` `NOTE(sec:eval-instruments)` and
`GOAL(subsec:eval-metrics)`.

**No result dependency.** Start now.

---

## 0. The module cannot do this today, and the reasons are specific

`src/components/metrics/bd_rate.py` was written for PSNR and it says so only in
its constants. Two things block a perceptual BD-rate outright:

1. **`MIN_QUALITY_SPAN_DB = 3.0` is in decibels**, with a docstring reasoning
   entirely about QP steps and dB. On a VMAF curve (0–100) a span of 3.0 is a
   sliver that would pass the guard — the very degenerate case the constant was
   introduced to reject, reintroduced through a unit change. On an LPIPS curve
   (0–1) *no* curve can ever span 3.0, so every comparison would be refused.
   The constant is right and its units are wrong for two of the three metrics
   the paper names.
2. **There is no quality direction.** `RDPoint`'s docstring says "Rate is
   lower-better" and nothing anywhere says which way quality runs. **LPIPS is
   lower-better**, so integrating it as written computes a BD-rate with the sign
   inverted, silently, on a curve that will look perfectly monotone.

Both are the shape of fault this project has been bitten by twice: a number that
comes back confident, plausible and meaningless. Fix them as *typed* properties
of the metric rather than as flags at the call site, so a future axis cannot
forget to pass one.

## 1. What to build

1. **A quality-axis descriptor per metric** — name, direction, valid range, and
   the minimum span below which a BD-rate is refused. Y-PSNR: higher-better,
   [0, ∞), 3.0 dB. VMAF: higher-better, [0, 100], **10 points** (see bounds).
   LPIPS: lower-better, [0, 1], **0.05**. Carry it on the curve, not the caller.
2. **Direction handling in the integration**, with a test that a lower-better
   curve and its negation give BD-rates of the same magnitude and opposite sign.
3. **Calibration anchors for every axis before any comparison uses it**, on the
   same identical / mild / severe / unrelated ladder
   `tests/invariants/test_metric_calibration.py` already applies to the metrics
   themselves — but applied here to the *BD-rate over those metrics*: a curve
   against itself must integrate to 0.000, and a curve against an unrelated
   arm's must land in a stated range. A BD-rate is an instrument too, and this
   project has published rankings from two instruments that were never checked.
4. **`compare_paired` reporting the axis it used** in its `describe()` output, so
   a number quoted in the paper cannot be mistaken for a PSNR one.

## 2. Bounds — write them before running the re-integration

The span floors above are themselves predictions and need a basis:

- **VMAF 10 points.** The BP23 tier run's VMAF spread across its rungs is the
  reference; one QP step on real 4K content moves VMAF by a few points, so ten
  is a little over one step, matching the reasoning that set 3.0 dB for PSNR.
  Check this against `outputs/bp23-tier/report.json` rather than adopting it —
  if the real curves span less than 20 VMAF, the floor is too tight and the
  reason must be recorded.
- **LPIPS 0.05.** `ENGINE-ROSTER.md` puts the separation between a static paste
  (0.4505) and an unrelated image (0.7358) at 0.285, so 0.05 is a sixth of the
  entire usable dynamic range of this content. Anything finer is inside the
  instrument's own noise (±0.022 on the paste anchor).
- **The re-integration of the existing ladders must not move the PSNR numbers.**
  Running the refactored module over `outputs/bp24-ladder/` must reproduce
  **+116.8%** and **+161.5%** to the tenth of a point. That is the cheapest
  available check that a refactor of an instrument did not change the
  instrument, and per `AGENTS.md` it is how a fired alarm gets closed cheaply.

## 2b. The second half of the blocker: the rungs sit where VMAF has no room

Fixing the module is necessary and **not sufficient**, and this is the part that
was missing from the first version of this brief.

`PLAN.md` §2.16 measured the instruments on this content: **VMAF's ceiling here is
97.54, not 100**, and it **floors at 0.00 for both severe blur and an unrelated
clip** — nothing resolves below its floor. The BP23 `tier_quality` run scored
**VMAF 97.4986**, which is the ceiling to three decimal places.

Now look at where the ladders sample. `outputs/bp24-ladder/av1-payload-lowmotion.json`
spans Y-PSNR **39.21 to 46.55 dB** for PointStream and 39.45 to 44.02 for the
anchor. That is the high-fidelity end of 4K broadcast content, where VMAF is
saturated. **So even with the module fixed, the existing rungs would produce a
VMAF curve with almost no span, and the span floor would correctly refuse it.**

**The consequence, and it converges with an axis already on the list.** To use a
perceptual axis the ladder has to extend *downward* — more rungs, lower rates,
into the regime where VMAF actually varies and where a generative reconstruction
is supposed to have an advantage over a starved transform codec. That is the same
move as the "rate regime" axis in `plans/prompts/next-session-bp31.md`, which
`plans/BP29-low-rate-report.md` §2 looked at once without the plate levers on.

So the sequence is: **fix the module, then check the span each metric actually
gets from the current rungs, and extend the ladder downward until it clears the
floor.** Report the span alongside the BD-rate every time — a perceptual BD-rate
over a two-point VMAF range is the degenerate case `MIN_QUALITY_SPAN_DB` exists to
reject, arriving through a different door.

**LPIPS may behave better than VMAF here** because it does not saturate the same
way, but it carries its own trap: `PLAN.md` §2.16 found **LPIPS's ordering
inverted at 960x540 while holding at 4K**, so calibration anchors do not transfer
across resolution. Every LPIPS number must state the resolution it was calibrated
at, and `plans/BP43-background-representation.md`'s downscaling sweep is exactly a
place where that could bite.

## 3. The result this enables, and how it must be reported

Re-integrate the ladders that already exist on the new axes. Expect the sign to
be more favourable to PointStream on VMAF and LPIPS than on Y-PSNR, because that
is the whole reason for doing it — **which is exactly why the result needs the
extra check rather than the celebration.**

- **Report all three quality axes for every curve, always — and encode/decode
  time beside them**, per `AGENTS.md`'s three-dimension rule. Reporting only the axis that
  flatters is the mistake `AGENTS.md` calls choosing the configuration after
  seeing the numbers. Three columns, one table, every time.
- **A BD-rate that flips sign between PSNR and VMAF is a finding about the
  metrics as much as about the system**, and must be stated that way: name what
  the perceptual metric is rewarding and check it on frames a human can look at.
  If VMAF prefers PointStream's output, put the frames in the paper and let the
  reader judge.
- **VMAF had its inputs crossed until 2026-08-23** (`PLAN.md` §2.7). Anything
  reading VMAF now inherits that history and must show its calibration.

## Done when

- BD-rate accepts a quality axis with a direction, a range and its own span
  floor; LPIPS and VMAF curves integrate; the PSNR numbers reproduce exactly.
- The three-axis table exists for every ladder currently on disk.
- `sections/evaluation.tex` `GOAL(subsec:eval-metrics)` can be answered: which
  metric answers which question, with a measured example of them disagreeing.
