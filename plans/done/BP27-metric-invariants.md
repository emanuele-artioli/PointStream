# B'27 — Pin the metric calibration findings into the invariants

**Small, cheap, and disproportionately valuable.** Two metrics in this project
were broken until 2026-08-23 and **every engine ranking taken before that date is
void** (`plans/done/RESEARCH-HISTORY.md` §2.7, §2.10). BP23 found two more instrument limits. They
currently live in a JSON file and a session transcript. That is not where
knowledge that invalidates rankings belongs.

**Owns:** `tests/invariants/**`, `src/components/metrics/**` docstrings only.
**Read first:** `AGENTS.md` ("control the instrument, then the result"),
`plans/done/RESEARCH-HISTORY.md` §2.7 and §2.16, `outputs/bp23-tier/metric-calibration.json`,
`outputs/bp23-tier/metric-notes.md`.

## The two findings to pin

**1. VMAF's usable range is narrower than assumed.** On this content its ceiling
is **97.54, not 100**, and it **floors at 0.00 for both severe blur and an
unrelated clip** — nothing resolves below its floor. Any comparison that lands
two arms near 0 is not measuring a difference; it is reporting saturation.

**2. LPIPS anchors do not transfer across resolution.** Its ordering **inverted
at 960x540** and held at 4K. A calibration done at one resolution says nothing
about another.

## What to do

1. **Add invariants** that fail loudly if either property changes: VMAF's ceiling
   and floor behaviour on the known anchors, and LPIPS's resolution dependence.
   These are properties of the instrument, so they belong beside the existing
   `tests/invariants/test_metric_calibration.py`.
2. **Make the floor case explicit.** An invariant that only checks ordering would
   have passed both of the metrics that were broken here. Assert the *absolute*
   scale against the published range, not just the ordering — that is the check
   that was missing last time.
3. **Document the range next to the metric.** Every place that returns a VMAF or
   LPIPS number should carry its usable range in the docstring, so a caller
   cannot quote 0.067 without knowing an unrelated image scores 0.645.
4. **Say what you did not cover**, per the project's test rules.

## Traps

- **Never add a test to raise a coverage number.** This brief is small on purpose.
  Three or four real properties beat twenty assertions.
- Do not re-derive the calibration; BP23 measured it. Pin it.
- If an invariant fails on arrival, that is a finding — report it, do not tune the
  threshold until it passes.

## Done when

- Invariants cover VMAF's ceiling/floor and LPIPS's resolution dependence.
- Absolute scale is asserted, not only ordering.
- Metric docstrings carry usable ranges.
- The report names what is deliberately untested.
