# B′23 — One tier config, end to end, producing real numbers

**This is `plans/done/RESEARCH-HISTORY.md` §7 P0 item 1**, the standing blocker. P0 items 2, 3 and 4 —
the codec ladder, the residual-coarseness curve, the ablation lattice — are all
waiting on it. It is the highest-value unblocked item in the project.

**Owns:** `src/runner/**`, `src/pipeline/**`, `config/tier_*.yaml`,
`experiments/tier/**` (new), `tests/runner/**`, `tests/pipeline/**`.
**Read first:** `AGENTS.md`, `plans/done/RESEARCH-HISTORY.md` §3 and §7, `plans/done/C3-runner.md`,
`src/contracts/lattice.py`, `src/contracts/config.py`.

**Does not own** `src/shared/**` or `src/decoder/**` — those are BP22's. Good
news: `src/pipeline` and `src/runner` currently import **nothing** from either,
so the two streams are genuinely disjoint. Keep it that way; if you find you
need something from `src/shared`, report it rather than importing it.

## The state

The parts exist and are individually green. `src/runner/` is 819 lines across
`routing.py`, `stages.py`, `accounting.py`, `run.py` with 11 tests.
`src/pipeline/` is 2016 lines with 110 tests. `config/` holds `default.yaml` and
`tier_fast` / `tier_balanced` / `tier_quality`. Metrics compute real numbers
(`plans/done/RESEARCH-HISTORY.md` §2).

What has never happened is the three of them meeting: **no tier config has been
driven through the runner to a scored reconstruction.** Until that runs, "the
platform works" is an assertion about unit tests.

## What to do

1. **Pick `tier_fast` and one short clip** from the BP21 probe set. Fast first —
   the point is a complete path, not a good number.
2. **Run it.** Whatever breaks, fix in `src/runner` / `src/pipeline` or in the
   config. Report every fix; a config that had to be edited to run is a finding
   about the config format, not a detail.
3. **Score it** through `QualityEvaluator` and write the result plus the size
   accounting to `outputs/bp23-tier/`.
4. **Then `tier_balanced` and `tier_quality`** on the same clip, so the tiers are
   comparable and it is visible that the tier knob does something.
5. **Add the required-behaviour test** that a tier config runs end to end and
   produces a scored result. This is the gate that stops it silently rotting.

## Bounds — write these to `outputs/bp23-tier/bounds-before-run.json` first

The generator roster is a known negative (`plans/done/RESEARCH-HISTORY.md` §2.10: every engine loses to
pasting the keyframe). **So do not expect good quality, and do not treat poor
quality as a bug in this stream.** Expect roughly:

- Reconstruction on an all-off / residual-only corner: should be **close to the
  codec anchor**, because that corner is essentially a codec passthrough. A
  large gap there is a pipeline bug, not a model result.
- Any generative corner: **at or below the static-copy floor** (BP14's bar).
  Better than static copy would be the first such result in this project and is
  an *alarm* to verify, not a win to report.
- Wall clock for `tier_fast` on one short clip: minutes, not hours. Hours means
  something is running at full resolution that should not be.

Write the actual numbers and reasoning before the first run, per `AGENTS.md`.

## Traps

- **A flag existing is not a feature working.** If a stage is configured off,
  prove it was not invoked rather than assuming.
- **Do not add a generative result to the paper from this stream.** This stream
  proves the *path*; the roster verdict is separate and already recorded.
- **Do not widen scope into the experiments layer** (Phase D). One tier, one
  clip, then the two other tiers. Stop there.

## Done when

- `tier_fast`, `tier_balanced`, `tier_quality` each run end to end on one clip
  and produce scored output plus size accounting under `outputs/bp23-tier/`.
- A required-behaviour test covers the end-to-end path.
- The report says what had to be fixed to make it run, and how the three tiers
  differed.
- `plans/done/RESEARCH-HISTORY.md` §7 P0 item 1 is checked off, or the report says precisely what still
  blocks it.
