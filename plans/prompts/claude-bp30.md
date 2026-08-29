# Prompt for Claude — BP30, the background as a stream

Paste below the line. **Merge PR #33 first** — it carries your brief's cleared
gate and the wave-8 prompts, and a session launched against docs it cannot see
is a mistake this project has already made twice.

---

You are running **BP30** on PointStream, an object-centric video codec targeting
ACM TOMM on **30 September**.

**Read, in order:** `/home/itec/emanuele/.agent-rules/AGENTS.md` · this
worktree's `AGENTS.md` · `PLAN.md` §2.20 and §2.21 ·
**`plans/BP30-background-stream.md`** (your brief, read it all — it is short) ·
`plans/BP24-findings.md` §§13, 16, 17, 18, 19. Do not read the whole plan tree.

## Setup — this changed on 2026-08-29

`assets/` and `outputs/` **no longer live in the checkout**. They are at
`/home/itec/emanuele/pointstream-data`, found via a gitignored `.ps-data-root`
marker. **Do not recreate the old symlinks** — a symlink is what editors follow,
and it is how one dataset became twelve worktrees' worth of NFS churn.

```bash
git worktree add -b wave8/bp30-stream /home/itec/emanuele/pointstream-w8-e origin/main
cd /home/itec/emanuele/pointstream-w8-e
echo /home/itec/emanuele/pointstream-data > .ps-data-root
conda run -n pointstream --no-capture-output python -c "from src.contracts import paths; print(paths.describe())"
```

Never join `"assets"` or `"outputs"` onto a repo root in new code;
`src/contracts/paths.py` is the only place that resolves them.

## You are running in parallel — this is the important part

Another Claude session is running **wave 8 streams A–D**
(`plans/prompts/cursor-wave8.md`). It owns:

| stream | owns |
|---|---|
| A | a new module under `experiments/tier/`, `outputs/bp29-plate-codec/**` |
| B | **`src/components/background/sidecar.py`** and its tests |
| C | `outputs/bp29-low-rate/**` |
| D | **`src/components/background/plate.py`** and **`make_background` in `src/runner/stages.py`** |

**You must not touch any of those.** Two of them are directly in your way, so
your scope is drawn to avoid them:

- **You do not wire anything into the runner.** `make_background` becoming
  stateful is the natural end point of this work and it is stream D's file this
  week. Build the component and the measurement; integration is a follow-up
  after D merges.
- **You do not add sidecar codecs.** Stream B is adding av1/vvc intra sidecars.
  If you need one, *consume* what exists or drive `coded_roundtrip` directly.

**You own:** a new `src/components/background/stream.py` (or similar), its
tests, a new `experiments/tier/` module for the measurement, and
`outputs/bp30-background/**`.

If you find yourself needing a file in that table, stop and say so rather than
editing around it.

## What is already settled — do not re-derive it

- **The gate is cleared** (findings §19). av1's inter saving is **causal**:
  0.671 and 0.470 unchanged *to the byte* with `-lag-in-frames 0`, because
  `-usage realtime` was already lookahead-free. You do not need to re-run this.
- **x265 is not av1 here.** 12% saving on one pair, 6% loss on the other. Treat
  the saving as a property of av1's inter tools, not of inter coding.
- **PSNR distance does not predict coding distance.** The pair *further apart*
  in PSNR saved more. This is why reference selection scores structure (Canny
  edge overlap), not pixel similarity — and why the proxy must be validated
  against trial encodes before you trust it.
- **Pixel subtraction is dead** (§17, retracted by §18). Do not revisit it.

## Do these, in order

1. **A stateful background transmitter.** Carry the previous *reconstruction*
   across scenes — never the original. This is the single place this work can
   silently go wrong: the paper already commits to exactly this discipline one
   level down (`sections/system_design.tex`: the residual is computed against
   the codec-decoded background, not the raw one), and the same rule applies
   here. A required-behaviour test should assert that encoder-side and
   client-side reconstructions are bit-identical across a multi-scene sequence.
2. **Reference-mode selection** as a named component: `first`, `last`,
   `best-scored` (Canny), `periodic-i`. Brief §3 has the reasoning.
3. **Validate the Canny proxy** on a handful of candidate pairs: run the real
   trial encode *and* the Canny score, and check they rank the same way. If they
   disagree, say so and recommend `first`, which is free and already worth
   31–53%.
4. **Measure a multi-scene sequence.** Bounds to
   `outputs/bp30-background/bounds-before-run.json` **before the first encode**.
   Report the marginal cost per scene and the total across N scenes, with the
   keyframe interval swept as an axis (`k` = 2, 4, 8, never) rather than imposed.

## Things that will bite you

- **Keep the control in every run.** Two consecutive frames of one scene must
  come back at a few percent (measured 1.2–3.3%). It has already caught one
  broken encoder configuration that would otherwise have been reported as a
  finding — see §19, where `rc-lookahead=0` made a P-frame come back *larger*
  than a fresh intra.
- **A decode that names no `-c:v` re-encodes**, capping every quality it returns
  (§14). A flat quality curve while bytes keep moving means a second encoder,
  not difficult content.
- **`RunResult.frames` is not the delivered clip** — `delivered_frames` is (§8).
- **A flag existing is not a feature working.** `background.codec` accepted
  three values and reached nothing until BP24 wired the background stage.
- **Report `n` and the uncertainty.** The current ladder is two clips, which
  does not meet the bar `presley` uses (n≥6 videos before a significance claim).
  Say what your `n` is.
- **NFS: ~10 ms per file open**, `import torch` ~124 s. Batch into one
  long-lived process; `conda run` swallows pytest's summary and exits 0 anyway,
  so use `--junit-xml` and read the XML.
- Never `git add -A` in a worktree. Confirm CI is green with `gh` before saying
  it is.

**Before opening a PR:** `ruff check`, `mypy --config-file pyproject.toml`, the
tests for what you touched, and `python -m src.contracts.layers`.

## Done when

The transmitter carries reconstructions across scenes with a bit-identity test
behind it; the four reference modes are implemented and ablated; the Canny proxy
is either validated against trial encodes or reported as not tracking them; and
a multi-scene sequence is measured against its own pre-written bounds with the
keyframe interval priced as an axis — or the report says precisely what blocked
which.

**What is explicitly not in scope:** wiring this into `make_background`, adding
sidecar codecs, and re-running the paired ladder. All three belong to other
streams or to the run that happens once every lever has landed.
