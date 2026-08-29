# Prompts for Cursor — wave 8, four parallel streams on the plate

**Context.** The paired ladder ran and PointStream loses to every codec: BD-rate
+116.8% (av1), +166.8% (hevc), +165.9% (avc), +378.1% (vvc) on the most static
clip available, and no overlap at all on the most dynamic one. The loss is
concentrated in one component: **the plate is 88–91% of the payload at every
rung of every sweep**, and it has three untried levers. These four streams are
the cheap ones.

**Merge state.** PR #31 and #32 are both merged. Branch off `origin/main`.

---

## Setup — read this, it changed on 2026-08-29

`assets/` and `outputs/` **no longer live in the checkout**. They are at
`/home/itec/emanuele/pointstream-data`, and a gitignored `.ps-data-root` marker
file points at them. Do not recreate the old symlinks: a symlink is what editors
follow, and it is how one dataset became twelve worktrees' worth of NFS churn.

```bash
git worktree add -b wave8/<stream> /home/itec/emanuele/pointstream-w8-<x> origin/main
cd /home/itec/emanuele/pointstream-w8-<x>
echo /home/itec/emanuele/pointstream-data > .ps-data-root
```

Verify before starting anything else:

```bash
conda run -n pointstream --no-capture-output python -c "from src.contracts import paths; print(paths.describe())"
```

Never join `"assets"` or `"outputs"` onto a repo root in new code —
`src/contracts/paths.py` is the only place that resolves them.

## Rules that bind all four streams

**Read first:** `/home/itec/emanuele/.agent-rules/AGENTS.md`, this worktree's
`AGENTS.md`, `PLAN.md` §2.20 and §2.21, and **your one brief**. Do not read the
whole plan tree.

**Do not re-run the paired ladder.** Every stream ends up wanting to, and four
ladders against four half-finished levers is four numbers nobody can combine.
Write your own results under `outputs/<your-stream>/`, report what your lever
did to the *plate*, and leave the paired ladder to one run once the levers land.

**Bounds before numbers.** Write a plausible best and worst case, with its
reasoning, to `outputs/<your-stream>/bounds-before-run.json` *before* the first
encode. A result outside them is an alarm to investigate, not a finding to
report. When a bound turns out to be wrong, record why it was wrong rather than
editing it away.

**Traps this project has actually hit**, all of them recently:

- **A decode that names no `-c:v` re-encodes.** ffmpeg picks the muxer's default
  encoder — libx264 at its own CRF for a `.mkv` — and caps every quality it
  returns. Fixed in `src/components/codec/command.py`; if you write new ffmpeg
  calls, name the codec. A flat quality curve while bytes keep moving means a
  second encoder, not difficult content (`plans/BP24-findings.md` §14).
- **`RunResult.frames` is not the delivered clip.** Use `delivered_frames` for
  anything paired with a byte count (§8).
- **A flag existing is not a feature working.** `background.codec` accepted
  three values and reached nothing at all until BP24 wired `make_background`.
  Drive the option and prove the output changed.
- **`conda run` swallows pytest's summary and exits 0 anyway.** Use
  `--junit-xml` and read the XML.
- **NFS: ~10 ms per file open.** `import torch` costs ~124 s. Batch work into
  one long-lived process rather than many short ones.
- Never `git add -A` in a worktree. Confirm CI is green with `gh` before saying
  it is.

**Before opening a PR:** `ruff check`, `mypy --config-file pyproject.toml`, the
tests for what you touched, and `python -m src.contracts.layers`.

---

## Stream A — does the plate's codec knob do anything?

**Brief:** `plans/BP29-plate-rate.md` §1.1.
**Branch:** `wave8/plate-codec-sweep` · **Worktree:** `pointstream-w8-a`
**You own:** `experiments/tier/**` (new module), `outputs/bp29-plate-codec/**`.
**Do not touch:** `src/runner/**`, `src/pipeline/**`, `src/components/codec/**`.

`background.codec` accepts `{jpeg, png, roi-video}`. `roi-video` is a
single-frame libx264 encode and **has never been measured against `jpeg`**,
because that axis reached nothing until BP24 wired the background stage.

Sweep all three with the residual held **fixed**, on
`alcaraz_highlights/scene_000`, 8 frames. Report per rung: plate bytes, plate
PSNR, total payload, delivered Y-PSNR. The question is what the plate costs and
what it buys — not a BD-rate.

**Bounds to write first.** `jpeg:75` measured 463,334 B at ~42.8 dB on this
plate. `roi-video` is x264 intra, which should land between JPEG and av1-intra —
expect roughly 250,000–400,000 B at matched fidelity. `png` is lossless and
should be *larger* than every JPEG rung; if it is not, it is not running.

**Done when** the three codecs are measured at matched plate fidelity and the
report says which is cheapest and by how much — or says which of them never
reached the encoder.

---

## Stream B — an intra-codec sidecar for the plate

**Brief:** `plans/BP29-plate-rate.md` §1.2.
**Branch:** `wave8/intra-sidecar` · **Worktree:** `pointstream-w8-b`
**You own:** `src/components/background/sidecar.py`, its tests,
`outputs/bp29-intra-sidecar/**`.
**Do not touch:** `src/runner/**`, `experiments/tier/ladder.py`.

Measured on one 4K plate at ~38 dB: **JPEG 283,431 B, av1-intra 79,726 B,
vvc-intra 68,477 B** (`outputs/bp24-ladder/plate-probe.json`). That is 3.6–4.1x
on 88–91% of the payload, for no architectural change — a modern intra frame is
what AVIF and HEIC already are.

Add `av1` and `vvc` as sidecar codecs on the existing interface.
`src.components.codec.measure.coded_roundtrip` already codes a single frame, so
this is a wrapper, not a new encoder path. Follow `JpegSidecar`'s shape: a
`codec_id` carrying the encoder and its setting, `encode` returning bytes,
`decode` returning pixels.

**The pairing discipline matters here.** When the ladder later uses this, the
plate must be on the **same codec as the anchor** in each pair. Do not hardcode
av1; take the codec from config.

**Bounds to write first.** At matched fidelity the new sidecars should
reproduce the plate-probe numbers within ~15%; a size within a few percent of
JPEG's means the encoder is not running.

**Done when** both sidecars round-trip a real 4K plate, the byte count moves
with the quality knob, and a required-behaviour test asserts a coarser setting
returns visibly worse pixels.

---

## Stream C — is there a crossover at very low rate?

**Brief:** `plans/BP29-plate-rate.md` §2.
**Branch:** `wave8/low-rate` · **Worktree:** `pointstream-w8-c`
**You own:** `outputs/bp29-low-rate/**` and a `--rungs` invocation of the
existing ladder. **Do not modify** `experiments/tier/ladder.py`.

The ladder stopped at QP 55. PointStream degrades to a clean plate; a starved
transform codec degrades to blocking, and the two are not the same kind of bad.
`presley`'s operating map records that *the same video flips sign along the QP
ladder*, so a crossover is a measured phenomenon in a sibling project rather
than a hope.

Extend the **anchor** to QP 58, 61, 63 (av1's range runs to 63) and pair it
against the cheapest PointStream configuration available. Four extra encodes.

**Bounds to write first.** At QP 55 av1 was 85,995 B at 39.45 dB. PointStream's
floor is its plate, which cannot go below roughly 70,000 B without falling
apart, so on frame PSNR a crossover is **unlikely** — say so before running.
Finding one anyway is the interesting outcome and gets an extra check, not a
celebration.

**Done when** the anchor's curve is extended and the report says whether the
curves cross, at what rate, and what quality both arms deliver there.

---

## Stream D — wire the panorama into the runner

**Brief:** `PLAN.md` §7 P0 item 8(c) · `plans/BP24-findings.md` §6.
**Branch:** `wave8/panorama` · **Worktree:** `pointstream-w8-d`
**You own:** `src/components/background/plate.py`, `make_background` in
`src/runner/stages.py`, `outputs/bp29-panorama/**`.
**Do not touch:** `src/components/background/sidecar.py` — that is stream B.

`build_plate` exists in `src/components/background/plate.py` and the runner does
not call it. Today `background.method` selects a *transmission strategy* over a
single source frame, so a panorama's whole argument — amortisation across the
clip — has never been available.

Wire it, and measure what it does to plate bytes and to the residual. A stitched
panorama should make the plate *larger* (it covers more area) and the residual
*smaller* (the background matches more frames). **Whether the trade pays is the
finding**, and it may not.

**Bounds to write first.** On `alcaraz_highlights/scene_000` the camera barely
moves — inter-frame MAD 0.33 — so the panorama should be close to the single
frame and the trade close to neutral. On `federer_djokovic/scene_003` (MAD 7.70)
the panorama should be substantially larger and the residual substantially
smaller. **If the static clip shows a large change, the stitcher is moving
something it should not.**

**Done when** `background.method` reaches `build_plate`, both clips are measured,
and the report states the plate/residual trade on each — with the caveat that
this is one clip per motion regime and not a corpus.
