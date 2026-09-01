# Prompt — BP31: every plate lever on, then the paired ladder

Paste below the line.

---

You are running **BP31** on PointStream, an object-centric video codec targeting
ACM TOMM on **30 September**. This is the run the paper's core claim depends on.

**Read, in order:** `/home/itec/emanuele/.agent-rules/AGENTS.md` · this repo's
`AGENTS.md` · **`plans/BP31-paired-ladder-across-scenes.md`** (your brief — read
all of it, especially §0 and §2) · `PLAN.md` §§2.20-2.23 and §7 P0 item 8 ·
`plans/BP24-findings.md` §§1, 8, 12, 14. Do not read the whole plan tree.

## Where the project actually stands

`PLAN.md` §2.20: PointStream **loses to every codec it is built on** — BD-rate
+116.8% against av1 at preset 10, worse elsewhere — and the cause is the plate,
88-91% of the payload at every rung. Since then all three plate levers have
moved, and **none of them has been priced in a ladder**:

- **(c) real panorama** — done. `make_background` calls `build_plate`; residual
  falls to 0.22x and delivered Y-PSNR rises 4.9-6.2 dB on a moving clip.
- **(b) cross-scene stream** — done and wired. `background.method:
  panorama-stream` costs 49.2% ± 6.2% of coding every plate fresh over five
  videos, best case 29.4%.
- **(a) plate codec** — **open, and cheapest.** `av1` and `vvc` intra sidecars
  exist but were never in the sweep that ran; that sweep found **`jpeg` cheaper
  at the ladder's rung**, which is not what `PLAN.md` §2.21 assumes.

So the question this run answers is not "does the stream help" but **"with every
plate lever on, has the gap closed?"**

## Do these, in order

1. **Sweep (a) before spending a ladder on it.** jpeg against av1-intra and
   vvc-intra, at the ladder's operating point, **on the panorama plate** rather
   than a source frame. Cheap, and it decides which codec the ladder arm uses.
   Picking by assumption is how the ladder gets run twice.
2. **Bounds to `outputs/bp31-ladder/bounds-before-run.json` before the first
   encode.** The brief §3 drafts them; sharpen them with (1)'s result.
3. **The paired ladder over N scenes, both arms.** PointStream with (a)+(b)+(c)
   on, against codec X over the **same N scenes**, same low-delay constraint,
   same keyframe interval, one codec on both arms so the preset cancels. The
   anchor getting the same footage is the fairness condition, not a nicety —
   without it the comparison is rigged.
4. **Report per video with the spread**, not one averaged BD-rate.

## The mistake most likely to be repeated

**Do not run this on one clip.** BP30 measured its lever on one video, drew two
conclusions, and **both inverted at five videos** — including which reference
mode to recommend; the video originally picked was the least favourable of the
five. The per-video spread (0.294-0.624) is larger than every effect measured
inside it. §2.20's own ladder ran on `alcaraz_highlights/scene_000`, explicitly
the most static of eight cached clips.

Keep that clip as the continuity arm so the new number can be compared with the
old one. Add others. `experiments/tier/scene_plates.py` enumerates point-class
scenes for any video (`djokovic_federer` has 224, `alcaraz_perricard` 88), and
the harnesses take `--video`.

## Alarms, and the asymmetry to watch

- **A result showing PointStream winning is an alarm, not a triumph.** Check the
  anchor got the same N scenes under the same constraint before believing it.
  These checks get applied to disappointing results and skipped on exciting
  ones; when the news is good, add a check rather than stopping.
- **The background's share must fall** from 88-91%. If it does not move, the
  stream is not reaching the ledger.
- **The anchor's own rate must fall too** when given N scenes instead of one —
  it can predict across the join as well. If it is unchanged it is being run
  per-scene, and the comparison favours PointStream wrongly.

## Traps already paid for

- `RunResult.frames` is not the delivered clip; `delivered_frames` is (§8).
- A decode naming no `-c:v` re-encodes and caps every quality (§14).
- RGB-PSNR cannot be the quality axis against a 4:2:0 codec (§12).
- Presets are not equal effort across codecs (§1) — one codec on both arms, and
  never order the magnitudes against each other.
- **`SizesBytes.panorama` is a marginal cost under `panorama-stream`**, and a
  total spanning scenes is right only because chunk 0's keyframe is in the sum.
- **One bound model is one stream.** `make_background` binds once per run;
  `_bound_background` sitting outside the per-chunk body is load-bearing, and
  `tests/runner/test_background_stream_stage.py` fails if it moves back in.
- NFS: batch into one long-lived process; `conda run` swallows pytest's summary
  and exits 0, so use `--junit-xml`. Long runs go detached.

**Before opening a PR:** `ruff check`, `mypy --config-file pyproject.toml` (it
now covers `experiments/`), the tests for what you touched, and
`python -m src.contracts.layers`. Confirm CI green with `gh` — it is faster than
a local mypy here. **One PR per independently revertible change.**

## When it lands

The paper is waiting for exactly this. `sections/evaluation.tex` carries a
`NEXT(subsec:eval-ladder)` saying its BD-rates describe the un-amortised system,
and `sections/system_design.tex` a `NOTE(subsec:lattice)` forbidding any claim
that the component is justified until a BD-rate exists. Clear both with the
`update-paper` skill, and update `PLAN.md` §7 P0 items 2 and 8.

## Done when

The paired ladder is reported over N scenes on both arms with every plate lever
on, per video with the spread, against pre-written bounds — **or** the report
says precisely which arm failed and why. A negative result, clearly reported, is
the deliverable if that is what the data says.
