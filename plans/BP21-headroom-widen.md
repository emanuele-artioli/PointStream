# B′21 — Widen the headroom to a citable n, and close its three gaps

**Owns:** `experiments/headroom/**`, `src/components/background/plate.py`.

**Not** the paper — `sections/problem.tex` is being rewritten in parallel from
the numbers `BP20` already produced. Do not touch it. When this brief lands,
report the new numbers and the rewrite gets a second pass.

## Why this exists

`BP20` measured the real thing and the premise held: a player is ~1% of the
pixels and **17–24%** of the bitrate, and a panorama plate saves **34–69%** of
the background. That is the paper's opening argument.

It rests on **n = 2 clips**. This project's own bar is n ≥ 8, and
`compare_paired` refuses to call a direction below that. An opening argument
carried by two clips is the kind of thing a referee asks about first.

Three narrower gaps came with it, all recorded in `BP20`'s *Delivered* section.

## What to do

### 1. Widen to n ≥ 8 clips — the main job

Same pipeline, more scenes. Spread them: **at least four different matches**,
`cluster_point` scenes rather than `cluster_interlude` (an interlude is mostly
crowd, and the claim is about play). Keep 48 frames per clip so the arms stay
comparable with `BP20`'s two.

**Re-run the paste-back check on every new clip.** `BP20` found that
`extract_24_frame_id` reproduces the source at MAE 0.0 while the native-fps and
positional conventions do not. Do not assume it generalises — a clip whose
paste-back fails is measuring the wrong region and must be dropped with its
reason recorded, not quietly fixed.

Report per-clip and as a mean with a standard error, and state n everywhere.

### 2. Rule out the VVC confound

VVC ran at QP 32/40/**47** where every other codec ran 32/40/48, because
`libvvenc` 1.11.0 writes an empty bitstream at 48 on some 4K fills. VVC's FG
saving is ~0.077 below the other three, repeating on both clips.

**Two explanations are live and this brief separates them:** VVC genuinely codes
the player region better, or the different rate ladder moved the BD-rate
integration interval. Either re-run every codec on a **common QP set** where all
four produce a valid bitstream, or integrate every codec's BD-rate over the
**common PSNR interval** and report that alongside.

Whatever the answer, it is a sentence in the paper. Do not leave it as an
asterisk.

### 3. Report AV1's background BD-rate

Currently absent: PSNR overlap between the arms was **0.46 and 0.20**, below the
50% the BD-rate implementation requires. Widen the QP sweep until the curves
overlap, or say why they cannot. AV1 is the rung the source video is already
encoded in, so a missing cell there is the most conspicuous one.

### 4. Fix the plate NaN

`src/components/background/plate.py` emits `All-NaN slice` from `nanmedian` on
real 4K masks — a column where every frame is masked has no median to take.
Decide what the plate should contain there (nearest valid, or a recorded hole)
and make it explicit. **Check whether it changed any `BP20` number**; if it did,
that is a correction to publish, not a silent improvement.

## Bounds, written before re-running

`BP20`'s bands were written from the synthetic run and several were wrong.
Re-derive from the real result, and write them down first:

- **FG plate saving, per codec, n≥8:** expect the mean within **±0.06** of
  `BP20`'s per-codec figure (AVC 0.244, HEVC 0.234, AV1 0.229, VVC 0.167). A
  mean outside that on a wider sample is an alarm — check the paste-back and the
  matched-quality pairing before reporting it.
- **Player area, alpha silhouette:** **0.004–0.020**. `BP20` saw 0.0055 and
  0.0102. Do **not** use bbox area; that was `BP20`'s wrong bound.
- **Concentration** (saving ÷ area) should stay in the **10×–60×** range. This
  is the number the paper leads with, so it earns its own bound.
- **BG saving:** **0.25–0.75**, improving with codec strength if `BP20`'s
  pattern holds.
- **After the VVC fix:** state before running whether you expect VVC's gap to
  survive. Record which way it went.

## Traps

- **Matched quality, not matched size.**
- **Flat fill is not an upper bracket.** It understated the prize on synthetic
  *and* on real 4K. Keep all three fills; plate is the honest estimate.
- **This is a headroom argument, not a PointStream result.** It says what is
  available if a generator works. Keep mechanism out of it.
- **4K is slow.** Encode scenes, not matches. Detached, `python -u`, checkpoint.
- Resolve encoders by **path and version**, and record both.

## Done when

- FG and BG headroom on **n ≥ 8 real 4K clips**, ≥4 matches, per-clip and
  meaned with a standard error, every codec that produced a valid bitstream.
- The VVC QP confound is resolved one way or the other, in a sentence.
- AV1's background BD-rate is reported, or its absence explained.
- The plate NaN is fixed, and any change to a `BP20` number is stated as a
  correction.
- `PLAN.md` §2.14's numbers are superseded by these — report them; the plan
  edit is made centrally.
