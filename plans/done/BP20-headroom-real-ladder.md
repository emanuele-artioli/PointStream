# B′20 — The headroom, on real 4K, across the ladder

**Owns:** `experiments/headroom/**`, and `sections/problem.tex` in the paper repo.

**Supersedes the measurement half of `BP13`.** That brief's design was right and
its harness works; what it produced is a number for a 96×128 synthetic court,
not for tennis. `BP13`'s *Delivered* section stands as the record of the harness.
This brief replaces the number.

## Why this exists

`BP13` landed a real harness — bounds written first, nulls run, an honest alarm
recorded — and then measured **synthetic content**. `experiments/headroom/synthetic.py`
generates a 96×128 court, and nothing under `experiments/headroom/` reads
`assets/`. Two things follow, and both are large:

- **The players are the wrong size.** ~4.7% of pixels on the synthetic clip
  against **~2.3% measured from real 4K bboxes** (§2.6 put a single player at
  1.07%). The quantity being measured is *what the players cost*, so getting
  their area wrong by 2× is not a detail.
- **The rung is wrong.** libx264 alone, where the paper compares against
  **AVC/HEVC/AV1/VVC** (§7). And the direction matters: a stronger codec
  compresses the near-static background *better*, so the players take a
  **larger** share of the bits that remain. AVC is therefore the *conservative*
  rung, and 12.2% may be a floor.

**The Wave-3 fork was decided on that number and is withdrawn** (`PLAN.md`
§2.13). Nothing about the project's direction should be settled until this
brief reports.

## The material is on this host — the earlier session missed it, not the data

The reason recorded for skipping real 4K was "no full-frame player masks". That
was wrong:

| Need | Where |
|---|---|
| Real 4K source | `assets/raw_4k/*.mp4` — seven matches, **3840×2160**, already AV1 (libaom-av1, ~11 Mb/s, ~12 min each). Also `assets/real_tennis.mp4`. |
| Player mask in **frame** coordinates | `assets/dataset/<video>/segmentations/<scene>/track_<id>_metadata.json` carries a per-entry `frame_id` and a **`bbox` in full 3840×2160 coordinates**. |
| Per-pixel silhouette | The track crop PNGs are RGBA; alpha is the object mask, at crop size. |

Composite: place the crop's alpha into the frame at its `bbox` and you have the
full-frame player mask. That is all that was missing. `BP18` used the same
sidecars, so there is working code to copy from
(`experiments/probe/player_labels.py` reads them).

**Verify the alignment before trusting a single byte.** Decode the frame named
by `frame_id`, paste the crop back into its `bbox`, and check it reproduces the
source pixels. If it does not, the frame-id convention differs from the
positional one (§2.2 — this dataset has bitten two agents with exactly that) and
everything downstream is measuring the wrong region.

## What to do

### 1. Real frames, real masks

At least **two matches**, at least **one scene each**, enough frames for
inter-prediction to matter (≥ 48). Use the `point` scenes, not `interlude`: the
claim is about play, and an interlude is mostly crowd.

### 2. The full ladder, not one rung

`src/components/codec/` already registers **avc** (libx264), **hevc** (kvazaar),
**av1** (SvtAv1EncApp, native `--roi-map-file`) and **vvc** (libvvenc). Run all
four, or say in the report which refused and why. Resolve each by **path and
version** and record both — this host has carried two builds of one encoder with
different capabilities.

**AV1 is the interesting rung** and the one the paper needs most: the source is
already AV1, so it is also the fairest comparison available.

### 3. Same removals, same brackets

Background-plate inpaint as the honest estimate; flat fill as the bracket. **And
carry `BP13`'s alarm forward**: flat fill *understated* the prize on the
synthetic clip (3.6% against 12.2%) because a grey hole in a green court is a
high-contrast object the encoder spends bits on edges for. Court-median fill
(9.0%) sat between. So report **plate, flat, and median**, and do not describe
flat as an upper bound until a real clip says it is one.

### 4. The background half, against video and not stills

`BP13`'s 17.4× is against a **JPEG-still** baseline, which overstates
conventional cost because real video is inter-predicted. Redo it against an
actual inter-coded background at matched quality. This is the half covering ~98%
of pixels and it deserves the honest comparison.

## Bounds, to be written before the encode

Carry `BP13`'s FG bands (≥25% strong / 10–25% modest / <10% weak) and the BG 10×
bar, and **add the prediction this brief exists to test**:

> A stronger codec should raise the FG share, so **AV1 ≥ HEVC ≥ AVC** on
> FG saving for the same clip. If AV1 comes out *below* AVC, that is an alarm —
> investigate the encode settings and the matched-quality pairing before
> reporting it.

Write per-codec bands before running, and record why any bound was wrong.

## Traps

- **Matched quality, not matched size** (`PLAN.md` §5).
- **A flag existing is not a feature working.** `DEFERRED.md` D3 records AVC
  `addroi` as a silent no-op under QP. This brief does not need ROI — it
  re-encodes modified pixels — but if you reach for an ROI map, drive it and
  measure that the output changed.
- **Report region quality separately.** A frame-level number barely moves when
  the player region is destroyed, which is why it cannot carry this argument.
- **This is a headroom argument, not a result.** It says what is available *if*
  a generator works. Keep mechanism out of the motivating example.
- **4K is slow.** Encode a scene, not a match. Long runs go detached with
  `python -u`, and checkpoint.

## Done when

- FG and BG headroom exist for **≥2 real 4K matches** on **≥2 codec rungs**
  including AV1, with the removal method named and plate/flat/median reported.
- The AV1 ≥ AVC prediction is checked and its outcome recorded either way.
- `sections/problem.tex` carries the real numbers, the provisional paragraph is
  removed, and the `CLAIM` line names the new `outputs/` path.
- `PLAN.md` §2.13's withdrawn fork is **re-decided** on these numbers.

---

## Delivered — 2026-08-23

**The premise holds, measured on real 4K.** Full numbers in `PLAN.md` §2.14;
`outputs/bp20-headroom/`.

Input, first line as the brief demanded: `alcaraz_highlights/scene_000` frames
[38:86] and `federer_djokovic/scene_001` frames [93:141], 3840×2160, 48 frames
each, from `assets/raw_4k/`.

**The correctness gate passed and mattered.** Pasting each crop back into its
sidecar `bbox` reproduces the source at **MAE 0.0** under the
`extract_24_frame_id` convention — and the native-fps and positional conventions
both **failed** it. §2.2 bit for a third time and was caught this time. Nulls:
empty mask 0.0, duplicate-encode ratio 1.0.

**Foreground.** Plate-inpaint BD-rate saving: AVC 0.244 ± 0.017, HEVC
0.234 ± 0.017, AV1 0.229 ± 0.030, VVC 0.167 ± 0.015. Player area by **alpha
silhouette** 0.55% and 1.02%. **A player pixel costs 15–47× an average pixel.**

**Background.** A plate plus homographies (1728 B) saves **34–69%** of the
background bitrate, best on VVC. Not orders of magnitude — inter prediction
already handles a near-static background — but a real second half.

**Bounds that fired, and why they were wrong.** The player-area band was written
on *bbox* area while the measurement correctly used the *alpha silhouette*,
about half of it — wrong by construction. The FG bands were carried from the
synthetic run and were too low. The BG [1.5, 12] gate was derived from the
discredited synthetic JPEG comparison. **Flat fill understates the prize on real
4K too** (0.12 against plate's 0.24), confirming the synthetic alarm on real
content; "flat is an upper bracket" stays void.

**The AV1 ≥ HEVC ≥ AVC prediction fails as stated**, but the first reading of
that — "headroom shrinks as codecs strengthen" — was over-reading n=2. AVC, HEVC
and AV1 agree; VVC is a ~0.077 step down that repeats on both clips. See §2.14
for the two candidate explanations, including the QP-47 confound.

### Left open, deliberately

- **n = 2 clips.** The project's own bar is n ≥ 8, and this is the paper's
  opening argument. Widening it is the first item of the follow-on brief.
- **AV1's background BD-rate is unreported**: PSNR overlap 0.46 and 0.20, below
  the 50% floor. Widen the QP sweep.
- **VVC's QP-47 confound is not ruled out.**
- **`sections/problem.tex` still carries the provisional synthetic paragraph.**
  The rewrite is assigned separately.
- `src/components/background/plate.py` emits `All-NaN slice` from `nanmedian` on
  real 4K masks. Recorded, not fixed.
