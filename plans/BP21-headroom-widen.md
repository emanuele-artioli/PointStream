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

## Delivered

**Input (first line of `outputs/bp21-headroom/report.json`).** Eight `cluster_point` 48-frame windows, 3840×2160, 24 fps, from `assets/raw_4k/`:

`alcaraz_highlights.mp4 scene_000 [38:86] extract_24_frame_id` | `federer_djokovic.mp4 scene_001 [93:141] extract_24_frame_id` | `sinner_alcaraz.mp4 scene_001 [54:102] extract_24_position` | `alcaraz_perricard.mp4 scene_002 [146:194] extract_24_frame_id` | `djokovic_federer.mp4 scene_003 [477:525] extract_24_frame_id` | `djokovic_zverev.mp4 scene_002 [23:71] extract_24_position` | `alcaraz_highlights.mp4 scene_010 [1:49] extract_24_frame_id` | `federer_djokovic.mp4 scene_003 [59:107] extract_24_frame_id`.

n=8 clips, 6 matches (alcaraz_ruud has no 2–30 s point scene on disk, so it contributed none). Round-robin across matches, not eight scenes from one.

### What was done

- Clip chooser now returns ≥8 `cluster_point` scenes from ≥4 matches (`choose_point_scenes` / `iter_point_scenes_spread`). Interludes never enter the pool.
- Paste-back is re-run on every new clip. A clip that fails `PASTE_MAE_MAX=2.0` is dropped with its reason and is not encoded (`load_clips_until`).
- Plate All-NaN columns take the **nearest finite pixel**, not a silent zero. `np.nanmedian`'s warning is swallowed because that fill is the documented follow-up. A plate with no finite pixel raises.
- Common QP set is **32/40/46** for all four codecs. VVC remaps ≥47 → 46 (`qps_for_codec`). `saving_on_interval` / `slice_rd_curve` slice curves in `experiments/headroom/measure.py` before `compare_rd_curves` (metrics package untouched).
- AV1 background arms widen to QPs 24/32/40/46/56/63.
- Bounds copied from the parent's `outputs/bp21-headroom/bounds-before-run.json` (2026-08-24T15:17Z) into `declared_bounds()` and `outputs/bp21-headroom/bounds-stream-a.json`. Stream A agrees with the VVC prediction: the ~0.077 gap should **survive** common-QP / common-interval, in [0.04, 0.10] AVC−VVC.
- Detached encode: `conda run -n pointstream --no-capture-output python -u -m experiments.headroom.real_ladder --out outputs/bp21-headroom` (PID 1135936 at launch, log `outputs/bp21-headroom/run.log`). Checkpoints `report.json` after every codec/clip. Original-arm QP 32/40 reused from BP20 where clip identity matches.

### What was deliberately not done

- Did not touch `67a9ea6275d3d9785ce57026/` (`sections/problem.tex`), `PLAN.md` §2, `plans/README.md`, `src/components/metrics/**`, `src/decoder/**`, `src/shared/**`, `scripts/train_controlnet.py`.
- Did not test third-party encoder binaries, libvvenc empty-bitstream behaviour, or a full 4K encode in CI.
- Did not wait for the n≥8 ladder to finish in this session. 4K is the long pole; the job is running and checkpointing. **n=1 encode is a direction, not a result.** `compare_paired` will refuse a winner until n=8; that refusal is correct.

### Paste-back

Drop list: **empty**. All eight survivors reproduced opaque pixels at MAE 0.0; window MAE [0, 0, 0]. Native-fps never won (MAE 29–127). `extract_24_frame_id` won on six clips. On `sinner_alcaraz/scene_001` and `djokovic_zverev/scene_002` both 24 fps conventions scored 0.0 and `extract_24_position` won the tie; window paste-back still MAE 0.0. Do not assume `extract_24_frame_id` is the only winner — but both 24 fps conventions matching means the clip is still on the source pixels.

### Player area (alpha silhouette, never bbox)

| clip | area |
|---|---|
| alcaraz_highlights/scene_000 | 0.005515 |
| federer_djokovic/scene_001 | 0.010226 |
| sinner_alcaraz/scene_001 | 0.008318 |
| alcaraz_perricard/scene_002 | **0.032659** |
| djokovic_federer/scene_003 | 0.009864 |
| djokovic_zverev/scene_002 | 0.005408 |
| alcaraz_highlights/scene_010 | 0.006713 |
| federer_djokovic/scene_003 | 0.010299 |

**Bound fired:** `alcaraz_perricard/scene_002` 0.0327 is outside the pre-written [0.004, 0.020]. The bound was taken from BP20's two clips (0.55% and 1.02%). This is still alpha (`masks.mean()`), not bbox. A closer point scene on another match is larger. The bound was too tight for match diversity; not retconned. Clip kept (paste-back passed). Concentration on that clip may land below 10× — check when its FG cell lands.

### Plate-NaN effect on BP20

Recomputed plate on `alcaraz_highlights/scene_000` and re-encoded the AVC plate arm at QP 32/40/48 against reused BP20 original bitstreams.

- BP20 plate saving: 0.26023
- BP21 plate saving: 0.25995
- **ΔFG = −0.00028** (expect |Δ| ≤ 0.01; alarm if > 0.02)

No correction to publish. Holes were canvas-margin / all-masked columns; nearest-valid vs silent zero does not move matched-quality BD-rate here. Instrument: PSNR ~20–50 dB; rate = payload bytes.

### First encode cell (n=1 — not citable)

AVC, `alcaraz_highlights/scene_000`, common QP **32/40/46**, ffmpeg `/opt/local/bin/ffmpeg` n7.1.1-56-gc2184b65d2 (libx264 via ffmpeg).

| arm | saving (BD-rate, matched quality) |
|---|---|
| plate vs original | **0.264** (overlap 33.89–42.18 dB, 99%) |
| flat vs original | 0.164 |
| median vs original | 0.156 |
| BG plate+homog vs intercoded plate | **0.349** (overlap 34.41–42.29 dB, 94%; homog 1728 B) |

BP20 on this same clip at QP 32/40/48 was plate 0.260 / BG 0.344. The common-QP 46 point moved the number by ~0.004. Inside the ±0.06 band. **n=1. Do not generalise.** Flat still understates plate.

VVC sentence: **not yet measured.** Prediction, written before any BP21 encode: the gap survives, AVC−VVC in [0.04, 0.10]. After the VVC cells land, one sentence in `report.json` `summary.vvc_gap.sentence` ("codec" vs "confound").

AV1 BG: **not yet measured.** Wider QP sweep is wired; overlap fractions will be in each `bg_intercoded.av1.*.bd_vs_conventional` cell, or a BD-rate if overlap ≥ 50%.

Nulls: scheduled on a non-BP20 clip after all codecs; not yet run.

### Tools (resolved by path+version)

- ffmpeg/ffprobe: `/opt/local/bin/ffmpeg`, `ffmpeg version n7.1.1-56-gc2184b65d2`
- AVC encoder: same ffmpeg (libx264)
- HEVC / AV1 / VVC: recorded per cell when those codecs run (`resolved_tools`)

### CI

PR https://github.com/emanuele-artioli/PointStream/pull/18
`pointstream-ci` run **32746233019** watched green (tests 3m33s, typecheck 4m12s, lint 2m29s). Node 20 deprecation annotations only.

### Outside this stream's files

- `alcaraz_ruud` point scenes are 120–370 s, above the 30 s extract cap, so that match is absent. Six matches still satisfy ≥4.
- Two clips tie at MAE 0.0 on both 24 fps conventions; winner is dict-order (`extract_24_position`). Worth a tie-break toward `extract_24_frame_id` if a later stream touches diagnosis; window MAE already gates the region.
- Encode crashed 2026-08-24T20:58Z on `alcaraz_highlights/scene_010` original:
  leftover QP 32/40 bitstreams from 03:49 (266601 / 125198 B) mixed with a new
  QP 46 (141623 B). Rates were not monotonic; the check aborted. Six AVC clips
  were already checkpointed. Reuse now copies only a complete QP set; a partial
  arm is deleted and re-encoded together. The poisoned `scene_010` originals
  were wiped and the ladder resumed. `PLAN.md` §2.14 is not edited here.

### Tests landed (approved bug-fix cases only)

1. All-masked plate column: no All-NaN warning, nearest-valid fill, not zero.
2. Chooser returns ≥8 point scenes from ≥4 matches (mocked listing).
3. Paste-back failure is recorded and the clip is dropped, not encoded.
4. Common-interval BD-rate helper: two shifted two-point curves integrate only on the overlap (hand-computed saving ≈ 0.3675).

5. `seed_reuse` copies a complete QP set and deletes a partial 32/40 leftover
   rather than stitching on a new 46.

Deliberately not tested: encoder binaries, libvvenc empty bitstreams, full 4K encode.
