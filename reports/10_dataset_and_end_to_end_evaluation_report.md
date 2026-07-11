# Dataset Curation & End-to-End Full-Match Evaluation — Report

*Status: Active | Last updated: 2026-07-11 | Code: scripts/process_dataset.py, src/main.py (gaps: see §Findings)*

## Scope

This report owns the **end-to-end story** of POINTSTREAM as a system, which
until now existed only as fragments across reports 2–5 and 7:

1. **Dataset curation as a contribution** — `assets/raw_4k` (7 full 4K tennis
   videos) processed into `assets/dataset` with high-quality (slow) models:
   scene segmentation, per-scene motion metrics, point/interlude
   classification, object tracking, RGBA segmentations, BLIP captions, Canny
   maps, and DWPose skeletons. Model-generated, but manually supervised and
   iteratively tweaked.
2. **GenAI finetuning** on that dataset — ControlNet variants (pose, seg,
   ip-adapter, custom/canny), SPADE4Tennis, Pix2Pix, Animate-Anyone (and
   possibly a Multi-ControlNet fusion, currently deferred).
3. **Full-match runtime evaluation** — feed an *entire* raw_4k video to
   POINTSTREAM, which uses the *same algorithm suite tuned for speed*
   (yolo26**n** at runtime vs yolo26**x** at curation time), splits it into
   scenes, routes interludes to a baseline codec and points to the semantic
   pipeline, then compares aggregate bitrate/quality **and encoding speed**
   against pure-baseline encoding — the speed/compression trade-off across
   model-complexity tiers.

Per-clip Residual-Guarantee ablations stay in report 8; codec anchor curves
stay in report 9. This report owns everything needed to go from "one 2 s
chunk of `real_tennis.mp4`" to "a whole broadcast match, end to end."

## Current state (TL;DR)

**As of 2026-07-10 this vision was not fully documented anywhere, and its
runtime half was not implemented.** As of 2026-07-11 (Phases 1–3a + Phase 4
G1), the runtime half exists and has been validated on real raw_4k data;
Phase 3b and Phase 4 G2–G4 remain open. Gap analysis (kept as a historical
record of what changed, not re-collapsed into one "done" row):

| Piece | Documented? | Implemented? |
|---|---|---|
| Dataset curation pipeline mechanics | ✅ Reports 2 (scene classification), 3 (pipeline unification), 4 (universal RGBA+DWPose format), 5 (canny/captions) | ✅ `scripts/process_dataset.py` (stages: classify, segment, pose, skeleton, canny, caption) |
| Dataset **as a catalogued contribution** (inventory, stats, curation methodology, coverage caveats) | ✅ §Dataset inventory below (2026-07-10) | ✅ data exists on disk |
| GenAI finetuning on the dataset | ✅ Reports 1, 4, 5 | ✅ checkpoints in `assets/weights/` (pose/seg/ip-adapter/custom controlnets, `spade4tennis_lite_generator.pt`, pix2pix); Multi-ControlNet **deferred** (report 7 §4) |
| **Runtime** scene classification + interlude/point routing | ✅ this report, Phase 1 (2026-07-11) | ✅ `src/shared/scene_classification.py`, verified byte-faithful against real cached data (Phase 1) |
| Full-match orchestrator (scene split → route → per-scene encode → aggregate accounting) | ✅ this report, Phase 2 (2026-07-11) | ✅ `src/encoder/match_orchestrator.py`, outcome-safe routing; validated on a real raw_4k excerpt with real evidence (Phase 4 G1, 2026-07-11) |
| Speed/compression trade-off across model-complexity tiers | ✅ this report, Phase 3a (2026-07-11) | ⚠️ 3 tier configs + realtime factor + anchor-encode cache done (Phase 3a); variant-ladder sweep harness, DAG intermediate cache, GPU fan-out **deferred to G3** (Phase 3b) |

The implementation plan to close these gaps is in §Implementation plan.
**Speed reality check (2026-07-11):** profiled in FPS terms, the full
GenAI pipeline runs at ~0.09 fps encode / ~0.06 fps decode at 4K against a
12 fps source — every stage is 5–150× off real time, so §Phase 5 defines
the combined speed/real-time + gated-training campaign (six workstreams,
parallel-session split included).
**Methodology locked 2026-07-11** (see that findings entry): held-out test
split = `alcaraz_highlights` + `djokovic_zverev` (no cross-validation);
anchors encoded post-hoc on PointStream's own scene spans; scene cuts
deterministic and shared across all tiers; ablations unified with speed
sweeps as per-component variant ladders with "off" as the leftmost rung;
LPIPS + FVD added alongside PSNR/SSIM/VMAF(4K model); routing made
outcome-safe via the server-side residual check.
**FVD implemented 2026-07-11** (`src/shared/fvd.py`, I3D R50/Kinetics-400
via `pytorchvideo`, wired into `evaluate_run_summary()`; see the dated
entry below) — real-data verified, `FVD(self,self)=0.0`,
`FVD(self,degraded)=7.20` on `assets/real_tennis.mp4`. **LPIPS is still
unimplemented** — despite this section reading as a package, only FVD was
built; a prior claim that LPIPS was "already backfilled" was checked and
found false (no `lpips` code anywhere in `src/`).

## Dataset inventory (as of 2026-07-10)

### Raw sources — `assets/raw_4k` (15 GB, gitignored)

All 3840×2160; `alcaraz_ruud.mp4` probed at 59.94 fps (others assumed
similar broadcast rates). Durations/sizes via ffprobe:

| Video | Duration | Size |
|---|---|---|
| alcaraz_highlights.mp4 | 8 m 14 s | 0.72 GB |
| alcaraz_perricard.mp4 | 29 m 52 s | 3.88 GB |
| alcaraz_ruud.mp4 | 11 m 59 s | 1.92 GB |
| djokovic_federer.mp4 | 78 m 35 s | 4.23 GB |
| djokovic_zverev.mp4 | 6 m 14 s | 2.78 GB |
| federer_djokovic.mp4 | 9 m 55 s | 0.97 GB |
| sinner_alcaraz.mp4 | 12 m 04 s | 1.02 GB |
| **Total** | **≈ 2 h 37 m** | **15.5 GB** |

### Curated dataset — `assets/dataset` (5.3 GB, gitignored)

Produced by `scripts/process_dataset.py` with the **quality tier**
(defaults `--seg-model yolo26x-eg.pt`, `--pose-model yolo26x-pose.pt`).
Per-video layout: `scenes/` (classification thumbnails whose filenames
encode duration + avg/std/max motion score + class + confidence),
`scene_scores.csv` (per-frame scene score), `scene_metadata.json` (cluster
stats), and `segmentations/scene_NNN/track_MMMM{,_skeleton,_canny}` PNG
sequences plus `track_MMMM_{caption,keypoints,metadata}.json`.

| Video | Scenes | Point scenes | Deep-annotated tracks |
|---|---|---|---|
| alcaraz_highlights | 71 | 20 | 20 |
| alcaraz_perricard | 110 | 90 | 14 |
| alcaraz_ruud | 6 | 3 | 4 |
| djokovic_federer | 632 | 224 | 20 |
| djokovic_zverev | 8 | 8 | 16 |
| federer_djokovic | 65 | 26 | 20 |
| sinner_alcaraz | 60 | 28 | 20 |
| **Total** | **952** | **399** | **114** |

**Coverage caveat:** the GPU stages default to `--max-scenes 10` point
scenes per video, so the deep annotations (segmentations/skeletons/canny/
captions) cover a *subset* of point scenes — classification metadata covers
all 952 scenes. Labels are model-generated (not human ground truth) but the
process was manually supervised and iteratively tuned; the paper should
state exactly that.

### Trained checkpoints — `assets/weights/`

`custom-controlnet/`, `pose-controlnet/`, `seg-controlnet/`,
`ip-adapter-controlnet/` (trained via `scripts/train_controlnet.py`, whose
dataset loader reads `*/segmentations/scene_*/track_*` directly),
`spade4tennis_lite_generator.pt`, `pix2pix_generator.pt` /
`pix2pix_checkpoint.pt`; Animate-Anyone fork weights under
`/…/Models/AnimateAnyone` (path per weights policy — symlink, don't cite in
user docs). No Multi-ControlNet checkpoint exists.

## Findings log

### 2026-07-10 — End-to-end vision audit: documented fragments, missing spine

**Problem/Question:** Is the full pipeline story (dataset contribution →
finetuning → full-match runtime with fast models → interlude/point routing →
speed/compression trade-off) documented in `reports/` and implemented?

**Diagnosis/Evidence:**
- `grep -rn -i "interlude\|scene_class\|classify" src/` → **no matches**.
  Scene classification exists only in `scripts/process_dataset.py`
  (`classify_scenes`, point-anchored thresholding from report 2).
- `src/main.py` ingests a single clip as one `VideoChunk` (config
  `num-frames`); there is no multi-scene driver, no fallback-codec routing,
  no whole-video payload aggregation.
- `run_summary.json` (e.g. `outputs/20260710_211734_419673/`) has
  `evaluation.timings_sec` per stage but no realtime factor or fps-vs-source
  metric; `config/` has no fast/balanced/quality tier presets.
- Speed tiering exists implicitly: `src/encoder/actor_pipeline.py` defaults
  `yolo26n.pt`/`yolo26n-pose.pt`/`yolo26n-seg.pt` while
  `scripts/process_dataset.py` defaults `yolo26x-eg.pt`/`yolo26x-pose.pt` —
  never stated as a deliberate design axis in any report.
- No report inventoried `assets/raw_4k`/`assets/dataset` or framed the
  curation as a paper contribution; reports 2–5 document the *mechanics* as
  engineering history.
- Multi-ControlNet: user-stated as part of the training suite, but report 7
  §4 defers it and no code/checkpoint exists (only the four single-condition
  controlnets).

**Resolution:** This report created to own the area; dataset inventoried
(§above); implementation plan drafted (§below); REPORTS.md dashboard updated
with the two new workstreams. Code gaps remain **open** — see plan.

**Paper impact:** Unlocks three paper elements: (a) the dataset-contribution
subsection (feeds report 7 §3.3), (b) the full-match evaluation table
(aggregate bitrate vs AV1/HEVC from report 9's sweep, answering R2/R5 at
match scale rather than clip scale), (c) the speed/compression Pareto figure
across model tiers (answers R3 real-time concerns honestly: "not real time,
here is the trade-off curve").

### 2026-07-11 — Evaluation methodology locked; experiment-efficiency architecture

**Problem/Question:** Three methodology risks threatened the Phase 4
experiment: (a) runtime scene cuts won't match dataset scene cuts, so how to
compare per-scene BD-rate without a "you picked the easiest scene"
objection; (b) train/test contamination — the generative models are
finetuned on the same seven videos; (c) the speed/quality sweep (many
components × many variants) plus the component ablations looked
combinatorially expensive and hard to condense into clear results.

**Diagnosis/Evidence:**
- Anchors don't need pre-agreed cuts: AV1/HEVC can encode any frame span, so
  the baseline can be derived *from* PointStream's runtime segmentation
  post-hoc. Cherry-picking is refuted by reporting **all** scenes (the
  per-scene win/loss distribution) plus the aggregate, not a chosen scene.
- Scene-score extraction is already computed and cached for all 7 full
  videos (`assets/dataset/*/scene_scores.csv`, 952 scenes); thresholding
  over cached scores is deterministic and ~free. Cuts can therefore be made
  bit-identical across tiers and runs.
- An ablation ("component off") is the degenerate fastest variant of that
  component — the two campaigns are one campaign.
- 4K frame decode dominates GPU extractor cost, so multiple model variants
  can share one decode pass if inference is timed per-variant.
- Leave-one-match-out CV would require retraining every generative model
  7×; no codec/generative-video venue expects it.

**Resolution (decisions, agreed 2026-07-11):**
1. **Post-hoc anchor protocol:** PointStream cuts at runtime; every
   resulting scene span is also encoded with AV1/HEVC across a CRF ladder;
   compare per-scene at matched VMAF + aggregate. Two anchor forms
   reported: whole-match continuous encode (deployment-honest, boundary
   alignment irrelevant) and per-scene segmented encode (isolates the
   semantic contribution; slightly handicaps the anchor via per-segment
   keyframes — state it).
2. **Held-out split:** `alcaraz_highlights` (diverse lighting/players/
   court conditions, stresses routing via fast cuts and replays) **and**
   `djokovic_zverev` (typical continuous match). All other videos train.
   Caveat for the paper: this tests unseen scenes/conditions, not unseen
   players (Alcaraz/Djokovic appear in training videos) — acceptable for
   the tennis-constrained scope, stated explicitly. **No cross-validation.**
   Also state: scene-classifier thresholds are per-video statistical
   (point-anchored), not learned, hence self-calibrating on unseen videos.
3. **Deterministic shared segmentation:** one cached scene segmentation per
   video shared by all tiers and runs (tiers differ in extractors/engines,
   never in cuts). Cut-detection wall-clock is measured once and charged
   into every tier's realtime factor. Only a deliberate cut-detector
   ablation may change cuts, paying one fresh anchor ladder.
4. **PointStream BD-rate** via sweeping the residual CRF (metadata ≈
   constant); the metadata floor is shown honestly — report BD-rate over
   the overlapping quality range and the crossover point below which
   anchors win.
5. **Outcome-safe routing:** the server already computes the true residual,
   so per scene it compares the semantic payload against the fallback
   encode and transmits whichever is smaller — the Residual Guarantee
   extended to routing; caps misclassification damage at zero. Paper point.
6. **Metrics:** PSNR/SSIM + VMAF **4K model** (default model is
   1080p-trained) + **LPIPS** (on a consistent downsample/crop — 4K LPIPS
   is expensive) + **FVD** (standard I3D protocol; per-scene clips are the
   population). MOS stays deferred; LPIPS/FVD is the substitute for
   generative-mode degradation that VMAF wasn't trained on.
7. **Whole-match comparison across representations:** compare total bytes
   (Σ per-scene payloads **including** per-scene container overhead +
   routing flags vs the single anchor file) and frame-aligned quality over
   the concatenated timeline (scenes tile the video completely). Guard with
   a frame-count-invariant test at scene boundaries.
8. **Goal-driven sequencing** (run the minimum that validates each claim):
   G1 plumbing (orchestrator on a held-out video, tier_fast, accounting
   adds up) → G2 headline (balanced tier vs anchors on held-out videos,
   BD-rate table — needs no sweeps) → G3 trade-off (variant ladders →
   composed tiers → Pareto figure) → G4 ablations (free from G3's "off"
   rungs, plus the routed-vs-all-semantic routing ablation).
9. **Dataset expansion & augmentation deferred until a data-hunger signal**
   (held-out quality ≪ training-video quality). When needed: process more
   of the 389 unannotated real point scenes (scores already cached; only
   GPU stages run) before any synthetic augmentation — flips risk breaking
   handedness-dependent racket heuristics.

**Paper impact:** This *is* the evaluation-protocol section: held-out
split, anchor definition, BD-rate protocol, metrics suite, and the routing
safety valve as a methodological contribution.

### 2026-07-11 — Per-stage FPS profile; combined speed/real-time + training campaign (Phase 5)

**Problem/Question:** G1 measured the pipeline at ~15–20 min per 2 s
sub-chunk. The user directed a systematic response: profile every stage in
**FPS terms** (raw seconds mislead across different chunk lengths/framerates),
re-evaluate each stage for inefficiency before assuming concurrency solves
it, add spatial *and* temporal down-processing knobs, make the panorama
stateful across scenes (or replace it with an ROI-weighted background
video), and run G2 retraining as a gated multi-variant campaign instead of
one long blind run.

**Diagnosis/Evidence:** single run (not swept),
`outputs/20260710_234603_892275/run_summary.json` — 60 frames of
`assets/real_tennis.mp4` (3840×2160 @ 12 fps, i.e. 5 s of video),
`genai_backend: canny-controlnet`, single chunk. Converted to throughput:

| Stage | Wall-clock | FPS | vs 12 fps source |
|---|---|---|---|
| encode total | 679.0 s | 0.088 | 136× too slow |
| decode total | 931.1 s | 0.064 | 186× too slow |
| genai_baseline (encoder) | 363.8 s | 0.16 | |
| genai_baseline (decoder) | 786.5 s | 0.076 | **2.16× the encoder's — suspect** |
| panorama | 130.5 s | 0.46 | |
| decode video_encoding (final 4K write) | 114.0 s | 0.53 | **suspect for an FFmpeg write** |
| detection | 60.6 s | 0.99 | |
| residual computation | 39.8 s | 1.5 | |
| ball | 38.9 s | 1.5 | |
| segmentation | 24.4 s | 2.5 | |
| composite (decoder) | 29.6 s | 2.0 | |

Reading: no single villain — *every* stage is 5–150× off real time at 4K,
so concurrency (tagged pool) alone cannot close the gap; per-stage
efficiency and resolution/framerate reduction are both required. Two
anomalies deserve diagnosis before optimization: (a) the **same**
canny-controlnet engine on the **same** 60 frames costs 2.16× more on the
decoder side than the encoder side — cost asymmetry in what must be a
byte-identical computation is also a cheap tripwire for a symmetry
(Residual Guarantee) violation, so the diagnosis must include a
bit-identity check of server-vs-client generated frames; (b) the decoder's
final video write at 0.53 fps suggests FFmpeg threading/preset flags are
not propagating. Also noted: `run_summary.json` does not echo the config
that produced it (the `config` key is absent) — a profiling campaign needs
that closed.

**Resolution:** plan locked as **§Phase 5** (below) — six workstreams
(contract fix; execution & profiling; resolution/framerate ladders;
background-layer variants; gated G2 training campaign; Phase 3b harness,
promoted), with an explicit parallel-agent-session split and worktree
isolation required (process lesson from the G1 concurrency clobber).
*Amended later the same day:* a seventh workstream (5.6,
residual-compression matrix) and a widened 5.0 were absorbed from the
Gemini-session `reports/implementation_plan.md`, which also exposed that
the earlier dynamic-thresholding "done" claim was invalid — full
diagnosis in report 8's 2026-07-11 entry.
Key design decisions folded in:

1. **FPS is the canonical speed unit.** Every stage timing in
   `run_summary.json` gains a derived `fps_throughput`; realtime factor is
   computed against the *source* framerate (the 12 fps eval asset flatters
   us ~2–5× vs 25–60 fps broadcast).
2. **Temporal decimation is a first-class ladder rung**, not a hack: actor
   motion already supports sparse keyframes + client interpolation
   (`keyframe`/`interpolate`/`static` events, `src/shared/schemas.py`); the
   extension is a deterministic frame interpolator *inside*
   `SynthesisEngine` so decimated-fps residuals stay guarantee-exact.
   Caution flagged: the ball is the fastest object in frame — ball
   extraction may need full-rate input even when other stages decimate.
3. **Spatial down-processing likewise:** a global processing-resolution
   key (semantic metadata is coordinates, hence resolution-independent) +
   a deterministic upsampler inside `SynthesisEngine`, making full-res
   residuals a *layer* priced by the Residual Guarantee (LCEVC-style
   scalability, same framework as every other component).
4. **Background layer becomes a ladder**, not an a-priori choice:
   `panorama-static` (today) → `panorama+delta` (stateful panorama, send
   full once then per-scene deltas — scoreboard/crowd updates ride the
   delta; both sides hold identical panorama state, so it is
   guarantee-compatible) → `roi-video` (actors masked out, very low
   bitrate, FFmpeg `addroi` quality boost on umpire/ball-kids/scoreboard —
   note `addroi` is consumed by libx264/libx265, **not** libsvtav1).
   `benchmark_matrix` decides which rung pays.
5. **GenAI speed ladder = the G2 training campaign.** If Animate-Anyone
   cannot be accelerated without destroying it, we don't fight it — the
   already-planned engine roster *is* the speed ladder: SPADE4Tennis
   (single forward pass — the fast rung by construction, a new reason to
   keep report 1 alive), ControlNet at reduced denoising steps (mid),
   Animate-Anyone (quality). Training multiple variants for G2 and
   producing G3's speed/quality trade-off rungs are the same work.
6. **G2 training is gated, not blind:** fixed probe set drawn from
   *training-split* videos only (held-out stays untouched), checkpoint
   evaluation (PSNR/SSIM/LPIPS on probes) every N steps, successive
   halving across variants — prune losers early, give survivors the GPU.

**Paper impact:** feeds the G3 Pareto figure (x = realtime factor) and adds
two candidate paper points: the layered spatial/temporal scalability under
the Residual Guarantee, and the outcome-safe background-layer ladder.

### 2026-07-11 — FVD implemented and wired into evaluation; LPIPS still unimplemented

**Problem/Question:** this report's methodology lock (§2026-07-11 above)
named "LPIPS + FVD added alongside PSNR/SSIM/VMAF" as a package. A follow-up
task assumed LPIPS was already done ("we already have LPIPS backfilled")
and asked only for FVD on top. That assumption was checked and is false: a
repo-wide grep of `src/` for `lpips` found no implementation anywhere — the
only hits were this report's own planning language (here and at the old
line ~814) describing LPIPS/FVD as methodology *to add*, not code that
exists. **Correction: as of this entry, FVD is implemented; LPIPS is not.**

**Diagnosis/Evidence:** `/home/itec/emanuele/Models` had no I3D checkpoint
of any kind (`find ... -iname "*i3d*"` empty), so the standard "search
Models first" step came up empty and a real fetch was required.
`pytorchvideo`'s I3D R50 (Carreira & Zisserman architecture, Kinetics-400,
the "8x8" checkpoint — 8 frames at temporal stride 8) was downloaded from
`https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D_8x8_R50.pyth`
(224 MB) and cached under the shared host `Models/i3d/` directory, symlinked
to `assets/weights/i3d_r50_kinetics.pyth` per the weights convention (never
the absolute host path). New pip deps in `pyproject.toml`:
`pytorchvideo==0.1.5`, `fvcore==0.1.5.post20221221`, `iopath==0.1.10`.

**Resolution:**
- `src/shared/fvd.py` — the metric: `frechet_distance`/
  `compute_feature_statistics` (pure numpy/scipy math, unit-tested with
  synthetic feature arrays, no model load required — mock-first per
  CLAUDE.md), `sample_clip_frame_indices`/`preprocess_frames_for_i3d`
  (BGR→RGB, Kinetics normalization, 8×8-stride clip windows with a
  single-clip fallback for videos shorter than 64 frames), and
  `I3DFeatureExtractor` (lazy-loaded I3D R50, classification head replaced
  with `nn.Identity()` to expose the pooled 2048-d pre-classifier feature).
- `src/experiment_evaluation.py` — `_compute_fvd` follows the existing
  `_compute_psnr`/`_compute_ssim_ffmpeg`/`_compute_vmaf_ffmpeg` pattern
  (existence checks, exceptions caught into a `note` field rather than
  raised) and is wired into `evaluate_run_summary()`'s metric dispatch and
  `_normalize_evaluation_metrics`'s allowed set (`"fvd"`). Unlike the other
  three, it decodes real frame tensors via `video_io.decode_video_to_tensor`
  (ffmpeg subprocess, not opencv — sidesteps the same AV1-decode gap noted
  in `_compute_psnr`'s docstring) rather than only shelling out to ffmpeg
  filters, since I3D needs actual pixel tensors.
- `scripts/download_weights.py` gained `ensure_fvd_i3d_weight()`, which
  actually downloads (the rest of the script only presence-checks and
  errors with instructions) since the I3D URL is stable and public.
- Tests: `tests/test_fvd.py` (fast, pure math + clip sampling — Frechet
  distance is 0 for identical distributions, grows with a synthetic mean
  shift, is symmetric, BGR→RGB swap verified numerically),
  `tests/test_experiment_evaluation_fvd.py` (fast, wiring + missing-file/
  decode-failure error paths via monkeypatch, mirroring
  `test_experiment_evaluation_coverage.py`'s existing PSNR/SSIM/VMAF
  pattern), `tests/test_fvd_integration.py` (`pytestmark = [pytest.mark.
  integration, pytest.mark.slow]`, real I3D weights + real video decode,
  self-comparison ≈ 0 and a degraded copy scoring clearly higher).
- **Real verification** (not just unit tests), run from this worktree:
  `assets/real_tennis.mp4` (3840×2160 @ 12 fps, 60 frames — below the
  64-frame single-clip window, so this exercises the degenerate/fallback
  covariance path) against itself and against a copy degraded by 20×
  downscale/upscale + additive noise + `libx264 -crf 45`:
  `FVD(self, self) = 0.0`, `FVD(self, degraded) = 7.20` — clearly higher,
  non-degenerate. Separately, `assets/scene_004.mp4` (3840×2160 @ 50 fps,
  248 frames, ≥3 full 64-frame windows) exercised the **non-degenerate
  multi-clip covariance path**: 3 clips per side, `FVD(self, self) = 0.0`,
  no fallback-path note, ~74 s of compute (model load + feature extraction
  + 2048×2048 `scipy.linalg.sqrtm` twice) after a ~51 s 4K ffmpeg decode.
  Full commands and output are in this session's transcript; not re-run
  under `pytest` at 4K scale (that belongs in `test_fvd_integration.py`'s
  smaller `real_tennis_10f_video` fixture instead, which was the actual
  regression test added).

**Paper impact:** FVD is now a real, wired-in metric for the G2 headline
comparison and the LPIPS/FVD generative-degradation substitute for VMAF
named in the methodology lock — only the LPIPS half of that pairing remains
open. No G2 numbers exist yet (blocked on the retraining gate, unchanged by
this entry); this closes the *tooling* gap, not a headline claim.

## Experiment-efficiency architecture

The design that keeps the G3/G4 campaign linear instead of combinatorial:

- **Variant ladders unify sweeps and ablations.** Every component declares
  an ordered ladder `off → fastest → … → quality` (segmenter:
  off/yolo26n-seg/yolo26s-seg/yolo26x-eg; likewise detector, pose, ball
  extractor, mask codec, panorama quality, keypoint event frequency,
  residual CRF/preset, GenAI engine + denoising steps + generation
  resolution, fp16/fp32, transport compression, captions on/off). Each rung
  is measured on payload (metadata + residual) and stage wall-clock, swept
  **one component at a time** around a fixed reference config. The ablation
  verdict (pays-for-itself) and the per-component speed/quality curve are
  the same data. Paper format: one repeated figure template per component,
  "off" leftmost.
- **Content-addressed intermediate cache** on the DAG orchestrator: stage
  outputs keyed by hash of (input span, stage config version, upstream
  keys), stored under `outputs/cache/<stage>/<key>/`. A sweep recomputes
  only the swept stage and its downstream cone (the residual always
  recomputes — it is the measurement).
- **Multi-variant fan-out** for GPU extractor stages: decode 4K frames
  once, run all rungs of the swept stage on the shared batch, write each
  rung under its own cache key. **Guard:** per-variant inference is timed
  individually; shared decode is counted once as pipeline overhead —
  otherwise fan-out corrupts the speed numbers it exists to produce.
- **Anchor-encode cache** keyed by
  `(video, start_frame, end_frame, codec, crf, preset)`. Because cuts are
  deterministic and shared (decision 3), the hit rate is ~100% after the
  first ladder per video — anchors are encoded once per video, ever.
  Anchor encodes are CPU-only FFmpeg and parallelize freely.
- **Composed tiers, predicted then verified:** tier configs pick per
  component the cheapest rung that doesn't blow up the residual; tier
  realtime factor is *predicted* as Σ of individually measured stage times
  and *verified* by one end-to-end run per tier per match. Encoder-side and
  decoder-side realtime factors are reported separately (broadcast
  feasibility vs client feasibility).

## Implementation plan

Ordered so each phase is independently verifiable; all follow mock-first and
the Residual Guarantee framing.

### Phase 1 — Runtime scene classifier in `src/` — **done (2026-07-11)**
Ported the scene-score extraction + point-anchored thresholding out of
`scripts/process_dataset.py` into `src/shared/scene_classification.py`,
split into pure functions (`find_candidate_cuts`, `filter_false_cuts`,
`classify_scene_stats`, `to_scene_spans`) plus the orchestrating
`detect_and_classify_scenes`/`classify_video_scenes`, mirroring how
`racket_heuristic.py`/`player_extraction.py` were unified in report 3.
Added `SceneClass` (point/interlude/other/blank) and `SceneSpan` to
`src/shared/schemas.py`. `scripts/process_dataset.py:classify_scenes` now
calls the shared module instead of duplicating the algorithm — net -444
lines in that script (461 removed, 17 added); the dead local
`get_video_duration`/`extract_scene_scores`/`load_scene_scores` and unused
`find_peaks`/`re`/`csv` imports were removed with it.
**Design rule:** scene segmentation is deterministic, computed once per
video, cached, and shared by all tiers/runs — tiers never differ in cuts
(methodology decision 3).

**Verification:** `tests/test_scene_classification.py` — 6 unit tests
(GMM-path clustering with explicit synthetic cut boundaries, small-N
fallback branch, blank-scene exclusion, `to_scene_spans` mapping, CSV
round-trip) plus 1 `integration`-marked regression test that runs
`detect_and_classify_scenes` on the real, already-cached
`assets/dataset/alcaraz_ruud/scene_scores.csv` (fed by
`assets/raw_4k/alcaraz_ruud.mp4`) and asserts it reproduces
`scene_metadata.json`'s `scenes` list exactly (`random_state=42` GMM +
identical false-cut filtering) — **passed**, confirming the refactor is
behavior-preserving. Also ran `python -m scripts.process_dataset --input
assets/raw_4k/alcaraz_ruud.mp4 --stages classify` for real (metadata moved
aside to bypass the already-exists skip guard): regenerated
`scene_metadata.json`'s `scenes` list and `cluster_info` per-cluster
content were identical to the pre-refactor original; the only diff was
`cluster_info`'s dict key order, an artifact of Python's string-hash
randomization in the (pre-existing, unchanged) `list(set(...))` call, not a
behavior change. Original file restored byte-for-byte afterward. Full
`pytest -q` suite (excludes `integration`/`slow`) and
`ruff check`/`mypy` on all touched files: clean.

### Phase 2 — Full-match orchestrator with interlude routing — **done (2026-07-11)**
Built `src/encoder/match_orchestrator.py` (config-first, per CLAUDE.md: no
new CLI flags — `run_mode: full_match` + `scene_chunk_duration_sec` added
to `PointstreamConfig`, dispatched from `src/main.py:run_cli`). Takes a
full raw_4k video, splits it into scenes via Phase 1's shared classifier,
then routes: **interludes/other/blank → the pipeline's own FFmpeg wrapper**
(`encode_video_frames_ffmpeg` at the configured `ffmpeg-codec`/`codec-crf`/
`codec-preset`, whole scene span, no semantic attempt — no GPU work wasted
on them); **points → the existing chunked semantic pipeline**, split into
`scene_chunk_duration_sec` (default 2 s) sub-chunks, each a real
`VideoChunk` run through `EncoderPipeline`/`DiskTransport`/
`DecoderRenderer`. Model construction was extracted from `src/main.py`'s
private `_build_*` helpers into public `src/encoder/pipeline_builders.py`
functions (also used by `run_pipeline`, unchanged behavior) so
`match_orchestrator` builds each component **once** and reuses it across
every sub-chunk in the match, instead of reloading YOLO/pose/segmenter
weights per chunk. **Outcome-safe routing** (methodology decision 5):
every point sub-chunk's semantic transport-total bytes are compared
against a fallback-codec encode of the *same* clip
(`choose_routing` — whichever is smaller is what would actually be
transmitted); a scene-level `routing_summary` (`semantic`/`fallback`/
`mixed`) is recorded per point scene from its sub-chunks' individual
choices. Sub-clip extraction (`_extract_scene_clip`) is lossless
(two-stage-seek, `libx264 -crf 0`) — the Residual Guarantee requires the
server to diff against the *actual* original pixels, so this intermediate
must not itself lose information. `assert_scenes_tile_video` is the
frame-count/duration invariant guard (scenes must partition
`[0, video_duration]` with no gaps/overlaps beyond a tolerance).

**Scope note (matches the report's own goal-driven sequencing):** this is
G1 scope only — byte/timing accounting, no PSNR/SSIM/VMAF/LPIPS/FVD.
Quality scoring is G2 (Phase 4), gated on retraining without the held-out
videos.

**Verification:** `tests/test_match_orchestrator.py` (19 unit tests:
sub-chunk splitting, outcome-safe routing decision, the tiling invariant
including gap/overlap/missing-tail/unsorted-input cases) +
`tests/test_match_orchestrator_coverage.py` (11 fast, mocked tests —
`EncoderPipeline`/`DiskTransport`/`DecoderRenderer`/the 5 builders faked,
scene classification fixed to 3 synthetic scenes — covering the point/
interlude dispatch, frame-cursor accumulation across scenes, byte
aggregation, zero-frame-clip skip, and error paths) +
`tests/test_pipeline_builders.py` (12 fast tests covering all 5 builders'
branches) + `tests/test_match_orchestrator_integration.py` (real,
non-mocked run on a real ~3.4 s clip from `assets/real_tennis.mp4` — real
ffmpeg scene-score extraction/classification, real yolo26n detector/pose/
segmenter, real residual computation, real fallback encode, real
outcome-safe comparison — **passed in 101 s**). `src/main.py`'s builder
refactor was behavior-preserving: `tests/test_main_coverage.py`,
`test_integration_main.py`, `test_end_to_end.py` all still pass (two
monkeypatch targets renamed to match the new public builder names).
Full `pytest -q` (265 passed, 1 skipped, 7 deselected) and
`ruff check`/`mypy` on all touched files: clean.

**Coverage note:** `scripts/check_coverage_gate.py` (excludes
`integration`/`slow`, matching pytest.ini) reads 83% total — passes CI's
80% threshold but not the stricter 85% local default
(`POINTSTREAM_COVERAGE_THRESHOLD`). `match_orchestrator.py` reached 94% and
`pipeline_builders.py` 100% after the coverage-focused test files above;
the remaining gap is concentrated in `src/shared/scene_classification.py`
(56% — `filter_false_cuts`'s moviepy path and `extract_scene_scores`'s live
ffmpeg loop are only exercised by the `integration`-marked tests) and other
pre-existing low-coverage files unrelated to this diff
(`src/decoder/attention_injection.py` 16%, `src/shared/geometry.py` 21%).
Judgment call: accepted at 83% (CI-passing) rather than chasing the local
85% bar on already-integration-tested branches, given three more phases
remain — flag if the stricter bar is required before merge.

**Discovered in passing (flagged via spawn_task, not fixed — out of
scope):** `tests/test_integration_main.py` has two tests calling
`run_pipeline()` with a stale keyword-argument signature (`TypeError`,
excluded from default runs by its own `integration`+`slow` markers, so CI
never caught it — pre-existing, confirmed via `git stash` to predate this
session). `pipeline_builders.build_execution_pool`'s `"tagged"` branch
passes kwargs that don't match the real `WorkerConfig`/
`TaggedMultiprocessPool` constructors (`# type: ignore[call-arg]` was
silencing a real `TypeError`, not a false positive) — `execution-pool:
tagged` has been completely non-functional; discovered because a new real
test for `pipeline_builders.py` actually reached that code path for the
first time.

### Phase 3 — Complexity tiers, realtime factor + anchor-encode cache — **3a done (2026-07-11), 3b deferred**

**3a (done):** Three standalone configs — `config/tier_fast.yaml` (yolo26n\*,
`genai-backend: null`, `ball-extractor: difference`), `config/tier_balanced.yaml`
(same nano-tier extraction — no yolo26s weight exists locally, see the
file's own comment — differentiated instead on the GenAI axis:
`canny-controlnet` at 10 steps / 384px), `config/tier_quality.yaml`
(yolo26x\*-class extraction matching `scripts/process_dataset.py`'s dataset-
curation defaults, `ball-extractor: segmentation`, `canny-controlnet` at 20
steps / 512px, full `evaluation-mode: [psnr, ssim, vmaf]`). All three set
`run-mode: full_match` and parse cleanly through `load_config`
(spot-checked: correct `detector`/`genai_backend`/`scene_chunk_duration_sec`
per tier).

**Realtime factor** (wall-clock ÷ duration, encoder/decoder separate) added
to both entry points: `src/main.py:run_pipeline`'s `timings_sec` gains
`encoder_realtime_factor`/`decoder_realtime_factor` (chunk duration from
`chunk.num_frames`/`chunk.fps`); `match_orchestrator.py`'s match summary
gains `timings_sec.encode_total`/`decode_total`/
`encoder_realtime_factor`/`decoder_realtime_factor` (encoder time counts
*both* attempted paths per point sub-chunk — semantic + fallback — since
outcome-safe routing genuinely pays for both; decoder time is the semantic
`DecoderRenderer.process()` cost only, since fallback-routed scenes never
run it).

**Anchor-encode cache** (`src/encoder/anchor_cache.py`): content-addressed
on `(video fingerprint, t_start, t_end, codec, crf, preset)` — fingerprint
is path+size+mtime (cheap, sufficient; full-file hashing would be correct
but expensive for multi-GB 4K matches we don't edit in place). Wired into
`match_orchestrator.py` via `_cached_fallback_encode`, keyed on the
*original* video and scene span (not the temp extracted clip, which gets a
fresh path every run) — so a second `encode_full_match` run on the same
video reuses every fallback encode instead of redoing them, exactly the
"anchors encoded once per video, ever" efficiency this report's
methodology section committed to. Defaults to
`outputs/_anchor_cache/<video_stem>/` (kept out of `assets/dataset`, which
is reserved for curated dataset content). Every sub-chunk/scene result now
carries a `cache_hit`/`fallback_cache_hit` flag for observability.

**Verification:** `tests/test_anchor_cache.py` (7 tests: key stability/
sensitivity to span/codec/video-content changes, miss-then-populate,
hit-skips-encode-fn, distinct spans don't collide) +
`tests/test_match_orchestrator_coverage.py` gained a same-video-twice test
proving the second `encode_full_match` run reports `cache_hit`/
`fallback_cache_hit: true` everywhere and produces byte-identical totals +
realtime-factor-field assertions on the existing routing test. Real
integration test (`assets/real_tennis.mp4`, same ~3.4s clip as Phase 2)
re-passed with all Phase 3 changes in place, 105s. Full `pytest -q` (273
passed, 1 skipped) and `ruff`/`mypy` clean. Coverage: 84% total
(`anchor_cache.py` 100%, `match_orchestrator.py` 95%, `pipeline_builders.py`
100%) — passes CI's 80%, one point short of the local 85% default; same
judgment call as Phase 2 (gap concentrated in pre-existing files this diff
didn't touch).

**3b (explicitly deferred, not started):** the variant-ladder spec
extending `scripts/benchmark_matrix.py`, the content-addressed intermediate
cache on the DAG orchestrator, and multi-variant GPU fan-out with
per-variant timing. Per this report's own goal-driven sequencing (§Goal
ladder), none of G1 (plumbing) or G2 (headline BD-rate) need this
machinery — only G3 (the trade-off curves) does. Deferred rather than
built hastily and undertested under tonight's time budget; pick this up
when G3 is actually being pursued, not before.

### Phase 4 — The headline experiments (goals G2→G4)

**G1 (plumbing validation) — done (2026-07-11), with a real bug found and
fixed along the way.** Scope note before G2: G2's full BD-rate headline run
needs the generative models retrained without the held-out videos first (a
genuine multi-hour-to-multi-day GPU training commitment) and G3/G4 need
the explicitly-deferred Phase 3b machinery (§Phase 3) — neither is
appropriate to start unsupervised overnight. G1 ("orchestrator runs
end-to-end on real data; frame-count invariant holds; accounting adds up;
no quality claims yet") was in scope and is what got validated tonight.

**Real run:** a 90 s excerpt of `assets/raw_4k/djokovic_federer.mp4`
(chosen for dense scene cutting — 632 scenes over the full 78 min match,
so even a short prefix samples real point/interlude diversity) through
`encode_full_match` at `config/tier_fast.yaml`.

**Bug found (real, in this report's own code, fixed same session):** the
first real run crashed with `ResidualCalculator received zero valid
frames` on the second point sub-chunk. Root cause:
`src/encoder/residual_calculator.py:_process_residuals` treats
`chunk.start_frame_id` as a literal seek offset into `chunk.source_uri`
(skips that many frames before reading), while `ActorExtractor` and every
other DAG node (`ball_extractor.py`, `segmentation_ball_extractor.py`,
`background_modeler.py`, `actor_components.py`, `synthesis_engine.py`,
`decoder_renderer.py`) only ever do `start_frame_id + local_idx`
arithmetic for frame_id numbering — never seek. `match_orchestrator.py`
had set `start_frame_id` to the match-global running frame counter, which
the residual calculator then tried to skip *into* each small,
self-contained extracted clip, running out of frames on every scene after
the first. **Fixed** by using `start_frame_id=0` for every extracted
sub-chunk (matching the convention every other consumer actually
implements); the match-global position is preserved as a new
`global_start_frame` field on each sub-chunk's result dict instead, purely
for bookkeeping. The underlying `ResidualCalculator`/`ActorExtractor`
inconsistency is real and separately dangerous (a shared multi-chunk
source file with `start_frame_id > 0` would misalign what actors were
detected on vs. what the residual is computed against) but is
**pre-existing and not fixed here** — flagged as its own follow-up task;
dormant until tonight because every existing real caller always used
`start_frame_id=0` and the test suite's nonzero cases only ever hit mocked
or synthetic-dummy paths. Regression test added:
`tests/test_match_orchestrator_coverage.py`'s routing test now asserts
every constructed `VideoChunk.start_frame_id == 0` and that
`global_start_frame` is monotonically non-decreasing across sub-chunks.

**Real evidence gathered (partial, by design — see below):** 11 real point
sub-chunks (`s0004c0000`–`s0004c0010`) completed successfully end to end —
real yolo26n detector/pose/segmenter actor extraction, real residual
computation producing genuine `metadata.msgpack` (40–87 KB)/`residual.mp4`
(3.4–12.6 MB)/`panorama.jpg` files, real `DecoderRenderer.process()`
output, real fallback-codec encodes, real outcome-safe size comparison —
plus 4 real fallback-only scenes (interlude/other/blank spans). Zero
crashes after the fix. **Notable real finding:** at `tier_fast`
(`genai-backend: null`, no generative compositing), the semantic payload
was *larger* than the fallback encode on **every single** point
sub-chunk observed (e.g. sub-chunk 0: metadata+residual = 12.7 MB vs.
fallback = 2.75 MB) — outcome-safe routing would pick fallback throughout.
This is expected, not a bug: with no generative reconstruction to shrink
the residual, the semantic path is close to the Whole-Frame Residual
Baseline (report 7 §1), which a well-tuned AV1 fallback beats easily on
real broadcast footage. It's a real, live demonstration of the routing
safety valve doing exactly its job (capping a non-paying tier's damage at
zero) rather than of `tier_fast`'s bitrate viability — that question is
G2/G3's, not G1's.

**Why partial, not the full 90 s:** each 2 s/100-frame real 4K sub-chunk
took ~15–20 minutes at `execution-pool: inline` (single-threaded, no
parallelism) — far slower than anticipated (CLAUDE.md's "~3 min for 3
frames" smoke-test timing doesn't extrapolate linearly to 100-frame real
chunks). The full excerpt would have taken several more hours for a single
validation pass; per the user's explicit call (asked mid-run), the 11
completed real sub-chunks plus 4 completed fallback scenes were accepted
as sufficient plumbing evidence rather than waiting out the full run.
**Implication for G2/G3/G4 planning:** `execution-pool: tagged` would
parallelize this substantially, but that mode is currently broken
(flagged separately) — fixing it is close to a prerequisite for G2's
full-match-scale runs being practical at all, not just a nice-to-have.

**Concurrency note:** this session ran alongside several spawned follow-up
sessions (each in what should be an isolated worktree per their own tool's
description) fixing the issues flagged above; despite that, one in-flight
edit to `match_orchestrator.py`/its test was clobbered mid-session and had
to be redone from scratch and committed immediately. Recorded here as a
process lesson, not a Phase 4 finding: don't leave fixes uncommitted for
extended periods when other sessions may be touching the same working
directory.

**G2/G3/G4 — not started tonight**, per the scope note above. On the
**held-out videos** (`alcaraz_highlights`, `djokovic_zverev`; methodology
decision 2): **G2** — full match at the balanced tier vs post-hoc AV1/HEVC
anchors on PointStream's own scene spans (decision 1), BD-rate via the
residual-CRF sweep with the metadata floor shown (decision 4), all-scenes
win/loss distribution + aggregate, both anchor forms (continuous +
segmented). **G3** — per-component variant ladders → composed tiers → the
speed/compression Pareto (x = realtime factor, y = bytes at matched VMAF;
anchor presets on the same plot) — needs Phase 3b (deferred). **G4** —
ablation verdicts read off the ladders' "off" rungs + the
routed-vs-all-semantic routing ablation. Run via the `pipeline-runner`
agent; these are multi-hour-to-multi-day jobs. **Prerequisite:** generative
models retrained without the held-out videos before any G2 quality claim.

### Phase 5 — Speed, real-time tier, and the gated training campaign (2026-07-11)

Locked in response to the per-stage FPS profile (see the 2026-07-11
findings entry); amended same day after absorbing
`reports/implementation_plan.md` (Gemini-session plan → 5.0 scope widened,
5.6 added). Seven workstreams (5.0–5.6), designed to be handed to
**separate agent sessions in isolated worktrees** (mandatory after the
G1-night clobbering incident; commit early and often). Dependency shape:
5.0 is a small serial gate; 5.1/5.3/5.4/5.6 then run in parallel; 5.2
follows 5.1 (overlapping config/orchestrator surfaces); 5.5 needs
5.1–5.3 merged. The session-to-agent table at the end of this section is
the reference for who tackles what.

**5.0 — Residual-calculator fixes (serial gate, small).** *(Scope widened
2026-07-11 after absorbing `reports/implementation_plan.md`, the
Gemini-session plan — all three items touch the same file, so they land
in one session to avoid a repeat of the clobbering incident.)*
(a) Finish making `VideoChunk.start_frame_id` numbering-only everywhere —
an uncommitted working-tree diff on `src/encoder/residual_calculator.py`
(removing the seek loop) already exists from the G1-night follow-ups and
must be finished or stashed before anything else touches that file. Flip
the regression tests to the new contract (seek behavior gone; numbering
arithmetic asserted). (b) Wire the dead `residual_block_threshold` config
knob (`src/shared/config.py:67` — read *nowhere* today) into
`ResidualCalculator` via `build_residual_calculator`
(`src/encoder/pipeline_builders.py:84`) and the orchestrator fallback
constructor; this re-lands a fix that provably existed uncommitted on
2026-07-10 night and was lost (see report 8's 2026-07-11 entry — the
"threshold 1.0 pays" claim is invalidated until 5.6 re-runs it). (c) Add
a `residual_pix_fmt` config key (default `"yuv444p"` for backward
compatibility) replacing the hardcoded value at
`src/encoder/residual_calculator.py:354`; the decoder needs no change
(`iter_video_frames_ffmpeg` normalizes any format back to bgr24). Run the
suite, commit.
*Session: quick interactive; no GPU.*

**5.1 — Execution & profiling.**
(a) Fix `execution-pool: tagged` (currently broken; prerequisite for any
full-match-scale run). (b) Add per-stage `fps_throughput` and full config
echo to `run_summary.json`. (c) Diagnose the GenAI decoder-vs-encoder
2.16× cost asymmetry **including a bit-identity check** of server/client
generated frames (symmetry tripwire). (d) Profile the FFmpeg paths (final
4K write at 0.53 fps; residual-encode threading/preset propagation).
(e) Per-scene panorama compute cache on the encoder (background is
near-static within a point; 0.46 fps stage today).
*Session: code + short pipeline-runner validation runs.*

**5.2 — Resolution & framerate ladders → `tier_realtime`.**
(a) Global processing-resolution config key (decode chunk frames at e.g.
960×540; keypoints/boxes are resolution-independent coordinates) with a
deterministic upsampler inside `SynthesisEngine` — full-res residual
becomes an optional layer priced by the Residual Guarantee. (b) Temporal
decimation key (process at N fps < native) with a deterministic frame
interpolator inside `SynthesisEngine`; actors already interpolate via
semantic events; ball extraction likely keeps full-rate input. (c)
`config/tier_realtime.yaml`: low res, decimated fps, yolo26n, `difference`
ball, cached panorama, `libx264 ultrafast`, no GenAI — completes
`tier_fast`'s own stated real-time intent.
*Session: after 5.1 merges (same surfaces); code + short validation runs.*

**5.3 — Background-layer ladder.**
Rungs: `panorama-static` (today) → `panorama+delta` (stateful panorama:
full send once, per-scene deltas for scoreboard/crowd; identical state
both sides keeps the guarantee) → `roi-video` (masked-actor low-bitrate
background video, `addroi` ROI boost for umpire/ball-kids/scoreboard;
needs libx264/x265 — libsvtav1 ignores ROI side data). Benchmark all
rungs via `benchmark_matrix`; the pays-for-itself table decides, not
architecture preference.
*Session: parallel with 5.1 (mostly disjoint encoder files).*

**5.4 — G2 training campaign (GPU long pole; start ASAP in parallel).**
(a) Protocol first: fixed probe set from **training-split videos only**;
checkpoint eval script scoring PSNR/SSIM/LPIPS on probes every N steps;
curves written to disk for cheap monitoring. (b) Launch variants —
ControlNet fine-tunes, Animate-Anyone, SPADE4Tennis — under successive
halving: evaluate at early checkpoints, prune losers, promote survivors.
(c) The surviving variants double as the GenAI speed-ladder rungs for G3
(SPADE fast / reduced-step ControlNet mid / Animate-Anyone quality).
Only after survivors stabilize: the G2 BD-rate headline run.
*Session: protocol design interactive, then hand training to
`pipeline-runner`; owns the GPU — 5.1/5.2 validation runs coordinate
around it.*

**5.5 — Phase 3b harness, promoted (this is G3).**
With the knobs from 5.1–5.3 merged, implement the variant-ladder sweep
harness + DAG intermediate cache + GPU fan-out (§Experiment-efficiency
architecture) and run the tier × component ladder on one held-out video:
the per-component **quality / time (FPS) / bitrate** attribution table and
the speed/compression Pareto figure fall out of the same run.
*Session: last; pipeline-runner for the sweep.*

**5.6 — Residual-compression matrix (added 2026-07-11, from the Gemini
plan).** Once 5.0(b)/(c) land: new spec
`config/benchmarks/ablation_residual_compression.yaml` sweeping
`residual-block-threshold` [0.0, 1.0] × `residual-pix-fmt`
[yuv444p, yuv420p, gray] full-length on `assets/real_tennis.mp4` with
`evaluation-mode: [psnr, ssim, vmaf]`. Answers three questions at once:
does 4:2:0 chroma subsampling beat the current hardcoded 4:4:4; does
luma-only add enough over 4:2:0 to justify dropping color correction
entirely; does the block threshold stack with subsampling (the old
thresh-3.0 observation — aggressive zeroing *adding* 14.8 KB by creating
artificial block edges — wants confirming at 1.0). This **replaces** the
invalidated dynamic-thresholding ablation (report 8, 2026-07-11 entry);
run with current pre-retrain checkpoints so results stay comparable with
report 8's racket/panorama entries. ~6 variants × ~28 min ≈ 3 h GPU —
schedule before 5.4's training occupies the GPU, or interleave.
*Session: pipeline-runner, spec written by whichever session closes 5.0.*

#### Session-to-agent assignment (the parallel split, for reference)

| # | Workstream | Agent / session | Isolation | Depends on | GPU |
|---|---|---|---|---|---|
| S1 | 5.0 residual-calculator fixes (a+b+c) | interactive main session (small, review-worthy diff) | main checkout — it owns the dirty file | — | no |
| S2 | 5.6 residual-compression matrix | `pipeline-runner` | none (run-only, writes `outputs/benchmarks/`) | S1 merged | yes (~3 h) |
| S3 | 5.1 execution & profiling (tagged pool, FPS metrics, 2 diagnoses, panorama compute cache) | `general-purpose` in an **isolated worktree**; short validation runs via `pipeline-runner` | worktree | S1 | brief |
| S4 | 5.3 background-layer ladder | `general-purpose` in an **isolated worktree** | worktree | S1 | brief |
| S5 | 5.4 training protocol, then variant training | protocol: interactive; training: `pipeline-runner` | worktree for protocol code | S1 (protocol); GPU free (training) | yes (multi-day, owns GPU once started) |
| S6 | 5.2 resolution/framerate knobs + `tier_realtime` | `general-purpose` in an isolated worktree | worktree | S3 merged (shared surfaces) | brief |
| S7 | 5.5 Phase 3b harness + G3 ladder run | `general-purpose` + `pipeline-runner` for sweeps | worktree | S3, S4, S6 merged | yes |

GPU scheduling rule: S2's 3 h matrix and S3/S4/S6's short validation runs
go **before** S5's multi-day training starts; once training owns the GPU,
code-only work continues but real-run verification queues behind it.
Process rules (from the G1-night clobbering): every parallel session works
in an isolated worktree, commits early and often, and no two sessions may
touch `src/encoder/residual_calculator.py` concurrently.

### Explicitly out of scope / decisions needed
- **Multi-ControlNet**: stays deferred (report 7 §4) unless the Phase 4
  results make single-condition ControlNet the bottleneck — revisit then.
- **GenAI engine selection** (report 7 §2A) is orthogonal but Phase 4's
  quality tier needs the winner; run Phase 4 first with the best current
  engine and re-run the quality tier if the selection changes.

## Open questions & next steps

1. ~~Phase 1: shared scene-classification module + runtime port~~ **done
   (2026-07-11)**.
2. ~~Phase 2: full-match orchestrator + match-level summary + outcome-safe
   routing + frame-count-invariant test~~ **done (2026-07-11)**.
3. Phase 3a (tier configs, realtime factor, anchor-encode cache) **done
   (2026-07-11)**. Phase 3b (variant-ladder harness, DAG intermediate
   cache, GPU fan-out) **still open** — deferred until G3 is pursued.
4. ~~Phase 4 G1: real end-to-end validation~~ **done (2026-07-11)** — real
   run on a `djokovic_federer.mp4` excerpt, found and fixed a real
   `start_frame_id` bug along the way (see Phase 4 findings). G2–G4 remain
   **open**: G2 needs the generative models retrained without
   `alcaraz_highlights` + `djokovic_zverev` first (a real multi-hour+ GPU
   commitment, not started); G3/G4 need Phase 3b.
5. Decide whether the dataset itself is released/described as a standalone
   contribution in the paper (affects report 7 §3.3 wording).
6. ~~Should interlude routing also get a Residual-Guarantee-style
   ablation?~~ **Resolved 2026-07-11:** yes — the routed-vs-all-semantic
   ablation is part of G4, and routing itself is made outcome-safe
   (methodology decision 5).
7. Add LPIPS + FVD (and the VMAF 4K model) to the evaluation stack in
   `src/experiment_evaluation.py` (open; needed by G2).
8. **New (2026-07-11):** fix `execution-pool: tagged` (currently broken,
   see the flagged follow-up task) before attempting G2/G3 at real
   full-match scale — `inline` execution measured ~15-20 min per real
   2s/100-frame 4K sub-chunk during the G1 run, which does not scale to
   full matches (or even the held-out videos in full) without
   parallelism. *Now Phase 5.1(a).*
9. **New (2026-07-11):** fix the `ResidualCalculator`/`ActorExtractor`
   `start_frame_id` contract inconsistency found during G1 (flagged
   separately) — dormant today only because every real caller uses
   `start_frame_id=0`, but a real correctness hazard for any future
   shared-source-file, multi-chunk usage pattern. *Now Phase 5.0 (the
   serial gate); a partial uncommitted diff already sits in the working
   tree and must be finished or stashed first.*
10. **New (2026-07-11):** execute Phase 5 (per-stage FPS profiling →
    real-time tier → background-layer ladder → gated G2 training campaign
    → promoted Phase 3b harness → residual-compression matrix); see
    §Phase 5 for the seven workstreams and the session-to-agent
    assignment table. Immediate open diagnoses inside it: the GenAI
    decoder/encoder 2.16× cost asymmetry (+ bit-identity symmetry check)
    and the 0.53 fps decoder-side FFmpeg write.
