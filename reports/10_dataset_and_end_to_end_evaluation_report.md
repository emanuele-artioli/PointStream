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

**The vision above was NOT fully documented anywhere before 2026-07-10, and
its runtime half is NOT implemented.** Gap analysis:

| Piece | Documented? | Implemented? |
|---|---|---|
| Dataset curation pipeline mechanics | ✅ Reports 2 (scene classification), 3 (pipeline unification), 4 (universal RGBA+DWPose format), 5 (canny/captions) | ✅ `scripts/process_dataset.py` (stages: classify, segment, pose, skeleton, canny, caption) |
| Dataset **as a catalogued contribution** (inventory, stats, curation methodology, coverage caveats) | ❌ → now §Dataset inventory below | ✅ data exists on disk |
| GenAI finetuning on the dataset | ✅ Reports 1, 4, 5 | ✅ checkpoints in `assets/weights/` (pose/seg/ip-adapter/custom controlnets, `spade4tennis_lite_generator.pt`, pix2pix); Multi-ControlNet **deferred** (report 7 §4) |
| **Runtime** scene classification + interlude/point routing | ⚠️ named as an architecture rule (CLAUDE.md "scene-classification routing stays modular") and as paper section 3.1 in report 7 — but no plan for the runtime port | ❌ **zero scene/interlude code under `src/`**; classifier lives only in `scripts/process_dataset.py:classify_scenes` |
| Full-match orchestrator (scene split → route → per-scene encode → aggregate accounting) | ❌ | ❌ `src/main.py` processes one chunk of one clip |
| Speed/compression trade-off across model-complexity tiers | ❌ (fast-vs-slow tiering exists only implicitly: runtime defaults yolo26n\*, curation defaults yolo26x\*) | ⚠️ per-stage timings exist (`PipelineProfiler`, `timings_sec` in `run_summary.json`) but no realtime-factor metric, no tier configs |

The implementation plan to close these gaps is in §Implementation plan.
**Methodology locked 2026-07-11** (see that findings entry): held-out test
split = `alcaraz_highlights` + `djokovic_zverev` (no cross-validation);
anchors encoded post-hoc on PointStream's own scene spans; scene cuts
deterministic and shared across all tiers; ablations unified with speed
sweeps as per-component variant ladders with "off" as the leftmost rung;
LPIPS + FVD added alongside PSNR/SSIM/VMAF(4K model); routing made
outcome-safe via the server-side residual check.

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

### Phase 3 — Complexity tiers, sweep harness + speed reporting
Three configs in `config/` (e.g. `tier_fast.yaml` = yolo26n\* +
`genai-backend: null` or lightest engine; `tier_balanced.yaml`;
`tier_quality.yaml` = yolo26x\*-class backends + chosen GenAI engine), and a
**realtime factor** (encode wall-clock ÷ source duration; encoder and
decoder reported separately) computed in the evaluation block of every
`run_summary.json`. Timings plumbing already exists
(`src/shared/profiling.py`). Also in this phase: the
§Experiment-efficiency machinery — variant-ladder specs (extending
`scripts/benchmark_matrix.py`), the content-addressed intermediate cache on
the DAG, multi-variant GPU fan-out with per-variant timing, and the
anchor-encode cache.

### Phase 4 — The headline experiments (goals G2→G4)
On the **held-out videos** (`alcaraz_highlights`, `djokovic_zverev`;
methodology decision 2): **G2** — full match at the balanced tier vs
post-hoc AV1/HEVC anchors on PointStream's own scene spans (decision 1),
BD-rate via the residual-CRF sweep with the metadata floor shown (decision
4), all-scenes win/loss distribution + aggregate, both anchor forms
(continuous + segmented). **G3** — per-component variant ladders → composed
tiers → the speed/compression Pareto (x = realtime factor, y = bytes at
matched VMAF; anchor presets on the same plot). **G4** — ablation verdicts
read off the ladders' "off" rungs + the routed-vs-all-semantic routing
ablation. Run via the `pipeline-runner` agent; these are multi-hour jobs.
**Prerequisite:** generative models retrained without the held-out videos
before any G2 quality claim.

### Explicitly out of scope / decisions needed
- **Multi-ControlNet**: stays deferred (report 7 §4) unless the Phase 4
  results make single-condition ControlNet the bottleneck — revisit then.
- **GenAI engine selection** (report 7 §2A) is orthogonal but Phase 4's
  quality tier needs the winner; run Phase 4 first with the best current
  engine and re-run the quality tier if the selection changes.

## Open questions & next steps

1. Phase 1: shared scene-classification module + runtime port (open).
2. Phase 2: full-match orchestrator + match-level summary + outcome-safe
   routing + frame-count-invariant test (open).
3. Phase 3: tier configs, realtime factor, variant-ladder harness,
   intermediate/anchor caches, GPU fan-out (open).
4. Phase 4 / G2–G4: headline BD-rate on held-out videos, Pareto figure,
   ladder ablations (open; depends 1–3 **and** on retraining the
   generative models without `alcaraz_highlights` + `djokovic_zverev`).
5. Decide whether the dataset itself is released/described as a standalone
   contribution in the paper (affects report 7 §3.3 wording).
6. ~~Should interlude routing also get a Residual-Guarantee-style
   ablation?~~ **Resolved 2026-07-11:** yes — the routed-vs-all-semantic
   ablation is part of G4, and routing itself is made outcome-safe
   (methodology decision 5).
7. Add LPIPS + FVD (and the VMAF 4K model) to the evaluation stack in
   `src/experiment_evaluation.py` (open; needed by G2).
