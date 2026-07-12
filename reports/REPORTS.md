# POINTSTREAM Reports — Index & Dashboard

*This file is the source of truth for the experimental effort. Update it (via
`/update-reports`) whenever a report changes or a workstream moves.*

*Last updated: 2026-07-11*

## The goal: the Residual Guarantee

```
size(metadata) + size(residual)  <  size(full-frame encoding at equal quality)
```

Server and client run the identical deterministic `SynthesisEngine`; the
transmitted residual makes reconstruction exact. A component (racket
tracking, panorama warping, GenAI compositing, …) is justified **iff** it
shrinks the residual payload by more than the metadata it adds — measured
against the Whole-Frame Residual Baseline (panorama + generated actors +
whole-frame residual). With every component disabled, the residual is the
whole video; each enabled component must pay for itself. Full framing:
[7_implementation_plan.md](7_implementation_plan.md) §1.

## Workstream status

| Workstream | Status | Evidence / where recorded |
|---|---|---|
| Scene classification (cuts vs pans, interlude routing) | ✅ Working | HSV-histogram invariance + point-anchored thresholding; CLIP/VLM zero-shot rejected ([2](2_scene_classification_research.md)) |
| Racket tracking (convex hull + wrist anchoring) | ✅ Working | Pixel-accurate extreme points, majority-hand voting ([2](2_scene_classification_research.md) Phase 6); ablation vs naive bboxes completed and pays for itself (+951KB net savings) ([8](8_residual_guarantee_benchmarks_report.md)) |
| Shared-library architecture (`src/shared/`) | ✅ Done | Refactor complete, absolute imports enforced ([3](3_architecture_refactor_research.md)) |
| SPADE4Tennis generative engine | ⚠️ In progress | Design + loss stack decided ([1](1_spade4tennis_plan.md)); final arch choice vs diffusion still open ([7](7_implementation_plan.md) §2A) |
| Animate-Anyone integration | ⚠️ Evaluated, not final | Fork reads RGBA PNG + DWPose dataset format ([4](4_animate_anyone_integration_research.md)) |
| ControlNet temporal consistency | ⚠️ Mechanisms built | Optical-flow warping, adaptive denoising, keyframe resets, cross-frame attention ([5](5_genai_temporal_consistency_research.md)) |
| Background panorama stitching (camera motion) | ⬜ Open | [7](7_implementation_plan.md) §2B |
| Codec baselines (AV1, HNeRV/DCVC) | ⚠️ In progress | Reviewer-critical ([6](6_action_matrix.md)); full-length AV1/HEVC anchor curve done (AV1 beats HEVC by 5-32% at matched VMAF, `slow`/`veryslow`). **HNeRV learned-codec baseline done (2026-07-11):** real 15,000-epoch training run on `real_tennis.mp4` — HNeRV loses to every AV1 point and most HEVC points on bytes *and* quality simultaneously, an honest negative result that still closes the R2/R5 gap. Still need to overlay a real PointStream run against the AV1/HEVC curve ([9](9_codec_baselines_report.md)) |
| Residual-Guarantee benchmark harness | ✅ Working | `scripts/benchmark_matrix.py`; first run exposed a panorama symmetry violation, now fixed ([8](8_residual_guarantee_benchmarks_report.md)) |
| Detector selection (SAM3 vs YOLOv26 vs RF-DETR) | ⬜ Open | [7](7_implementation_plan.md) §2C |
| Dataset curation (raw_4k → assets/dataset) | ✅ Built, now catalogued | 7 videos (≈2h37m 4K) / 952 scenes / 399 points / 114 deep-annotated tracks; quality-tier models (yolo26x), manually supervised ([10](10_dataset_and_end_to_end_evaluation_report.md)) |
| End-to-end full-match evaluation (runtime scene routing, complexity tiers, speed/compression Pareto) | ⚠️ In progress | Phases 1–3a + Phase 4 G1 done (2026-07-11): real `encode_full_match` run against a real `djokovic_federer.mp4` excerpt found and fixed a `start_frame_id` bug, then completed 11 real point sub-chunks + 4 fallback scenes with zero crashes; 3b + G2-G4 open. Phase 5 locked (2026-07-11): FPS profile shows every stage 5–150× off real time at 4K → six-workstream speed/real-time + gated-training campaign with a parallel-agent split ([10](10_dataset_and_end_to_end_evaluation_report.md)) |
| Perceptual quality metrics (PSNR/SSIM/VMAF + LPIPS/FVD) | ⚠️ In progress | PSNR/SSIM/VMAF wired in `src/experiment_evaluation.py`. **FVD implemented 2026-07-11** (`src/shared/fvd.py`, I3D R50/Kinetics-400 via `pytorchvideo`, real-data verified: `FVD(self,self)=0.0`, `FVD(self,degraded)=7.20` on `assets/real_tennis.mp4`). **LPIPS still unimplemented** — no `lpips` code anywhere in `src/`; a prior "already backfilled" claim was checked and found false ([10](10_dataset_and_end_to_end_evaluation_report.md) 2026-07-11 FVD entry) |
| TOMM resubmission | ⚠️ In progress | Action matrix tracks all 8 reviewer themes ([6](6_action_matrix.md)) |

## Prioritized next steps

Seeded from [6_action_matrix.md](6_action_matrix.md) and
[7_implementation_plan.md](7_implementation_plan.md); strike items through
(`~~…~~ **done (date)**`) rather than deleting them.

1. ~~**Fix the suspected panorama symmetry violation**~~ **done (2026-07-10)**
   — server residuals were computed against the raw panorama while the
   client decoded the JPEG. Fixed by round-tripping the panorama through
   the configured codec inside the encoder pipeline, before residual
   computation; re-verified with a real benchmark run (q50 vs q90
   `residual.mp4` hashes now differ)
   ([8](8_residual_guarantee_benchmarks_report.md), 2026-07-10 entry).
2. Finalize the generative architecture: complete ControlNet + Animate-Anyone
   evaluation, compare against SPADE, pick one for the paper (R1, R5).
3. ~~AV1 baseline benchmark on the tennis dataset; one learned codec
   (HNeRV or DCVC) (R2, R5).~~
   *AV1/HEVC anchor curve done (2026-07-10):* `scripts/codec_baseline_sweep.py`
   encodes the raw source directly with AV1/HEVC across a CRF ladder using
   the pipeline's own FFmpeg wrapper, now supports sweeping multiple presets
   in one run, and can report side-by-side against a PointStream
   `run_summary.json`. Hit and fixed a preset-parity bug along the way
   (`fast` on both codecs made HEVC dominate AV1 — a preset-name mismatch,
   not a real reversal). Full-length run (all 60 frames, `slow` +
   `veryslow`) confirms AV1's expected edge once compared at matched VMAF:
   5-32% fewer bytes than HEVC, widening at lower bitrates
   ([9](9_codec_baselines_report.md)).
   *Learned-codec baseline done (2026-07-11):* from-scratch HNeRV
   (`src/shared/hnerv_arch.py`, `scripts/hnerv_baseline.py`), trained for
   real on `real_tennis.mp4` (15,000 epochs, ~55 min GPU). Honest result:
   HNeRV (5.94 MB, PSNR 32.5/VMAF 79.1) is dominated on bytes *and* quality
   by every AV1 point and most HEVC points in the anchor curve above — the
   R2/R5 gap needed a comparison to exist, not to win
   ([9](9_codec_baselines_report.md)).
   Still owed: run PointStream itself at a matching preset and overlay via
   `--pointstream-run` to get the actual paper claim.
4. Component ablations under the Residual-Guarantee framework: racket
   heuristics and dynamic thresholding vs residual payload size, and the
   panorama-quality trade-off itself.
   *Tooling ready (2026-07-10), gate cleared (2026-07-10):*
   `scripts/benchmark_matrix.py` runs a baseline-vs-variants matrix from a
   spec in `config/benchmarks/` and emits the pays-for-itself table.
   *Racket heuristics ablation:* ~~still owed~~ **done (2026-07-10)** (convex hull tracking drastically outperforms naive bboxes).
   *Panorama quality trade-off:* ~~still owed~~ **done (2026-07-10)** (higher qualities do not pay, as metadata cost exceeds residual savings).
   *Dynamic thresholding:* ~~still owed~~ ~~**done (2026-07-10)** (threshold 1.0 optimally gates noise and saves bitrate)~~ **Superseded 2026-07-11 — claim invalid:** the config knob was never wired into `ResidualCalculator`; the matrix that varied ran uncommitted, since-lost code ([8](8_residual_guarantee_benchmarks_report.md) 2026-07-11 entry). **Re-run done (2026-07-11)** as the combined threshold × pixel-format matrix under `libx264`: `gray` (luma-only) at threshold 0.0 wins on both bytes and VMAF; thresholding alone barely helps and can cost bytes once chroma is already reduced ([8](8_residual_guarantee_benchmarks_report.md) "Residual-compression matrix run" entry). Latent pending Phase 5.2's real-time codec choice — the project's default libsvtav1 forces yuv420p regardless of this knob.
5. Background panorama stitching for moderate camera motion.
6. Detector/segmenter selection: SAM3 vs YOLOv26 vs RF-DETR (R3).
7. **End-to-end full-match evaluation** — the headline experiment
   ([10](10_dataset_and_end_to_end_evaluation_report.md)):
   ~~Phase 1 port scene classification into `src/`~~ **done (2026-07-11)**
   (shared module, deterministic cuts shared across tiers,
   `tests/test_scene_classification.py`, verified byte-faithful against
   real cached `alcaraz_ruud` data).
   ~~Phase 2 full-match orchestrator~~ **done (2026-07-11)**
   (`src/encoder/match_orchestrator.py`: interludes → baseline codec,
   points → semantic pipeline in sub-chunks, outcome-safe
   semantic-vs-fallback routing per sub-chunk, match-level summary dict;
   model builders extracted to `src/encoder/pipeline_builders.py` so
   weights load once per match, not per chunk; real run passed on a
   ~3.4 s `assets/real_tennis.mp4` clip, 101 s).
   ~~Phase 3a tier configs + realtime factor + anchor cache~~ **done
   (2026-07-11)** (`config/tier_{fast,balanced,quality}.yaml`;
   `src/encoder/anchor_cache.py` — content-addressed on video+span+codec,
   second run on the same video reuses every fallback encode). *Phase 3b
   (variant-ladder harness, DAG intermediate cache, GPU fan-out) explicitly
   deferred* — only needed for G3, not G1/G2, per this report's own
   goal-driven sequencing.
   ~~Phase 4 G1: real end-to-end validation~~ **done (2026-07-11)** — a
   real `encode_full_match` run on a 90s real `djokovic_federer.mp4`
   excerpt at `tier_fast` found and fixed a real bug
   (`VideoChunk.start_frame_id` was set to the match-global frame counter,
   but `ResidualCalculator` seeks that many frames into `source_uri` while
   every other DAG node only does numbering arithmetic — fixed by using
   `start_frame_id=0` per extracted sub-chunk); after the fix, 11 real
   point sub-chunks + 4 real fallback scenes completed with zero crashes
   (stopped there by design — full-90s completion would have taken several
   more hours at `execution-pool: inline`, ~15-20 min per real 2s/100-frame
   4K sub-chunk). Notable real finding: at `tier_fast` (no GenAI), the
   semantic payload was larger than the fallback encode on every observed
   sub-chunk — outcome-safe routing picking fallback throughout, exactly
   as designed, not a bug. G2-G4 headline BD-rate vs post-hoc AV1/HEVC
   anchors on the held-out videos (`alcaraz_highlights`,
   `djokovic_zverev`) + speed/compression Pareto remain open. *Methodology
   locked (2026-07-11)* — held-out split, post-hoc anchor protocol,
   ablations unified with speed sweeps as variant ladders, LPIPS/FVD added
   ([10](10_dataset_and_end_to_end_evaluation_report.md) findings
   2026-07-11). **Gate:** generative models must be retrained without the
   held-out videos before any G2 quality claim; `execution-pool: tagged`
   (currently broken, flagged separately) likely needs fixing first too,
   given the `inline`-mode timing observed during G1.
   **Phase 5 locked (2026-07-11)** — the execution path for all of the
   above ([10](10_dataset_and_end_to_end_evaluation_report.md) §Phase 5):
   per-stage FPS profiling showed the GenAI pipeline at ~0.09 fps encode /
   ~0.06 fps decode at 4K (every stage 5–150× off real time), so speed is
   attacked on three axes at once — per-stage efficiency, spatial/temporal
   down-processing as Residual-Guarantee layers (LCEVC-style), and
   concurrency (tagged pool). **5.0 fully closed same night**: the
   `start_frame_id` contract (commit `f25fc7b`) and the dead
   `residual_block_threshold`/new `residual_pix_fmt` wiring (commit
   `3dea9ce`) are both done — **5.6 is now unblocked**. **5.1(a) also
   closed**: `execution-pool: tagged` was calling
   `TaggedMultiprocessPool`/`WorkerConfig` with kwargs that didn't match
   their real constructors (silenced with `# type: ignore`, never fixed)
   — fixed in commit `768cb5d`. **Bonus find while sweeping worktrees for
   unmerged work before spawning new sessions:** a complete, tested HNeRV
   learned-codec baseline (report 9) was sitting unmerged in a spawned
   worktree — cherry-picked onto main as commit `7047207`, CI green. One
   worktree (`suspicious-blackwell-724748`) holds a substantial
   in-progress `RigidObjectPacket` dead-code removal, left untouched
   pending a human decision (report 10 §Phase 5 session-table note).
   Remaining workstreams, in dependency order: **5.1(b–e)** FPS metrics +
   config echo, GenAI decode-vs-encode 2.16× cost-asymmetry diagnosis (+
   bit-identity symmetry check), 0.53 fps FFmpeg-write diagnosis,
   per-scene panorama cache → parallel with **5.3** background-layer
   ladder (panorama-static / panorama+delta / ROI background video),
   **5.4** gated G2 training campaign (train-split probe set, successive
   halving across ControlNet/Animate-Anyone/SPADE — survivors double as
   G3's GenAI speed ladder — protocol/harness only, not the real training
   run). **5.6 done (2026-07-11):** residual-compression matrix ran under
   `libx264` — `gray` (luma-only) at threshold 0.0 wins on bytes and VMAF,
   thresholding alone barely helps
   ([8](8_residual_guarantee_benchmarks_report.md) 2026-07-11 entry) →
   **5.2** resolution/framerate knobs + `tier_realtime` → **5.5**
   promoted Phase 3b harness = G3's per-component quality/FPS/bitrate
   table. Session-to-agent assignments are written down in report 10
   §Phase 5 (session table). Parallel sessions must use isolated
   worktrees (G1-night clobbering lesson).
8. Deferred (post-core): second domain, MOS study, demo video, VVC,
   Multi-ControlNet — see [7](7_implementation_plan.md) §4.

## Reports catalog

Reports 1–7 predate this index (Antigravity era) and are kept as-is; new
reports follow the numbering and the standard format below.

| Report | Type | Scope |
|---|---|---|
| [1_spade4tennis_plan.md](1_spade4tennis_plan.md) | Plan | SPADE-conditioned GAN to replace the blurry Pix2Pix baseline: loss stack, multi-scale discriminator, racket synthesis, Lite→Full tiers |
| [2_scene_classification_research.md](2_scene_classification_research.md) | Research | Scene extraction/classification evolution (HSV invariance, point-anchored thresholding, VLM failure) + pixel-perfect racket skeleton tracking |
| [3_architecture_refactor_research.md](3_architecture_refactor_research.md) | Research | Shared-library refactor: `src/shared/` consolidation, dataset pipeline, compat patches, absolute imports |
| [4_animate_anyone_integration_research.md](4_animate_anyone_integration_research.md) | Research | Universal dataset format (RGBA PNG + DWPose) and the Moore-AnimateAnyone fork adaptation |
| [5_genai_temporal_consistency_research.md](5_genai_temporal_consistency_research.md) | Research | ControlNet conditioning + temporal consistency: flow warping, adaptive denoising, keyframe resets, cross-frame attention |
| [6_action_matrix.md](6_action_matrix.md) | Matrix | ACM MM reviews → TOMM resubmission: reviewer themes, status, execution checklist |
| [7_implementation_plan.md](7_implementation_plan.md) | Plan | Comprehensive source of truth: Residual-Guarantee paradigm, engineering tasks, paper structure |
| [8_residual_guarantee_benchmarks_report.md](8_residual_guarantee_benchmarks_report.md) | Report | Benchmark harness (`scripts/benchmark_matrix.py`) + ablation findings: panorama symmetry violation (fixed 2026-07-10), null-PSNR evaluation bug (fixed 2026-07-10) |
| [9_codec_baselines_report.md](9_codec_baselines_report.md) | Report | Conventional-codec anchor sweep (`scripts/codec_baseline_sweep.py`): AV1/HEVC direct encode of the source video, no semantics, for the reviewer-requested baseline comparison |
| [10_dataset_and_end_to_end_evaluation_report.md](10_dataset_and_end_to_end_evaluation_report.md) | Report | The end-to-end story: raw_4k/dataset inventory (dataset-as-contribution), trained-checkpoint census, gap analysis (runtime scene routing missing from `src/`), and the 4-phase full-match evaluation plan (routing, tiers, speed/compression Pareto) |

## Standard report format (new reports)

Name new reports `N_<topic>_report.md`, continuing the numbering (next: 11).
Every report follows:

```markdown
# <Topic> — Report
*Status: Active | Last updated: YYYY-MM-DD | Code: src/<...>*

## Scope
One paragraph: what part of the system this report owns.

## Current state (TL;DR)
The up-to-date conclusions someone should read before touching this area.

## Findings log
### YYYY-MM-DD — <short title>
**Problem/Question:** what was observed or asked.
**Diagnosis/Evidence:** how it was isolated; the actual numbers.
**Resolution:** what changed (code, config, methodology), or "open".
**Paper impact:** which claim/section/figure this feeds.

## Open questions & next steps
```

**Rules of evidence:**
- Every dated entry cites real numbers and paths — an `outputs/<timestamp>/`
  dir, a `run_summary.json` size/timing, a test name. No numbers, no claim.
- Superseded findings stay in the log with a
  `**Superseded YYYY-MM-DD:** see <entry>` pointer — never rewrite history.
- Invalidated output runs are moved aside (e.g.
  `outputs/_superseded/<timestamp>_<reason>/` via `mv`), never deleted.
