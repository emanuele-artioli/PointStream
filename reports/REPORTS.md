# POINTSTREAM Reports — Index & Dashboard

*This file is the source of truth for the experimental effort. Update it (via
`/update-reports`) whenever a report changes or a workstream moves.*

*Last updated: 2026-07-10*

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
| Codec baselines (AV1, HNeRV/DCVC) | ⬜ Open | Reviewer-critical ([6](6_action_matrix.md)) |
| Residual-Guarantee benchmark harness | ✅ Working | `scripts/benchmark_matrix.py`; first run exposed a panorama symmetry violation, now fixed ([8](8_residual_guarantee_benchmarks_report.md)) |
| Detector selection (SAM3 vs YOLOv26 vs RF-DETR) | ⬜ Open | [7](7_implementation_plan.md) §2C |
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
3. AV1 baseline benchmark on the tennis dataset; one learned codec
   (HNeRV or DCVC) (R2, R5).
4. Component ablations under the Residual-Guarantee framework: racket
   heuristics and dynamic thresholding vs residual payload size, and the
   panorama-quality trade-off itself.
   *Tooling ready (2026-07-10), gate cleared (2026-07-10):*
   `scripts/benchmark_matrix.py` runs a baseline-vs-variants matrix from a
   spec in `config/benchmarks/` and emits the pays-for-itself table.
   *Racket heuristics ablation:* ~~still owed~~ **done (2026-07-10)** (convex hull tracking drastically outperforms naive bboxes).
   *Panorama quality trade-off:* ~~still owed~~ **done (2026-07-10)** (higher qualities do not pay, as metadata cost exceeds residual savings).
   The dynamic thresholding ablation remains owed as a full-length (`num-frames: null`) swept matrix.
5. Background panorama stitching for moderate camera motion.
6. Detector/segmenter selection: SAM3 vs YOLOv26 vs RF-DETR (R3).
7. Deferred (post-core): second domain, MOS study, demo video, VVC,
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

## Standard report format (new reports)

Name new reports `N_<topic>_report.md`, continuing the numbering (next: 9).
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
