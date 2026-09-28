# 6. Evaluation protocol and current results

This is the prospective protocol for the redesigned research campaign. It
supersedes stale instructions to select a paper claim solely on 0.7 FG + 0.3 BG
PSNR or to skip confirmation. Historical outcomes retain their original
metrics and eligibility. No old result is relabeled a win by changing its metric.
Execution proceeds through [experiment cards](07-experiment-plan.md), not from
this chapter alone. The goal is to discover whether an advantage exists; a
negative or inconclusive outcome is an admissible result.

## Research questions and claim boundaries

| Question | Main comparison / artifact | Permissible conclusion |
|---|---|---|
| RQ1: Does semantic decomposition improve the rate–quality tradeoff on tennis? | Same-input, fully decoded PointStream and conventional/neural/generative ladders | Advantage only over named methods, metrics, domains and overlapping ranges |
| RQ2: Does the design preserve the event's geometry and motion? | Independent court-line, ball and racket labels plus player fidelity; equal-rate pairs | Task fidelity, not general pixel-losslessness or universal perceptual quality |
| RQ3: Which components cause any improvement? | Paired ablations with total bytes, same reference/adaptation policy | Mechanism-specific evidence; no general model-family ranking from one checkpoint |
| RQ4: Does cached appearance/background amortize? | Actual consecutive points including refreshes, entrants, cuts and baseline references | Measured duration/access-specific break-even, not repeated-cost extrapolation |
| RQ5: What are the practical limits? | End-to-end time, memory, startup, failure/fallback incidence and domain strata | Named-hardware deployment envelope; offline preprocessing cannot prove live streaming |

## Source and split protocol

Tennis is primary. Use development strata spanning still/PTZ motion, small and
large players, grass/clay/hard courts where available, shadows, fast swings,
occlusion and scene transitions. Existing Federer007, Alcaraz000 and
Perricard002 measurements are exposed development material. The seven tennis
videos used for earlier AnimateAnyone fine-tuning are not unseen tests.

Reuse the checked-in source manifests and reservations; inspect their actual
external files and timestamps before freezing a run. The September reservation
contains three acquired 1080p matches, not certified native 4K confirmation:
[source-count policy](../../manifests/evaluation_20260914_source_count_policy.json),
[reservation](../../manifests/evaluation_20260916_coordinator_confirmation_reservation.json),
[timestamp audit](../../manifests/evaluation_20260915_e03a_confirmation_timestamps.json).
Reservation is not proof of independence or unexposed status. Audit source IDs,
event overlap, training/extraction logs, PTS and exact frame hashes. If a source
was inspected for quality or selection, record exposure and replace it or narrow
the generalization claim. Do not silently reduce required source count.

Freeze development/validation/confirmation membership at match level, with
model-fitting and reference selection policies explicit. Camera calibration or
appearance adaptation from evaluation frames counts as test-time adaptation;
record lookahead, time and transmitted updates. Select configurations on
development, freeze them before confirmation, then report every selected
confirmation source including failures. Bootstrap uncertainty at match level,
not by treating correlated frames as independent samples. With only three
matches, state the limited precision and population scope.

An egocentric pilot can test domain boundaries after a tennis configuration is
qualified. It is not a replacement for tennis or confirmation of a planar-court
mechanism. A deployment claim for that domain requires its own split and controls.

## Rate and receiver contract

For a clip with N frames, dimensions W×H and measured duration τ:

\[
T=B+F+M+R+H_{env},\qquad r=8T/\tau,\qquad bpp=8T/(NWH).
\]

The categories are disjoint: background B; appearance/foreground F (including
sequence-specific weights); motion/masks/geometry M; residual R; envelope,
indexing, container, refresh/fallback overhead H_env. Reconcile the sum to
actual persisted bytes. Do not conflate H_env with image height H. Record
initial model download, storage and cold start separately for shared models;
apply the same shared-weight allowance to competitors. Per-video fitted weights
or deltas must be available to the decoder and charged under the declared policy.

The decoder receives only this payload plus explicitly shared model/configuration
assets. It must run in a fresh process with source images, uncoded masks, latent
caches and training targets inaccessible. Hash the payload and decoded frames.
Score `RunResult.delivered_frames`, not pre-codec `.frames`. A name ending in
`.bin`, a bpp field, or a passing helper test is not proof of a complete codec.

## Baseline parity

- Freeze the same source frame identities, FPS/PTS, duration, display size,
  color conversion, bit depth and crop/resize policy. Report padding and score
  only the original display extent. Resized baseline arms reconstruct to the
  same display grid and are labeled separately.
- Separate low-delay causal, offline/random-access and repeated-segment access.
  A future-built panorama or bidirectional generator must not claim causal
  latency. Compare cached PointStream across points with eligible continuous
  conventional reference access, and also show independently seekable segments.
- Pin native binary path/version, command, preset, rate-control settings and
  reference policy. VVenC is not VTM. AV1 libaom is not SVT-AV1. Do not copy a
  paper's VTM curve onto a VVenC experiment. Verify that optional ROI/reference
  settings actually alter coding; unsupported settings remain absent and labeled.
- Start with at least four valid distinct rate points per curve, add points only
  to resolve a declared overlap gap. Public learned checkpoints may provide fewer;
  report that limitation instead of manufacturing extra points.
- BD-rate requires common quality support, appropriate direction, non-degenerate
  curves and the reported integration interval. No extrapolation. For a
  lower-is-better error metric, use a declared monotone conversion consistently
  (for example q=−DISTS), with tests. If no valid overlap exists, report measured
  points and local dominance; do not force a BD-rate statistic.
- A point dominates only if it is no worse on both declared axes and strictly
  better on at least one, including uncertainty/tolerance. Dominance of a
  development point is not a universal codec advantage.

## Quality: metrics are instruments, not codec baselines

AV1/VVC/MTTF/GLC are comparison methods. VMAF, LPIPS and DISTS are measurements.

| Dimension | Planned measurement | Validation / limits |
|---|---|---|
| Whole-frame fidelity | PSNR, SSIM, VMAF; exact color and VMAF model recorded | Identity and known degradations; correct reference/distorted order; do not claim VMAF alone validates generative accuracy |
| Perceptual reconstruction | LPIPS with backbone/version; DISTS | Identity/blur/noise/shift controls; DISTS integration pending; no substitution of raw VGG distance for LPIPS |
| Player appearance | Aligned per-object crop LPIPS/DISTS; source identity and padded crop policy | Same source-derived evaluation boxes for all methods; do not use each method's own detections as the metric region; masked rectangular metrics need a declared common compositing policy |
| Court fidelity | Independent line/keypoint displacement in source pixels plus court-region pixel error | Calibrate synthetic shifts; separate annotation uncertainty; line-mask PSNR is not geometric displacement |
| Ball/racket event fidelity | Visibility precision/recall, trajectory/keypoint error, ID continuity and failure rates | Independent labels; count missed detections and false hallucinations, not just error on surviving detections |
| Temporal stability | FloLPIPS or another qualified temporal implementation; motion/flicker failure examples | Pin flow/backbone and handle cuts; distinguish true motion from flicker; temporal metric integration must be verified |
| Human viewing | Blinded randomized paired study on frozen equal-rate examples | Preregister sampling, question, observer count/power rationale, exclusions and match-level aggregation; report source-fidelity separately from pleasant appearance |

FID/FVD can contextualize literature but are not primary decision metrics on a
few tennis clips: distribution scores and small-sample variation do not verify
individual ball trajectories. ReID/pose metrics can be useful diagnostics only
after null controls; do not use the encoder's own extractor as independent truth.

Prefer separate quality dimensions and geometry constraints over an arbitrary
weighted sum of LPIPS, pixels and VMAF. Any composite weight/threshold must be
calibrated on development and frozen before confirmation. Preserve the historical
0.7 FG + 0.3 BG PSNR only as a diagnostic column, not a perceptually justified new
primary metric. Geometry/failure tolerances are protocol parameters to calibrate
in E01, not invented success thresholds.

## Required controls and ablations

Use staged paired comparisons, not a Cartesian sweep. Start with conventional
whole video; coded/warped reference foreground; coded foreground video; and the
selected generator. Compare the same background/appearance/placement budget.
Then isolate background removal/registration/refresh, decoded masks versus
charged silhouettes, pose representation/quantization, racket/ball transport,
residual off/coarse/fine, fallback, and reference reuse. Include source-geometry
oracles only as explicitly noncompetitive ceilings. Charge model updates in
overfit controls. A useful representation must improve the complete measured
frontier, or justify a separately measured task/computation benefit with its
quality/rate cost disclosed.

Record preparation, sender inference, entropy coding, network serialization,
decoder/model loading, reconstruction, residual decode, compositing and scoring
separately. Report end-to-end latency, throughput and allocated/reserved memory
on the named GPU/CPU. Use synchronized timing and include startup/lookahead;
exclude interrupted/foreign-GPU-contaminated runs from timing claims while
preserving their files. Quality results and timing eligibility are separate.

## Current results and intended paper outputs

No confirmed generative-comparison or general AV1/VVC win exists at this audit.
The [evidence ledger](08-evidence-ledger.md) preserves losing development rows,
the counterfactual headroom study, data-readiness results and withdrawn claims.
External artifacts have not been rehashed by this documentation task.

Produce: (1) an input/split/availability table; (2) full-wire rate–quality curves
with metric and domain scope; (3) court/ball/racket fidelity at matched rate;
(4) component byte/error ablations; (5) actual cumulative-rate/quality curves
across consecutive points; (6) client time/memory and failure table; (7) frozen
confirmation outcomes with uncertainty; (8) negative results and artifact
limitations. Generate them from qualified manifests and saved decoded frames.
The manuscript must remove unsupported live/never-worse/priority assertions,
and reconcile stale source versions before quantitative integration.
