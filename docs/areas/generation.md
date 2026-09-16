# Generation Area

## Conditional pilot acceptance — 16 September 2026

#118 at `e73c8ba` has green CI. Corrected first-reference/domain handling and
exact tested G/D/Adam-moment CPU CLI continuation are advances; no CUDA identity
claim is accepted. The card still lacks a real tiny subset, valid evaluator/API
commands and an aggregate deadline. The current brief authorizes these fixes
then a conditional E05 pilot within one total GPU-hour, including diagnostics
and ambiguity extension. Federer scene 007 is development; the reserved trio is
quarantined. #118 is not accepted as an executable release at its reviewed head.
Follow [the current launch gates](../workflow/session/evaluation-campaign/tasks/20260916-bounded-pilot-release.md).


## Historical review — 16 September 2026

#109 at `c9c4ded` has green CI, an actual fresh-CLI continuation regression,
restored Python RNG and isolated DataLoader RNG. Supplied diagnostic/adapter
artifact hashes match. These are advances over the prior review.

Acceptance remains open for two reproduced cases: deterministic references use
`colors[idx % len(colors)]`, feeding every first-track target as its own
reference; and a cheaper NaN-PSNR candidate can dominate a measured candidate.
Correct the reference policy to match receiver availability and reject invalid
evidence before promotion. The fresh CLI test is CPU-only and uses tolerance;
do not label it bit-identical CUDA continuation without that evidence. Reconcile
the report's bands with code. Repeat only the tiny continuation check after the
reference fix, preserving old checkpoints/diagnostics. E05 and confirmation
scoring remain unreleased. Follow the [focused assignment](../workflow/session/evaluation-campaign/tasks/20260916-probe-review.md).

## Historical acceptance — 15 September 2026

#109 `013a4c8` retains useful epoch sampling, RNG tensor restoration, explicit
residual-OFF handling and extra control corners. E05 remains unreleased: its
test does not exercise fresh-process CLI/DataLoader/discriminator continuation;
Python reference-choice and iterator RNG are not restored deterministically.
Selection ignores SSIM and treats missing quality/time as equal. Complete only
these cases and supply exact host/path/command/digest pointers for the reported
artifacts; do not repeat the historical matrix. See the
[bounded assignment](../workflow/session/evaluation-campaign/tasks/20260915-next-stage.md).
Cursor's adapter recognizes the producer same-seed corner after #104 integration.
Full-trajectory execution is already verified, not proof of useful fidelity or
generalization. E03/E04 proceed independently with generation OFF.

## Historical acquisition / integration review

E02S has completed bounded real acceptance on GPU 1 (RTX 6000 Ada) under verified
atomic resource claim: intra-epoch atomic checkpoint saving and fresh-process resume
achieved bit-identical optimizer update (`diff: 0.00e+00`), and a 7-corner diagnostic
matrix verified bit-identical same-seed determinism and pose conditioning sensitivity.
Cursor exclusively owns result adapters (E03A); adapter edits are excluded from this PR.
E05 remains unreleased awaiting E03A/E04 completion; offline/deferred families remain preserved.

## Coordinator follow-up — E01/E02

Coordinator review of #105 (`2f63ae1`): implementation partial; E05 not released.
Merged CI and request-count/synthetic controls do not establish loaded-backend
readiness. Adapter schema/default-control defects, epoch-bound checkpoints and
always-on residual path require [E02R](../workflow/session/evaluation-campaign/tasks/02r-readiness-evidence.md).
SPADE full is not yet a different generator architecture; correct pretrained
initialization before relying on it. Supersedes broader Ready/completed labels below.

## Current campaign — 14 September 2026

E02/E05 in the [campaign](../workflow/session/evaluation-campaign/plan.md) supersede
older dispatch restrictions. PR #102 repairs the identified boundary paths;
full-trajectory evaluation and training readiness still need verification.
Low-resolution staged training can run alongside background work after evaluator
and split readiness. At least one neural model must clear declared baselines;
then prioritize writing and keep further improvements bounded.

**Evidence Revision**: Reconciled through PR #28 (`66da545dcf`) and PR #72.
**Owned Scope**: Generative backends (`src/components/generation/`), ControlNet, IP-Adapter, Animate-Anyone, SVD, generator dispatch.

---

## Current audit — 2026-09-12

Keep generation OFF as the current conservative operational baseline, not an architecture ranking.
[The audit](../history/antigravity-audit-2026-09-12.md) finds that the 16-frame diagnostic matrix
tested sparse placements (frame index 0 only, per `load_long_scene_clip` which creates an
`ObjectRequest` at first appearance per track). This proves pixel modification and conditioning
sensitivity at injection, but is not full-sequence continuous pose tracking. Whole-frame null
scores were almost unchanged across subsequent frames.
Broad claims that pasted reference is a "strictly superior model" are retracted: pasted reference
is retained as a conservative operational baseline (`generation-off`), while `pix2pix` and
`spade4tennis` remain available for multi-frame evaluation rather than permanently discarded.
`GEN-ACT-08` is only partially resolved at integration boundaries; complete the relevant `EVAL-ACT-11`
repairs before model comparison. Next session follows [the handoff](../workflow/session/evaluation-handoff.md).

## 1. Current State

PointStream is architected so that generative synthesis is a modular, optional stage (`STAGE_GENERATION` in `src/contracts/lattice.py`).

### Key Empirical Finding
Earlier benchmark reports favored **pasted reference keyframes over the tested generative configurations** on objective fidelity metrics (PSNR and SSIM). These are scoped historical findings, not a validated comparison of every backend or a prediction about future training:
- Generative models introduced spatial hallucination, boundary bleeding, and frame-to-frame temporal flicker.
- In-domain fine-tuning improved perceived realism in isolated crops but degraded whole-frame fidelity compared to reference pasting.
- **Operating Policy (revised 2026-09-09)**: The audited Gate A/B sweeps and shipped tiers configure generation OFF. Earlier comparisons favor pasted references in their tested settings; this does not establish that future trained generators cannot help. Permit bounded generator validation and training before a generation-free win. Select by whole-codec rate–distortion and runtime, including residual demand and transmitted model data, rather than crop realism alone.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Engine Roster | #20 (`fd36b4b338`), #27 (`8d9a4b1669`), #28 (`66da545dcf`) | Evaluated ControlNet, IP-Adapter, Animate-Anyone against pasted reference. |
| Generator Interface | #28 (`66da545dcf`) | Established `GeneratorRef` contract and disabled-by-default behavior. |
| Negative Result | Paper & #28 | Formally documented as a paper finding: pasted reference was the stronger control in the tested configurations; broader claims require renewed validated comparisons. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `GEN-ACT-01` | Previously deferred (D2) | Gate A passed | `plans/DEFERRED.md` | **SAM3 segmentation evaluation**: Assess SAM3 in an isolated conda environment with newer PyTorch support. Acceptance: Verify whether segmentation quality improves crop boundary precision without breaking current environment pins. |
| `GEN-ACT-02` | Previously deferred (D4) | Gate A passed | `plans/DEFERRED.md` | **SVD temporal consistency test**: Evaluate Stable Video Diffusion components under isolated license and runtime boundaries. Acceptance: Strict evaluation against the pasted-keyframe control under matched bitrates. |
| `GEN-ACT-03` | Closed / Archived | None | `plans/DEFERRED.md` (D7) | **Training stop rules**: Unconstrained training remains deferred; bounded pilots are allowed under GEN-ACT-04 without prior generation-free parity. |
| `GEN-ACT-04` | Ready | Validated evaluator before training; no generator-free win prerequisite | 2026-09-09 residual/generation audit | **Generator readiness and bounded training**: Inventory actual checkpoints, verify native temporal inference and conditioning, restore the retired training evaluator, then compare a bounded pilot by total coded rate–distortion/runtime. See [worker brief](../workflow/session/generator-readiness.md). |

## PR #88 generator audit — 2026-09-10

The one-epoch pix2pix/SPADE4Tennis-lite pilot is **not a defensible architecture ranking or proof of adequate training**. Both logs show a completed one-epoch run (pix2pix reports 16,272 images and batch 8); SPADE also has an OOM retry log. There is no recorded hyperparameter sweep, convergence study, paired untrained/pasted baseline or source-level uncertainty. The default commands use learning rate 0.0002; pix2pix uses pixel loss weight 100, while this custom SPADE variant uses pixel/VGG/feature-matching weights 10 each. Those differences need model-specific tuning, not forced identical hyperparameters. The campaign state contains only these two variants; the default driver also lists ControlNet, and Animate-Anyone training is explicitly unwired. Omitted models were not shown inferior.

More fundamentally, `evaluate_checkpoint` reads resized actor crops, installs an artificial central-quarter bbox/mask, silently fills missing sources with grey, and calls its results full reconstructed clips. The current full-dataset/probe-manifest pairing has **9/12 tracks with zero requested local-numbered source files**, while their global-numbered source files exist. The manifest distinguishes these coordinates. The archived campaign does not capture enough input-path/configuration provenance to certify which data were scored; invalidate its promotion/pruning decision and rerun after repairing this path. The evaluator also silently falls back from an empty held-out-video selection to all probes; this manifest has no probes from its listed held-out videos. Development track holdouts must not be described as independent-source confirmation.

Other blockers: an empty/missing-metric aggregate can be accepted as successful; temporal frame-difference error is mislabeled FVD; device/seed parameters are accepted without being applied in the evaluator; checkpoint identity uses filename/size/mtime rather than a content hash. Ranking fixed-QP residual bytes does not enforce matched final quality or complete payload cost. The campaign halts with one survivor instead of training the selected candidate to its allocated final budget. Do not resume its pruned state as a justified selection.

Animate-Anyone's smoke script exercises sequence inference and conditioning sensitivity, but does not itself launch or compare two fresh processes. A claimed fresh-process determinism result requires separate immutable evidence. Sensitivity alone does not establish a useful predictor. Readiness-screen all registered backend families, collapse aliases, and record an explicit reason for deferring each untested candidate.

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `GEN-ACT-05` | Ready; blocks ranking | EVAL-ACT-06 / CODEC-ACT-05 integration | PR #88 audit | Repair dataset coordinates, missing-data rejection, real-frame evaluation, checkpoint identity and controls. Re-evaluate both candidates; restore SPADE to the pool. |
| `GEN-ACT-06` | Inventory ready; training gated on evaluator | GEN-ACT-05 | PR #88 audit | Multi-fidelity model/configuration search: existing weights first, meaningful resource rungs and per-architecture hyperparameters, matched-fidelity total wire cost/runtime, learning curves and confirmation-safe splits. Follow [submission search](../workflow/session/submission-search.md). |

Repair update — 2026-09-10: coordinate resolution, missing-input rejection, whole-codec evaluation, SHA-256 checkpoint identity, matched configuration controls and the single-survivor campaign continuation have been integrated and tested. `manifests/candidate_inventory.json` is intentionally marked inventory-only: its readiness labels and runtime figures are not selection evidence without linked immutable artifacts. No candidate has been newly ranked or trained by the repaired path.

Recovery closeout — 2026-09-10: PR #88 merged as `2b7c2b0` with CI passing. A three-frame CPU smoke of the immutable pix2pix checkpoint passed its non-source-fallback and shuffled-pose sensitivity controls. This only establishes basic backend readiness; it is neither a quality comparison nor a runtime result.

GPU diagnostic matrix (gpu5, 16 frames, pix2pix, residual QP 32, artifact `diagnostic-pix2pix-rq32.json`): **invalid for generator comparison**. Generation-off and generation-on without residual produced identical PSNR/SSIM/VMAF because `_finish_chunk` converted appearance-conditioned objects into pasted-reference placements (`is_gen` inferred from empty `supplied_crop`). Generation-on added only ~180 bytes; client time was unchanged. Encoder-side generation timing of 0.7–1.2 s is not evidence that generated pixels reached the client. Residual-on still moved quality on the paste arm (~+8 dB PSNR-Y); that does not validate pix2pix. Reports lacked checkpoint SHA, delivered hashes, and a shuffled-conditioning control. Preserve the artifact; do not cite it as a model result. Staged training remains unlaunched until a repaired path shows generation changing delivered frames and residual demand.

## Foreground experiment priority — 2026-09-11

`GEN-ACT-07` — Planned; broad model search follows the component/headroom diagnosis
in `EVAL-ACT-08` and `CODEC-ACT-06`, rather than a required generation-free win.
Follow [resource-dependent frontiers](../workflow/experiment-design.md#4-foreground-models-test-resource-dependent-frontiers).
Start from pasted-reference and ready-model controls, with small-model and
diffusion advantages stated as falsifiable hypotheses under named client budgets.
Charge conditioning, weights/updates and matched-quality correction; measure
latency, throughput and peak memory. Preserve older trials without reusing their
invalid rankings. SPADE/training remains off pending a newly scoped experiment
card. Acceptance: a supported Pareto tradeoff or a bounded rejection/inconclusive
result that determines the next experiment; no family-wide claims from one pilot.

`GEN-ACT-08` — **Complete** (PR #93, `4a55573` / `e9f781f`). Enforced fail-closed client checkpoint resolution against transmitted SHA-256 digest (`resolve_client_checkpoint`), full effective configuration identity, declared control validation, and dynamic clip start frame resolution without synthetic fallback.

`GEN-ACT-09` — **Complete with Scope Limitations** (Wave 2 Diagnostic Matrix, artifacts `outputs/development-recovery/diagnostic-pix2pix-alcaraz.json` SHA-256 `ccfaa34b8ceca48409839c3baefec20e91f87a0425aea6da2b7429c37ed2fa50` and `diagnostic-pix2pix-federer.json` SHA-256 `407148416bf1455ccfb68cc025a2ff31899689ebd0ffc722fd4f0dddc085af25`).
- **Validity Criteria Met**: Generator comparison validity is strictly `true` for both development scenes; `delivered_pixels_changed` is `true`; `gen_on_vs_paste_hash_match` is `false`; `shuffled_conditioning_changed_pixels` is `true`; `wire_reconciliation` matched 100% of serialized bytes.
- **Instrument Verification & Testing Scope**: Resolved skeleton pose alignment by track position (matching crop global IDs to 0-based skeleton index). However, per audit Finding 4, `load_long_scene_clip` creates one `ObjectRequest` at first appearance per track (frame 0 only). The test therefore exercised sparse first-frame placement rather than continuous 16-frame sequence tracking. Whole-frame null scores across the 16-frame span remained almost unchanged.
- **Empirical Findings on pix2pix (16 frames @ 4K)**:
  - *Alcaraz*: Pasted reference (`gen_off_res_off`) achieves 34.55 dB PSNR-Y / 90.81 VMAF for 523,610 B. pix2pix without residual (`gen_on_res_off`) yields 33.50 dB PSNR-Y (-1.05 dB) / 90.56 VMAF (-0.25) while costing 957,945 B (+434,335 B, primarily pose conditioning metadata). With residual (QP 32), pasted reference (`gen_off_res_on`) reaches 42.62 dB / 93.35 VMAF for 600,656 B, while pix2pix (`gen_on_res_on`) achieves 38.88 dB (-3.74 dB) / 93.16 VMAF for 1,035,763 B.
  - *Federer*: Pasted reference (`gen_off_res_off`) achieves 30.46 dB PSNR-Y / 78.13 VMAF for 627,735 B. pix2pix without residual (`gen_on_res_off`) yields 29.82 dB (-0.64 dB) / 77.75 VMAF for 1,282,341 B (+654,606 B). With residual, pasted reference (`gen_off_res_on`) reaches 39.76 dB / 90.01 VMAF for 796,646 B, while pix2pix (`gen_on_res_on`) reaches 37.40 dB (-2.36 dB) / 89.50 VMAF for 1,451,803 B.
- **Decision & Calibration**: Confirmed that client generator executes faithfully and alters reconstructed pixels at injection as conditioned. However, in this sparse placement setting, uncompressed pose conditioning overhead degrades whole-codec rate–distortion against pasted reference. Broad claims that pasted reference is a "strictly superior model" are retracted; generation-off is retained as a conservative operational baseline while keeping `pix2pix` and `spade4tennis` available for multi-frame sequence evaluation.

`GEN-ACT-10` — **Complete** (E02 Generator Readiness & Interface Compliance, 2026-09-14).
- **Full-Trajectory Sequence Placement**: Added `full_trajectory` flag to `experiments/long_scenes/loader.py`, resolving Finding 4. Multi-frame clips now emit `ObjectRequest`s for all frames where tracks are visible (verified on 48-frame scene `alcaraz_highlights/scene_028`, generating 96 frame-indexed requests).
- **Pose Alignment & Fail-Closed Conditioning**: `_augment_objects_with_pose` dynamically aligns bounding box aspect ratios to appearance shapes while strictly rejecting missing or unaligned pose files.
- **Campaign Evaluator & Uncertainty-Aware Promotion**: Cleaned `scripts/train_campaign.py` by removing uncalibrated LPIPS from `LOWER_IS_BETTER` and `RANKED_METRICS`, denominating primary ranking in wire `residual_bytes`, and upgrading `promote_survivors` with uncertainty-aware threshold preservation ($\le 2\%$ rate or 0.1 dB PSNR).
- **Hourly Checkpointing & Progress**: Sourced host-wide requirement into `scripts/train_pix2pix.py` and `scripts/train_spade4tennis.py` (`time.time() - last_ckpt >= 3600` and 10-minute progress heartbeats).
- **E01 Schema Adapter**: Implemented `src/runner/generation_adapter.py` with fail-closed validation of experiment identity, conditioning sensitivity, same-seed determinism, timing evidence, and claim eligibility.
- **Candidate Cards & Roster**: Documented candidate roster, native training recipes, hyperparameter endpoints, and 3-stage budgets in `docs/workflow/session/evaluation-campaign/tasks/02-generator-readiness-report.md`.

`GEN-ACT-11` — **In Progress (E02S Acceptance Completion)** (E02R/E02S Generator Readiness & Real-Backend Evidence, 2026-09-14).
- **Exact E01 Result Contract Roundtrip**: Upgraded `src/runner/generation_adapter.py` (Cursor-owned after #104) to produce schema `pointstream.campaign_result.v1` with full 64-hex SHA-256 digest validation, handling real producer `matrix` outputs from `run_diagnostic_matrix.py`, single-source grouped uncertainty without artificial zero-width assumptions, and fail-closed claim eligibility (`standalone_transport=True`, `rd=False` with explicit reason exclusions, `generalization=False` with single-source exclusions). Validated against E01 schema validator (`validate_campaign_record` and `ingest_for_claim`).
- **Residual-OFF Disabling & Ranking**: Repaired `scripts/train_campaign.py:evaluate_checkpoint` so `residual_settings=None` disables `STAGE_RESIDUAL` in `StageLattice`, sets `residual_cfg=None`, records `residual_bytes=0` and `residual_calls=0`. Updated `rank_variants` to rank on total wire rate when residual is OFF, removing min-max composite.
- **Intra-Epoch Hourly Checkpointing & Atomic State Capture**: Enhanced `scripts/train_pix2pix.py` and `scripts/train_spade4tennis.py` with intra-epoch hourly deadline checks inside batch loops, atomic saving (`save_checkpoint_atomic`), full state capture (`epoch`, `step`, `G`, `D`, `opt_G`, `opt_D`, `sched_G`, `sched_D`, RNGs), and partial-epoch skip resume semantics.
- **SPADE Initialization Order Fix & Architecture Transparency**: Corrected `scripts/train_spade4tennis.py` so `weights_init_normal` is called before loading `args.pretrained_g`. Clarified that both lite and full tiers instantiate `SPADEResNet9Generator` (differing in discriminator count 2 vs 3), with multi-scale UNet / LocalEnhancer documented as unbuilt future work.
- **Bounded Real-Backend Probe on GPU 1 (RTX 6000 Ada)**:
  - Executed 5-corner diagnostic probe on 16 frames of `alcaraz_highlights/scene_028` (`CUDA_VISIBLE_DEVICES=1`, 860 MiB VRAM allocated, GPU 0 undisturbed, 16.3 min wall time within 30 min budget).
  - 100% of wire bytes reconciled (`exact_match: True`).
  - All 16 frames exhibited distinct delivered pixel hashes between generation-off and generation-on (`gen_on_res_off`), and between normal and shuffled conditioning (`gen_on_shuffled_conditioning`).
  - Generated residual-off control confirmed 0 residual bytes and 0 residual calls.
  - Generative residual demand changed: 110,913 B for Pix2Pix vs 113,287 B for pasted reference keyframe.
  - Immutable artifacts recorded under `outputs/evaluation-campaign/e02r/`:
    - `diagnostic_matrix_pix2pix_scene028.json` (SHA-256: `cec034102a3e00c2073158807278312ea3f7ccf56221918331e86dc308af7426`)
    - `campaign_result_pix2pix_scene028.json` (SHA-256: `296dc0c98dc441ab85676cf3d4950b58ff35b6e2ce61f3ae97c2e9db9151df84`)
    - `checkpoint_resume_evidence/pix2pix_interrupted_checkpoint.pt` (SHA-256: `9109f53237098ff71dfcba7f1a25fe24ae23814a06311df56c29ea7782a61e9d`)

