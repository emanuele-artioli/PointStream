# E05 First-Stage Decision Card: Foreground Generation Candidates

**Date**: 2026-09-16  
**Status**: CONDITIONAL HOLD — E05 and confirmation scoring remain **unreleased**  
**Author**: Antigravity (Pair Programming Session)  
**Task Reference**: [05-foreground.md](05-foreground.md) / [20260916-probe-review.md](20260916-probe-review.md)

---

## 1. Executive Summary

This decision card establishes the first-stage baseline-clearing evaluation status for foreground generation candidate families (`pix2pix`, `spade4tennis`, and pasted reference baseline) derived from existing immutable diagnostic and readiness artifacts.

No new candidate encodes, no confirmation scoring, and no multi-hour training jobs were launched. This assessment bounds the candidate readiness from existing artifacts and sets explicit gates before any E05 Stage 1 execution (1 GPU-hour cap) may be released.

---

## 2. Evidence Base & Existing Artifacts

The analysis is based strictly on existing, immutable artifacts recorded under `$PS_DATA_ROOT/outputs/evaluation-campaign/e02r/`:

1. **Diagnostic Matrix Report**:
   - Path: `outputs/evaluation-campaign/e02r/diagnostic_matrix_pix2pix_scene028.json`
   - SHA-256: `cec034102a3e00c2073158807278312ea3f7ccf56221918331e86dc308af7426`
   - Content: 5-corner diagnostic probe on 16 frames of `alcaraz_highlights/scene_028` (crop $256 \times 256$, native $1080 \times 1920$).
   - Reconciled Wire: 100% byte reconciliation (`exact_match: True`, total 7,023,087 bytes).
   - Residual-Off Control: Confirmed 0 residual bytes and 0 residual calls.
   - Generative Residual Demand: 110,913 bytes for Pix2Pix vs 113,287 bytes for pasted reference keyframe.
   - Conditioning Sensitivity: Confirmed (16/16 frames have distinct hashes between conditioned and shuffled conditioning).

2. **Campaign Ingestion Result**:
   - Path: `outputs/evaluation-campaign/e02r/campaign_result_pix2pix_scene028.json`
   - SHA-256: `296dc0c98dc441ab85676cf3d4950b58ff35b6e2ce61f3ae97c2e9db9151df84`
   - Schema: `pointstream.campaign_result.v1`
   - Claim Eligibility:
     - `standalone_transport`: `true` (valid client envelope).
     - `rd`: `false` (exclusions: missing measured objective quality across corners, missing same-seed determinism control row in probe run).
     - `runtime`: `false` (exclusions: timing missing or not a transferable host stratum).
     - `generalization`: `false` (exclusions: single development scene `scene_028`, not held-out split).

3. **Checkpoint Continuation Evidence**:
   - Path: `outputs/evaluation-campaign/e02r/checkpoint_resume_evidence/pix2pix_interrupted_checkpoint.pt`
   - SHA-256: `9109f53237098ff71dfcba7f1a25fe24ae23814a06311df56c29ea7782a61e9d`
   - Proven: Hourly intra-epoch atomic checkpointing with exact bitwise continuation of G, D, optimizers, and RNG state on single-worker CPU verification.

---

## 3. Candidate Family Evaluation & Deficiencies Identified

| Candidate | Architecture / Backend | Reference Policy | Status | Identified Deficiencies / Roadblocks |
|---|---|---|---|---|
| **Pasted Reference** | Baseline (no neural generator) | Keyframe (frame 0) | Benchmark Baseline | High residual demand (113,287 B), no temporal adaptation |
| **Pix2Pix** | `UNetGenerator` (54M params) | Historical target-copy shortcut | Pledged / Hold | Prior training used target-copy shortcut (`used_reference_shortcut=True`); needs retrain under `"first"` reference policy |
| **SPADE4Tennis** | `SPADEResNet9Generator` | Historical target-copy shortcut | Pledged / Hold | Weights initialization bug fixed; needs retrain under `"first"` reference policy; multi-discriminator capacity costed |

### Critical Protocol Gaps Closed in E02S Repair:
1. **Reference Selection Policy**:
   - *Previous Flaw*: In `src/shared/tennis_dataset.py`, deterministic reference mode returned target image `colors[idx]`, allowing the generator to learn an identity copy shortcut instead of generation from reference.
   - *Correction*: Explicit reference policy implemented (`"first"`, `"keyframe"`, `"offset"`, `"random"`). Default `"first"` mode anchors on track start (`colors[0]`), ensuring `ref != target` for all frames $t > 0$.
   - *Legacy Handling*: Historical checkpoints are flagged with `used_reference_shortcut = True` to prevent silent misattribution while preserving model families.
2. **Invalid Candidate Handling**:
   - `scripts/train_campaign.py` now marks evaluations containing `NaN` or out-of-domain metrics (`psnr < 0`, `ssim < -1` or `> 1`) as incomparable. Missing quality metrics can no longer outrank measured candidates.
   - Indifference bands declared and reconciled: rate 2%, PSNR 0.10 dB, SSIM 0.005, client latency 5%.

---

## 4. First-Stage Decision & Release Gate

### Decision: **HOLD (Unreleased)**
E05 Stage 1 (1 GPU-hour pilot training budget) is **NOT RELEASED**.

### Release Conditions:
Before advancing to Stage 1 execution:
1. **Pilot Clean Run**: Execute a single bounded 1-epoch pilot training run of `pix2pix` under the corrected `--reference-mode first` policy on a single GPU with isolated device reservation.
2. **Pre-Registered Bounds**: State explicit two-sided bounds on PSNR, SSIM, and bits-per-pixel before evaluating the resulting checkpoint.
3. **Controls Compliance**: The evaluation must execute all required diagnostic controls (conditioned vs. blank, conditioned vs. shuffled, same-seed determinism, residual-off).
4. **Multi-Scene Split**: Initial baseline-clearing must be confirmed on at least two distinct scenes (`scene_028` and an unseen test scene) before claim promotion.
