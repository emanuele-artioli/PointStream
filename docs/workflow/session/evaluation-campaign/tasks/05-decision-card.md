# E05 Stage 1 Executable Card & Baseline Decision: Foreground Generation Candidates

**Date**: 2026-09-16  
**Status**: RELEASE-READY FOR STAGE 1 PILOT (Specification complete; authorized under 1.0 GPU-hour cap; not launched yet)  
**Author**: Antigravity (Pair Programming Session)  
**Task Reference**: [05-foreground.md](05-foreground.md) / [20260916-probe-review.md](20260916-probe-review.md) / [02s-training-and-controls.md](02s-training-and-controls.md)

---

## 1. Executive Summary & Release Status

This card defines the executable Stage 1 pilot protocol for foreground generation candidate families (`pix2pix`, `spade4tennis`, and baseline reference models).

### Circularity Resolution
Previous draft criteria required running a pilot training run *before* releasing the Stage 1 specification. That circular gate is resolved: this document establishes the frozen, executable Stage 1 specification, pre-registered bounds, controls, charged accounting ledger, and promotion criteria. Stage 1 is **RELEASE-READY FOR PILOT EXECUTION** under an aggregate 1.0 GPU-hour cap. No training runs are launched in this specification session.

---

## 2. Evidence Base & Existing Artifact Acceptance

The protocol builds on existing, immutable acceptance and diagnostic artifacts under `$PS_DATA_ROOT/outputs/evaluation-campaign/e02r/`:

1. **Diagnostic Matrix Report**:
   - Path: `outputs/evaluation-campaign/e02r/diagnostic_matrix_pix2pix_scene028.json`
   - SHA-256: `cec034102a3e00c2073158807278312ea3f7ccf56221918331e86dc308af7426`
   - Scope: 5-corner diagnostic probe on 16 frames of `alcaraz_highlights/scene_028` (crop $256 \times 256$, native $1080 \times 1920$).
   - Reconciled Wire: 100% byte reconciliation (`exact_match: True`, total 7,023,087 bytes).
   - Residual-Off Control: Confirmed 0 residual bytes and 0 residual calls.
   - Generative Residual Demand: 110,913 bytes for Pix2Pix vs 113,287 bytes for pasted reference keyframe.
   - Conditioning Sensitivity: Confirmed (16/16 frames have distinct delivered pixel hashes between conditioned and shuffled conditioning).

2. **Campaign Ingestion Result**:
   - Path: `outputs/evaluation-campaign/e02r/campaign_result_pix2pix_scene028.json`
   - SHA-256: `296dc0c98dc441ab85676cf3d4950b58ff35b6e2ce61f3ae97c2e9db9151df84`
   - Schema: `pointstream.campaign_result.v1`
   - Claim Eligibility: `standalone_transport: true`; `rd: false`, `runtime: false`, `generalization: false` (correctly scoped historical probe).

3. **Checkpoint Continuation Evidence**:
   - Path: `outputs/evaluation-campaign/e02r/checkpoint_resume_evidence/pix2pix_interrupted_checkpoint.pt`
   - SHA-256: `9109f53237098ff71dfcba7f1a25fe24ae23814a06311df56c29ea7782a61e9d`
   - Verified Scope: Tested via `test_fresh_process_trainer_cli_continuation` under single-worker CPU execution (`CUDA_VISIBLE_DEVICES=""`, `--num-workers 0`, `--reference-mode first`).
   - Verified Numerical Exactness: Proved exact bitwise equality (`torch.equal`, `max_diff == 0.0`) for generator weights (G), discriminator weights (D), Adam optimizer moment buffers (`exp_avg`, `exp_avg_sq`), and RNG states (torch, numpy, python) across fresh-process resume. Scope is strictly bounded to CPU single-worker execution; no CUDA bit-identity is claimed without GPU continuation evidence.

---

## 3. Corrected Protocol Foundations

The Stage 1 pilot enforces the protocol repairs implemented in E02S:

1. **Corrected First-Reference Policy**:
   - Explicit reference selection via `--reference-mode first` in `TennisSkeletonDataset` (`src/shared/tennis_dataset.py`).
   - The reference appearance for each object track is anchored to its initial appearance ($t=0$, `colors[0]`).
   - For all frames $t > 0$, `ref != target` is guaranteed, preventing the network from exploiting the historical target-copy shortcut (`target == ref`).
   - Target match is legitimately allowed only at frame $t = 0$.
   - Any legacy checkpoint trained under the uncorrected shortcut is flagged with `used_reference_shortcut = True` and disqualified from candidate promotion.

2. **Invalid Candidate Handling & Reconciled Indifference Bands**:
   - In `scripts/train_campaign.py`, candidate evaluation rejects missing, `NaN`, or domain-invalid values (`psnr < 0`, `ssim < -1` or `> 1`), marking them incomparable.
   - Incomplete or invalid candidate records can never dominate valid measured candidates.
   - Reconciled indifference bands:
     - Total wire rate: $\le 2.0\%$
     - PSNR: $\le 0.10$ dB
     - SSIM: $\le 0.005$
     - Client latency: $\le 5.0\%$ relative

---

## 4. Stage 1 Pilot Execution Protocol

### 4.1 Resource Budget & Execution Limits
- **Aggregate GPU Cap**: Hard aggregate cap of **1.0 GPU-hour total wall-clock time** across all Stage 1 pilot training and diagnostic runs combined.
- **Hardware Isolation**: Single GPU device allocation (e.g. `CUDA_VISIBLE_DEVICES=1` on RTX 6000 Ada); GPU 0 remains undisturbed.
- **Checkpointing Cadence**: Mandatory intra-epoch hourly checkpointing (`--checkpoint-interval-sec 3600.0`) with atomic file writes (`save_checkpoint_atomic`).
- **Heartbeat Cadence**: Progress logging heartbeat at least every 10 minutes (`now - last_progress_time >= 600.0`).
- **Process Timeout**: Training command wrapped in a hard timeout (`timeout 3600`) to guarantee adherence to the 1 GPU-hour cap.

### 4.2 Phase 1A: Tiny-Scene Learnability Check
Before conducting broader validation, the candidate model must demonstrate basic learnability on a tiny development sequence:
- **Clip**: 16–32 consecutive frames from development scene `alcaraz_highlights/scene_028` (crop size $256 \times 256$, single worker/batch).
- **Reference Mode**: `--reference-mode first`.
- **Learnability Gate**:
  - Generator loss ($L_{\text{GAN}} + \lambda L_1$) and discriminator loss ($L_D$) must train stably across steps.
  - Gradients must propagate without numerical divergence (`NaN` or `Inf`).
  - Output samples (`assets/samples/epoch_*.png`) must show meaningful appearance transfer conditioned on target pose rather than mode collapse or complete blanking.

### 4.3 Phase 1B: Disjoint Development Validation
Following the tiny-scene learnability check:
- **Evaluation Split**: Evaluated on disjoint development data strictly held out from the Phase 1A training frames (e.g. frames 32–47 of `alcaraz_highlights/scene_028` or disjoint development scene `scene_001` from the development pool).
- **Preserved Confirmation Sources**: Confirmation test scenes (e.g. `federer007` and held-out evaluation test splits) remain **strictly quarantined**. Zero pilot steps, parameter tuning, or exploratory evaluations touch confirmation data.

### 4.4 Baseline Controls
The candidate generative model (`gen_on_res_off`) must be compared against standard reference baselines on the identical disjoint development sequence:
1. **Control A (Pasted Reference Baseline)**:
   - Keyframe pasted reference (`gen_off_res_off`), transmitting reference frame 0 and placing it directly into target bounding boxes.
   - Secondary matched residual-on point (`gen_off_res_on` vs `gen_on_res_on` at QP 32) for codec context.
2. **Control B (Supported Warped Reference Baseline)**:
   - Warped reference baseline, where reference keyframe 0 is warped via optical flow or affine transformation to match target pose/bbox geometry prior to placement.
3. **Diagnostic Diagnostic Matrix Controls**:
   - Same-seed determinism: two independent forward passes with the same seed must produce bit-identical outputs.
   - Pose conditioning sensitivity: delivered pixel hashes must differ between normal and shuffled pose conditioning.
   - Blank conditioning control: delivered pixel hashes must differ between normal pose and zeroed/blank conditioning.
   - Residual-off control: when residual is disabled, residual bytes and calls must strictly equal 0.

### 4.5 Charged Total Wire Accounting
Comparisons are evaluated on **whole-codec charged rate**, not isolated crop payloads:
$$B_{\text{total}} = B_{\text{bg\_plate}} + B_{\text{bg\_stream}} + B_{\text{fg\_ref}} + B_{\text{pose\_motion}} + B_{\text{masks}} + B_{\text{weights\_adapter}} + B_{\text{residual}}$$
- Background plate and stream must be explicitly costed.
- Foreground reference keyframe transmission must be costed.
- Pose/motion metadata and segmentation mask metadata must be fully charged to the bitstream.
- Per-video or per-sequence model weights/adapters (if transmitted) must be charged.
- Wire byte reconciliation (`exact_match: True`) is mandatory.

### 4.6 3D Metric Reporting
Every evaluation must report all three core dimensions:
1. **Rate**: Total charged wire bytes and bits-per-pixel (bpp).
2. **Quality**: Objective fidelity measured as PSNR-Y (dB), whole-frame windowed SSIM, and VMAF.
3. **Speed**: Measured client decode and inference latency (seconds per frame and fps) on a standardized client hardware stratum.
*No candidate may be promoted on rate or quality alone without reported client latency.*

---

## 5. Explicit Promotion & Baseline-Clearing Criteria

### 5.1 Pre-Registered Two-Sided Bounds
Prior to analyzing pilot results, measured metrics must fall within the pre-registered plausible bounds:
- **PSNR-Y**: $[25.0\text{ dB}, 45.0\text{ dB}]$ (Values $< 25.0$ dB indicate generator collapse; values $> 45.0$ dB on lossy video indicate instrument corruption).
- **SSIM**: $[0.70, 0.98]$ (Values $< 0.70$ indicate severe distortion; values $> 0.98$ indicate target leakage).
- **Wire Rate Ratio**: $[0.80, 1.50]$ relative to pasted reference baseline.
- **Client Inference Latency**: $[10\text{ ms}, 250\text{ ms}]$ per frame.

Any measurement outside these intervals constitutes an **alarm** requiring instrument and data pipeline auditing before reporting.

### 5.2 Baseline-Clearing Promotion Rules
A candidate model is **promoted to Stage 2** if and only if it satisfies all of the following conditions on the disjoint development evaluation:
1. **Fidelity / Rate Pareto Advantage**:
   - Achieves lower total charged wire rate at matched or superior quality (PSNR $\ge \text{baseline} - 0.10$ dB and SSIM $\ge \text{baseline} - 0.005$), OR
   - Achieves superior quality (PSNR $> \text{baseline} + 0.10$ dB or SSIM $> \text{baseline} + 0.005$) at matched wire rate ($\text{rate} \le \text{baseline} \times 1.02$).
2. **Client Latency Budget**:
   - Measured client latency does not exceed the declared client budget ($\le 105\%$ of baseline latency target).
3. **Passed Controls**:
   - Passes same-seed determinism, conditioning sensitivity, blank conditioning, and residual-off verification.
4. **Valid Evaluation**:
   - Zero `NaN` values, no domain-invalid numbers, and full byte reconciliation.

### 5.3 Ambiguity Extension Policy
If a candidate demonstrates consistent loss reduction and passes all controls but finishes within the indifference bands of the baseline (neither strictly clearing nor failing):
- It is granted a single bounded **ambiguity extension of $\le 0.5$ GPU-hour** for targeted learning-rate refinement before final disposition.
- If it still fails to clear the baseline after the extension, it is formally classified as **failed under recorded budget** or **deferred**.

---

## 6. Concrete Executable Run Commands

When dispatched by the coordinator, the pilot execution will run the following concrete commands:

```bash
# 1. Environment & GPU Isolation (GPU 1)
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="."
export PYTHONNOUSERSITE=1

# 2. Phase 1A: Tiny-Scene Pilot Training (1 GPU-hour cap with 10-min heartbeat)
timeout 3600 python scripts/train_pix2pix.py \
  --data-root "$PS_DATA_ROOT/dataset" \
  --condition pose_body \
  --reference-mode first \
  --epochs 20 \
  --batch-size 4 \
  --img-size 256 \
  --lr 0.0002 \
  --seed 42 \
  --checkpoint-interval-sec 3600.0 \
  --out-weights "$PS_DATA_ROOT/outputs/evaluation-20260914/e05/stage1_pilot/generator_pix2pix.pt" \
  --checkpoint-path "$PS_DATA_ROOT/outputs/evaluation-20260914/e05/stage1_pilot/checkpoint_pix2pix.pt" \
  --sample-dir "$PS_DATA_ROOT/outputs/evaluation-20260914/e05/stage1_pilot/samples"

# 3. Diagnostic Matrix & Controls Verification on Disjoint Development Frames
python scripts/run_diagnostic_matrix.py \
  --weights "$PS_DATA_ROOT/outputs/evaluation-20260914/e05/stage1_pilot/generator_pix2pix.pt" \
  --scene-path "$PS_DATA_ROOT/dataset/alcaraz_highlights/scene_028" \
  --output-dir "$PS_DATA_ROOT/outputs/evaluation-20260914/e05/stage1_pilot/matrix" \
  --reference-mode first \
  --full-trajectory \
  --seed 42

# 4. Result Ingestion & Schema Validation
python -m src.runner.generation_adapter \
  --input-matrix "$PS_DATA_ROOT/outputs/evaluation-20260914/e05/stage1_pilot/matrix/diagnostic_matrix.json" \
  --output-record "$PS_DATA_ROOT/outputs/evaluation-20260914/e05/stage1_pilot/campaign_result.json"
```

---

## 7. Status & Dispatch Handoff

- **Stage 1 Specification**: COMPLETE and VERIFIED.
- **Circular Gate**: REMOVED.
- **Confirmation Sources**: QUARANTINED and PRESERVED.
- **Execution Status**: PENDING COORDINATOR RELEASE / DISPATCH.
- **Immediate Action**: Return to coordinator for pilot release; **do not launch yet**.
