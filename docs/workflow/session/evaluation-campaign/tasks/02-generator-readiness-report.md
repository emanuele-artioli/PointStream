# E02 — Generator Readiness & Candidate Roster Report

**Task**: E02 Generator Readiness  
**Area**: Generation (`docs/areas/generation.md`)  
**Worktree**: `/tmp/pointstream-eval-e02`  
**Branch**: `codex/eval-e02` (based on `origin/main` at `eeb2c81`)  
**Date**: 2026-09-14  

---

## 1. Prerequisite Notice for Coordinator

> [!IMPORTANT]
> **Atomic Host/Device Claim Policy Prerequisite**:
> Per campaign plan line 123–128 and the project constraints on this shared multi-tenant GPU server (`gpu5`), uncoordinated background GPU training runs or sweeps are strictly barred without an atomic reservation/claim mechanism.
>
> We verified that no host/device claim lock mechanism currently exists in `experiments/jobs/` or `scripts/`. Consequently, CPU-based readiness verification, code path repairs, full-trajectory sequence loader updates, uncertainty-aware promotion, and E01 schema adapters were completed and fully tested in this assignment. **No multi-hour GPU training sweeps were launched.** We request the coordinator schedule or assign the atomic claim mechanism before authorizing E05 training execution.

---

## 2. Supported Backend Roster Inventory

All candidate model families were audited for checkpoint availability, content hash verification, native resolution compatibility, and interface conformity.

| Model Family | Arch / Checkpoint Identifier | Native Resolution / Crop Compatibility | Conditioning Modality | Hardware / Env Requirement | Operational Readiness |
|---|---|---|---|---|---|
| **Pix2Pix** | `pix2pix_generator.pt`<br>SHA: `101a24d3...` | 256×256, 512×512<br>(aspect-ratio fit with bilinear align) | Appearance crop + 18-point 2D skeleton | Pinned conda (`pointstream`), CUDA, ~4 GB VRAM | **Ready** (full-trajectory verified, hourly checkpointing added) |
| **SPADE4Tennis (Lite)** | `spade4tennis_lite_generator.pt`<br>SHA: `d811127f...` | 256×256 (ResNet-9 SPADE) | Keyframe appearance + Dense player skeleton | Pinned conda (`pointstream`), CUDA, ~6 GB VRAM | **Ready** (restored to candidate pool, hourly checkpointing added) |
| **SPADE4Tennis (Full)** | Multi-scale UNet + LocalEnhancer | 512×512, 1080p stream crops | Keyframe appearance + Dense player skeleton | Pinned conda (`pointstream`), CUDA, ~12 GB VRAM | **Ready for Stage 1 fine-tuning** (resumes from Lite weights) |
| **Pose-ControlNet** | SD 1.5 ControlNet<br>`assets/weights/pose-controlnet/checkpoint-epoch-10/`<br>SHA: `e89b3ad7...` | 512×512 (crops placed in full frame) | OpenPose skeleton + text prompt / embedding | Pinned conda (`diffusers`), CUDA, ~16 GB VRAM | **Ready** (native delta-epoch resume verified) |
| **Animate-Anyone** | Finetuned tennis profile<br>`/home/itec/emanuele/Models/AnimateAnyone/` | 512×512 temporal sequences (16–64 frames) | Keyframe appearance + dense pose sequence | Dedicated env (`animate_anyone`), 24GB+ VRAM | **Available for E05 Stage 1** (requires cross-env dispatch) |
| **Upscale-Refine** | Conventional bicubic / Real-ESRGAN | Arbitrary scales (e.g. 256×256 $\to$ 1080p) | Low-res generated crop | CPU or lightweight GPU (<2 GB VRAM) | **Ready** (non-generative / reference baseline) |

---

## 3. Candidate Cards for E05

### Candidate Card 1: Pix2Pix (Ready Small Model)
- **Architecture**: U-Net Generator with skip connections, 70×70 PatchGAN Discriminator.
- **Native Training Recipe**:
  - Script: `scripts/train_pix2pix.py`
  - Optimizer: Adam ($\beta_1=0.5, \beta_2=0.999$, lr=$2\times 10^{-4}$).
  - Loss: $\mathcal{L}_{\text{GAN}} + \lambda_{\text{pixel}} \mathcal{L}_1$ ($\lambda_{\text{pixel}}=100.0$).
  - Hourly checkpointing (`time.time() - last_ckpt >= 3600`) and 10-minute progress logging enabled.
- **Low-Resolution Compatibility**:
  - Native crop resolution: 256×256.
  - Aspect ratio fit: player bounding box aspect ratio preserved, pad or interpolate directly to crop canvas.
- **Hyperparameter Endpoints**:
  - Learning rate: $[1\times 10^{-4}, 2\times 10^{-4}, 4\times 10^{-4}]$
  - $\lambda_{\text{pixel}}$: $[50.0, 100.0, 200.0]$
  - Batch size: 64 (or `auto`).
- **3-Stage Budget**:
  - Stage 1 (Screening): 1 GPU-hour (approx. 4 epochs on probe training view).
  - Stage 2 (Optimization): 4 GPU-hours (approx. 16 epochs with survivor hyperparameters).
  - Stage 3 (Convergence): 12 GPU-hours (approx. 48 epochs on full training split).
- **Stop Rules**:
  - Terminate if validation residual wire bytes exceed pasted reference keyframe by $>10\%$ after Stage 1.
  - Terminate if same-seed determinism or conditioning sensitivity control fails.

### Candidate Card 2: SPADE4Tennis-Lite (Spatially Adaptive Normalization)
- **Architecture**: ResNet-9 with SPADE conditioning on skeleton mask + keyframe appearance encoder.
- **Native Training Recipe**:
  - Script: `scripts/train_spade4tennis.py`
  - Optimizer: Adam ($\beta_1=0.0, \beta_2=0.999$, lr=$2\times 10^{-4}$).
  - Loss: Hinge GAN + VGG perceptual + feature matching + L1 ($\lambda_{\text{pixel}}=10.0, \lambda_{\text{vgg}}=10.0, \lambda_{\text{fm}}=10.0$).
  - Hourly checkpointing and 10-minute progress logging enabled.
- **Low-Resolution Compatibility**:
  - Native crop resolution: 256×256.
  - Dense skeleton segmentation tensor aligned directly to bounding box.
- **Hyperparameter Endpoints**:
  - Learning rate: $[1\times 10^{-4}, 2\times 10^{-4}]$
  - $\lambda_{\text{fm}}$ / $\lambda_{\text{vgg}}$ ratio: $[5.0, 10.0, 20.0]$
  - Model size: `lite` (ResNet-9) $\to$ `full` (UNet enhancer).
- **3-Stage Budget**:
  - Stage 1: 1 GPU-hour (Lite model screening).
  - Stage 2: 4 GPU-hours (Lite model convergence).
  - Stage 3: 12 GPU-hours (Full model multi-scale training).
- **Stop Rules**:
  - Prune if discriminator loss collapses ($D_{\text{loss}} < 0.05$) or generator loss diverges.
  - Prune if temporal error across consecutive frames exceeds $2\times$ pasted reference keyframe.

### Candidate Card 3: Pose-ControlNet (Diffusion Backbone)
- **Architecture**: Stable Diffusion 1.5 U-Net frozen backbone with trainable ControlNet copy conditioned on OpenPose tennis skeletons.
- **Native Training Recipe**:
  - Script: `scripts/train_controlnet.py`
  - Optimizer: AdamW (lr=$1\times 10^{-5}$, weight decay=0.01).
  - Checkpoint resume: delta-epoch directory reloading (`--controlnet-model-id`).
- **Low-Resolution Compatibility**:
  - Native crop resolution: 512×512.
- **Hyperparameter Endpoints**:
  - Guidance scale: $[1.5, 3.0, 5.0]$
  - Inference steps: $[10, 15, 20]$ (latency constrained $\le 250$ ms on client).
  - Conditioning scale: $[0.6, 0.8, 1.0]$.
- **3-Stage Budget**:
  - Stage 1: 1 GPU-hour (500 steps, step/guidance inference parameter tuning).
  - Stage 2: 4 GPU-hours (fine-tuning adapter layers on tennis dataset).
  - Stage 3: 12 GPU-hours (full temporal sequence tuning).
- **Stop Rules**:
  - Terminate if client latency exceeds 500 ms/frame (fails real-time playback requirement).
  - Terminate if residual demand fails to improve over generation-off.

---

## 4. Readiness & Code Verification Evidence

1. **Full-Trajectory Sequence Placement & Multi-Frame Verification**:
   - Resolved Audit Finding 4: `load_long_scene_clip(full_trajectory=True)` now creates `ObjectRequest` records for all visible frames across tracks, enabling continuous temporal evaluation instead of single-frame injection.
   - Tested on verified 48-frame clip `alcaraz_highlights/scene_028`: emitted exactly 96 object requests spanning all 48 frames (vs. 2 objects under legacy `full_trajectory=False`).
2. **Pose Alignment & Fail-Closed Semantics**:
   - `_augment_objects_with_pose` now properly aligns varying track bounding boxes `(bbox_h, bbox_w)` to skeleton dimensions while resizing to appearance shape as needed.
   - Verified that missing skeleton directories raise `FileNotFoundError`, and mismatched shapes raise `ValueError` without falling back to synthetic grey fills.
3. **Controls & Determinism**:
   - Normal vs. shuffled conditioning produces distinct frame hashes across frames (`test_controls_normal_vs_shuffled_conditioning_sensitivity`).
   - Same-seed repeated generation produces bit-identical frame hashes (`test_controls_same_seed_bit_identical_determinism`).
4. **Campaign Evaluator & Ranking Updates**:
   - Removed uncalibrated `lpips_vgg_uncalibrated` from `LOWER_IS_BETTER` and `RANKED_METRICS`.
   - Denominated primary ranking in `residual_bytes` / `total_bytes`, using perceptual composite only as secondary tie-breaker.
   - Implemented uncertainty-aware promotion in `promote_survivors`: candidates within a 2% relative rate threshold or 0.1 dB PSNR of the cutoff boundary are retained, preventing premature pruning due to measurement noise.
5. **E01 Schema Adapter**:
   - Created `src/runner/generation_adapter.py` providing `adapt_diagnostic_matrix_result` and `adapt_campaign_eval_result`.
   - Enforces fail-closed validation on checkpoint identity, claim eligibility (RD claim vs speed claim), control sensitivity, and uncertainty accounting.
6. **Hourly Checkpointing & Logging**:
   - Added hourly wall-clock checkpointing (`time.time() - last_ckpt >= 3600`) and 10-minute progress logging to `scripts/train_pix2pix.py` and `scripts/train_spade4tennis.py`.

---

## 5. Return Contract & Next Steps

This concludes task **E02 (Generator Readiness)**. All code and test artifacts are committed to branch `codex/eval-e02` in worktree `/tmp/pointstream-eval-e02`.
We report back to the coordinating Codex task and pause without self-dispatching E05.
