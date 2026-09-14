# E02 — Generator Readiness & Candidate Roster Report

> **SUPERSEDED BY E02R (14 September 2026)**:
> This document represents the historical E02 implementation report. All readiness
> claims, candidate rosters, and candidate cards have been re-evaluated and superseded
> by **E02R** (`docs/workflow/session/evaluation-campaign/tasks/02r-readiness-evidence.md`).
> Concrete real-backend evidence is recorded in immutable run artifacts under
> `outputs/evaluation-campaign/e02r/`. Ready labels have been revised to distinguish
> **Validated** (Pix2Pix), **Loaded & Code-Ready** (SPADE4Tennis ResNet-9), and
> **Available / Deferred** (ControlNet, Animate-Anyone).

**Task**: E02 Generator Readiness (Superseded by E02R)  
**Area**: Generation (`docs/areas/generation.md`)  
**Worktree**: `/home/itec/emanuele/worktrees/pointstream-eval-e02r`  
**Branch**: `codex/eval-e02r`  
**Date**: 2026-09-14  

---

## 1. Prerequisite Notice for Coordinator

> [!IMPORTANT]
> **Atomic Host/Device Claim Policy (Resolved by R0)**:
> Atomic host/device claim mechanism has been implemented by R0 (`codex/eval-r0`, commit `d39893c9`) in `experiments/jobs/claims.py` and integrated into `experiments/jobs/monitor.py`.
> Bounded real-backend probe for E02R was executed on free GPU 1 (`CUDA_VISIBLE_DEVICES=1`), strictly avoiding GPU 0 where active processes were detected.

---

## 2. Supported Backend Roster Inventory (Audited for E02R)

All candidate model families were audited for checkpoint availability, content hash verification, native resolution compatibility, and interface conformity.

| Model Family | Arch / Checkpoint Identifier | Native Resolution / Crop Compatibility | Conditioning Modality | Hardware / Env Requirement | Operational Readiness (E02R Status) |
|---|---|---|---|---|---|
| **Pix2Pix** | `pix2pix_generator.pt`<br>SHA: `101a24d3...` | 256×256, 512×512<br>(aspect-ratio fit with bilinear align) | Appearance crop + 18-point 2D skeleton | Pinned conda (`pointstream`), CUDA, ~4 GB VRAM | **Validated** (Real-backend 16-frame probe completed, full-trajectory sequence placement verified, atomic intra-epoch resume proven) |
| **SPADE4Tennis (Lite)** | `spade4tennis_lite_generator.pt`<br>SHA: `d811127f...` | 256×256 (ResNet-9 SPADE) | Keyframe appearance + Dense player skeleton | Pinned conda (`pointstream`), CUDA, ~6 GB VRAM | **Loaded & Code-Ready** (Pretrained load order fixed; atomic intra-epoch resume proven) |
| **SPADE4Tennis (Full)** | ResNet-9 generator (same backbone as Lite; multi-scale UNet is future work) | 256×256, 512×512 | Keyframe appearance + Dense player skeleton | Pinned conda (`pointstream`), CUDA, ~6 GB VRAM | **Loaded & Code-Ready** (Shares ResNet-9 architecture; discriminator count adjusted; honest architectural declaration) |
| **Pose-ControlNet** | SD 1.5 ControlNet<br>`assets/weights/pose-controlnet/checkpoint-epoch-10/`<br>SHA: `e89b3ad7...` | 512×512 (crops placed in full frame) | OpenPose skeleton + text prompt / embedding | Dedicated env (`diffusers`), CUDA, ~16 GB VRAM | **Available / Deferred** (Requires cross-env dispatch; sequence inference unprobed in pinned env) |
| **Animate-Anyone** | Finetuned tennis profile<br>`/home/itec/emanuele/Models/AnimateAnyone/` | 512×512 temporal sequences (16–64 frames) | Keyframe appearance + dense pose sequence | Dedicated env (`animate_anyone`), 24GB+ VRAM | **Available / Deferred** (Requires cross-env dispatch; preserved on offline quality frontier) |
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
- **Staged Aggregate Budget (Pool-level Allocation)**:
  - Stage 1 (Screening): 1 GPU-hour aggregate across candidate hyperparameters.
  - Stage 2 (Optimization): 4 GPU-hours aggregate on survivor rungs.
  - Stage 3 (Convergence): 12 GPU-hours aggregate for final model convergence.
- **Stop Rules & Frontier Retention**:
  - Terminate candidate if same-seed determinism or conditioning sensitivity control fails.
  - For residual-off evaluation, rank strictly on total wire rate at matched fidelity/budget; residual bytes alone cannot rank.
  - Slow models remain on the offline quality frontier; latency failure alone does not terminate a family.

### Candidate Card 2: SPADE4Tennis (ResNet-9 Spatially Adaptive Normalization)
- **Architecture**: ResNet-9 generator (`SPADEResNet9Generator`) with SPADE conditioning on player skeleton mask + keyframe appearance encoder. Both `lite` and `full` tiers instantiate this ResNet-9 generator (discriminator count varies: 2 vs 3). Multi-scale UNet / LocalEnhancer is deferred future work.
- **Native Training Recipe**:
  - Script: `scripts/train_spade4tennis.py`
  - Optimizer: Adam ($\beta_1=0.0, \beta_2=0.999$, lr=$2\times 10^{-4}$).
  - Pretrained initialization order verified: `weights_init_normal` called before loading `args.pretrained_g`.
  - Loss: Hinge GAN + VGG perceptual + feature matching + L1 ($\lambda_{\text{pixel}}=10.0, \lambda_{\text{vgg}}=10.0, \lambda_{\text{fm}}=10.0$).
  - Intra-epoch hourly checkpointing, atomic saving, and partial-epoch skip resume verified.
- **Low-Resolution Compatibility**:
  - Native crop resolution: 256×256.
  - Dense skeleton segmentation tensor aligned directly to bounding box.
- **Hyperparameter Endpoints**:
  - Learning rate: $[1\times 10^{-4}, 2\times 10^{-4}]$
  - $\lambda_{\text{fm}}$ / $\lambda_{\text{vgg}}$ ratio: $[5.0, 10.0, 20.0]$
  - Discriminator count: 2 (`lite`) vs 3 (`full`).
- **Staged Aggregate Budget**:
  - Stage 1: 1 GPU-hour aggregate screening.
  - Stage 2: 4 GPU-hours aggregate optimization.
  - Stage 3: 12 GPU-hours aggregate convergence.
- **Stop Rules & Frontier Retention**:
  - Prune configuration if discriminator loss collapses ($D_{\text{loss}} < 0.05$) or generator loss diverges.
  - Preserve slow models on offline quality frontier; do not discard solely on decode latency.

### Candidate Card 3: Pose-ControlNet (Diffusion Backbone — Available / Deferred)
- **Architecture**: Stable Diffusion 1.5 U-Net frozen backbone with trainable ControlNet copy conditioned on OpenPose tennis skeletons.
- **Environment**: Dedicated `diffusers` conda environment; cross-env dispatch required.
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
- **Staged Aggregate Budget**:
  - Stage 1: 1 GPU-hour aggregate (inference step / guidance parameter sweep).
  - Stage 2: 4 GPU-hours aggregate (adapter tuning).
  - Stage 3: 12 GPU-hours aggregate (full sequence tuning).
- **Frontier Rules**:
  - Preserved on offline quality frontier; latency exceeding real-time does not eliminate candidate from high-quality offline regime.

### Candidate Card 4: Animate-Anyone (Temporal Sequence Diffusion — Available / Deferred)
- **Architecture**: ReferenceUNet appearance encoder + PoseNet + Spatial-Temporal Denoising UNet with temporal attention.
- **Environment**: Dedicated `animate_anyone` conda environment (`/home/itec/emanuele/Models/AnimateAnyone/`).
- **Conditioning**: Keyframe appearance + continuous 18-point skeleton sequence (16–64 frames).
- **Readiness Status**: Available for cross-env dispatch. Sequence inference unprobed in current pinned env; preserved on offline quality frontier.

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
