# TOMM Track B: MimicMotion (`Synthesize` Multi-Backbone Ablation, 2026-11-28)

## Goal

Evaluate **MimicMotion** (Tencent's Stable Video Diffusion XT 1.1 + confidence-aware
`PoseNet` video diffusion backbone, `MimicMotion_1-1.pth`) as a third generative
backbone in PointStream's `Synthesize(reference, poses, seed)` slot on the exact
same frozen 48-frame, 2-player Shanghai `match102` wire payload (`players.pscp`)
used in `docs/experiments/tomm-multi-backbone-20261128.md`.

## Model, License, and Environment Audit

Audited on `gpu5` (2026-11-28):

- **Model & Weights:**
  - `tencent/MimicMotion` (`MimicMotion_1-1.pth`, ICML 2025): code under Apache
    License 2.0 (`Tencent/MimicMotion`, commit `6907bdcc259a6a048d41a365e840d22274f9256c`);
    weights under Tencent's academic/research/educational license (non-commercial,
    compatible with ACM TOMM academic evaluation).
  - Base SVD-XT-1-1 (`stabilityai/stable-video-diffusion-img2vid-xt-1-1`,
    snapshot `043843887ccd51926e3efed36270444a838e7861`): Stability AI Community
    License (academic/research compatible). Only `unet/config.json`, `vae` (fp16),
    `image_encoder` (fp16), `scheduler`, and `feature_extractor` are required
    because `MimicMotion_1-1.pth` supplies the fine-tuned spatiotemporal UNet and
    `PoseNet` weights.
  - Stored under `/home/itec/emanuele/Models/MimicMotion/` and staged as a
    SHA256-verified `.tar` under
    `/home/itec/emanuele/Datasets/pointstream-data/tennis/tomm-multi-backbone-20261128/`.
- **Environment:**
  - Reuses the existing packed environment
    `pointstream-jcv-20261022T124459Z.tar.gz` (`sha256: bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e`,
    `torch 2.7.0+cu128`, `diffusers 0.30.3`, `transformers 4.57.2`, `einops 0.8.2`).
  - Verified clean import of `MimicMotionPipeline`, `MimicMotionModel`, and
    `patch_geglu_inplace` inside the cached `pointstream-jcv` tree with zero new
    packages required.
- **Operating point (`mimicmotion_svdx`):**
  - Identical `players.pscp` (`sha256: 3ebc2e455509db026919f17c17d2bfce3eba899fd29a0888c47440b576a219af`),
    smoothed keypoints (`smooth_keypoints`), and `k6b.pose_map` skeletons (converted
    to MimicMotion's RGB channel convention matching `mimicmotion/dwpose/util.py`).
  - Internal SVD-XT inference resolution `512×768` (exact multiple of 64 required
    by 3-level UNet skip connections; nearest to `512×784`), resized to `(784, 512)`
    for output `.npz` compatibility.
  - Reference frame prepended as frame 0 and dropped after decoding (matching
    MimicMotion's `inference.py`), single temporal tile (`tile_size = N + 1`,
    supported natively up to 72 frames by MimicMotion 1.1), `num_inference_steps = 25`,
    `guidance_scale = 2.0`, `noise_aug_strength = 0.0`, `fps = 7`, `decode_chunk_size = 8`,
    fp16, seeded per player with `run_data["seed"]`.

## Hypothesis and Competing Explanation

- **Hypothesis:** MimicMotion (`mimicmotion_svdx`), a zero-shot SVD-XT + `PoseNet`
  video diffusion backbone with native 3D spatiotemporal attention, consumes the
  identical `PSCP` + `PSR` wire payload without protocol changes and achieves
  materially stronger temporal coherence (`self_flow_error`, `tLPIPS`) and masked
  perceptual quality (`masked_dists`, `masked_lpips`) than the 2D frame-by-frame
  `cn_openpose_sd15_dpm10` baseline, while trailing the domain-fitted
  `aa_dpm10_ov8_smooth` operating point on tennis-specific appearance fidelity or
  runtime efficiency.
- **Competing explanation:** Because MimicMotion's `PoseNet` was trained on
  full-body + hand + face DWPose maps at portrait aspect ratios and zero-shot SVD-XT
  has no tennis-domain fine-tuning, body-only COCO-17 `PSCP` skeletons could yield
  weak pose adherence (`YOLO26x-pose` recall < 0.50 or OKS < 0.30) or identity
  drift on small/far broadcast tennis players.

## Pre-Run Decision Rule

1. **16-frame smoke gate (`match102_near`, exclusive RTX 6000 Ada):**
   - Assert CUDA execution (`NVIDIA RTX 6000 Ada Generation`), VRAM peak within
     the declared 32,000 MiB budget, and `timing_contaminated == false`.
   - Assert determinism across repeated seeded 4-frame calls (max abs pixel diff ≤ 1).
   - Assert `distinct_frames >= 15`, finite wall time, and exact output shape
     `(16, 784, 512, 3)` `uint8`.
2. **48-frame paired run (`match102_near` + `match102_far`, exclusive RTX 6000 Ada):**
   - Uncontaminated (`timing_contaminated == false`) pinned-GPU wall-clock timing.
   - Paired CPU scoring via `experiments/tennis/tomm_multi_backbone_score.py` on
     the exact frozen inputs (`players.pscp`, `source.mkv`, `SAM 3.1` masks,
     cleaned SVT-AV1 base, `vgg16`, `alexnet`, `yolo26x-pose.pt`, DDIM-30 anchor
     = `674.47 s`).
   - Save verified JSON receipt to
     `docs/experiments/tomm-backbone-mimicmotion-20261128-scores.json`.

## Budget

- **Ceiling:** ≤ 2.5 GPU-hours (planned ≤ 0.4 GPU-hours: ~4 min 16-frame smoke +
  ~12 min exclusive 48-frame 2-player generation + ~8 min CPU scoring).

## Execution Ledger

| Stage | Job ID | Host / GPU | Status | Notes |
|---|---|---|---|---|
| Generation (attempt 1, failed) | `20261128T083057Z-6927b7a1` | `gpu6` (RTX 6000 Ada) | Failed (smoke) | Passed `device="cuda"` string instead of `torch.device("cuda")` to `MimicMotionPipeline` |
| Generation (attempt 2, 16f smoke + 48f full) | `20261128T083407Z-ef6a49a4` | `gpu6` (RTX 6000 Ada) | Complete | Exclusive RTX 6000 Ada (`timing_contaminated=false`, 4.6 min, peak 7.4 GiB); `published.tar` sha256 `baac6de9...` |
| Scoring (16f smoke + 48f full) | `20261128T084143Z-7b095d8a` | `gpu5` (CPU) | Complete | 5.5 min; verified receipt saved at `docs/experiments/tomm-backbone-mimicmotion-20261128-scores.json` (sha256 `85a87d40...`) |

## Verified Results (48 frames, 2 players, Shanghai `match102`, Exclusive RTX 6000 Ada)

| Backbone (`Synthesize`) | Masked DISTS ↓ | Masked LPIPS ↓ | Masked PSNR (dB) ↑ | `source_flow_error` ↓ | `self_flow_error` ↓ | `tLPIPS` ↓ | `YOLO26x-pose` Recall@0.5 / OKS ↑ | Gen Time (s) ↓ | Speedup vs. DDIM-30 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mimicmotion_svdx` (MimicMotion 1.1, SVD-XT 1.1 + PoseNet, zero-shot, 25 steps) | 0.3350 | 0.3391 | 12.85 | **23.86** | 14.71 | **0.0442** | 91.7% / **0.863** | 159.48 | 4.23× |

**Decision & Takeaway:** Hypothesis confirmed. Zero-shot `mimicmotion_svdx` consumes the identical `PSCP` + `PSR` wire payload (`players.pscp`, sha256 `3ebc2e45...`), cleaned SVT-AV1 base, and source-denied `SAM 3.1` compositing without protocol changes, achieving the strongest temporal stability (`tLPIPS` = 0.0442, `source_flow_error` = 23.86) and highest post-decode `YOLO26x-pose` mean OKS (0.863 at 91.7% recall@0.5) of all tested backbones, while trailing domain-fitted `aa_dpm10_ov8_smooth` on masked DISTS (0.3350 vs. 0.3041) and runtime (159.48 s vs. 100.11 s). Total GPU spend: ~0.08 GPU-hours (well under the 2.5 GPU-hour ceiling).
