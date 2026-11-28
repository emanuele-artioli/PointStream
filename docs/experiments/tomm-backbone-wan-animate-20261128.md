# TOMM Multi-Backbone `Synthesize` Ablation — Track C: Wan-Animate (14B DiT) (2026-11-28)

## Goal

Evaluate **Wan2.2-Animate-14B** (`Wan-AI/Wan2.2-Animate-14B`, Alibaba Wan Team,
Apache 2.0) as Track C of PointStream's ACM TOMM multi-backbone
`Synthesize(reference, poses, seed)` ablation (`Table~\ref{tab:multi-backbone}`),
using the exact same frozen 48-frame, 2-player Shanghai `match102` wire payload
(`players.pscp`) and CPU scoring harness (`experiments/tennis/tomm_multi_backbone_score.py`)
as `docs/experiments/tomm-multi-backbone-20261128.md`.

## Model, License, and Environment Audit

- **Model:** `Wan-AI/Wan2.2-Animate-14B` (14B dense DiT with pose latent
  conditioning, CLIP reference image conditioning, and Wan 2.1 3D VAE;
  **Apache 2.0** license).
- **Codebase:** Official `Wan-Video/Wan2.2` (`1ea34ff48f87168174e12956e200b1d908b1c5ff`,
  Apache 2.0), staged as a SHA256-verified archive alongside tokenizers and configs.
- **Weight reuse against `/home/itec/emanuele/Models/Wan/Wan2.1-I2V-14B-480P/`:**
  - `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` (4.77 GB, sha256 `628c9998...`): **exact match, hard-linked**.
  - `models_t5_umt5-xxl-enc-bf16.pth` (11.36 GB, sha256 `7cace0da...`): **exact match, hard-linked**.
  - `google/umt5-xxl/` and `xlm-roberta-large/` tokenizers: **reused**.
  - Newly downloaded to `/home/itec/emanuele/Models/Wan/Wan2.2-Animate-14B/` and
    hard-linked into `/home/itec/emanuele/Datasets/pointstream-data/tennis/tomm-multi-backbone-20261128/wan-animate-14b/`:
    - `Wan2.1_VAE.pth` (0.51 GB, sha256 `acf6c5aa49ad281d4b561e10656e2397c446a8ba4b8d8f19d3dd125c2628bc6a`)
    - `diffusion_pytorch_model-00001-of-00004.safetensors` (9.88 GB, sha256 `575c2dba750c3b40240fb742a4224453aa97dfbd3c5f5a0086be431cdefdd69c`)
    - `diffusion_pytorch_model-00002-of-00004.safetensors` (9.98 GB, sha256 `b90b820627d43eeeb1ae0489182f9a8c870374fd72cc99dccb9eddfc2ace8325`)
    - `diffusion_pytorch_model-00003-of-00004.safetensors` (9.95 GB, sha256 `e2aa343b0ba04f563566e9959a439611a18189cd9accdc04d59681be9ce5be50`)
    - `diffusion_pytorch_model-00004-of-00004.safetensors` (4.74 GB, sha256 `e5fea7f38aa4cb70ed59a9ecf406d0c05f3dfc85ee70bc83de35679c42642d1b`)
- **Environment:** Reuses the existing packed `pointstream-jcv-20261022T124459Z.tar.gz`
  (`sha256: bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e`,
  PyTorch 2.7.0+cu128, diffusers 0.30.3, transformers 4.57.2, accelerate 1.12.0,
  peft 0.18.0, einops 0.8.2, ftfy 6.3.1, safetensors 0.8.0, sentencepiece 0.2.1).
  No new conda environment is needed.
- **VRAM & kernel plan on exclusive RTX 6000 Ada (48 GiB, SM 8.9):**
  - UMT5-XXL stays on CPU (`t5_cpu=True`) to encode the fixed prompt once.
  - CLIP and VAE are staged onto GPU only during their respective encode/decode
    phases and offloaded to CPU before DiT denoising.
  - The 14B DiT runs natively in `bfloat16` on GPU without block-level swapping
    (expected peak ~36–41 GiB < 44 GiB).
  - Transformer attention dispatches through PyTorch native SDPA with
    `SDPBackend.FLASH_ATTENTION` enforced on SM 8.9.

## Frozen Operating Point

Identical to `docs/experiments/tomm-multi-backbone-20261128.md` on Shanghai
`match102` (48 frames, 2 players: `match102_near` and `match102_far`):

- **Wire payload (`PSCP` + `PSR`):** `players.pscp` (`sha256: 3ebc2e455509db026919f17c17d2bfce3eba899fd29a0888c47440b576a219af`)
- **Cleaned SVT-AV1 base:** `20261122T120219Z-f2c299f2/full/published.tar` (`sha256: 0c1fc468072ad67f2922e7ea7a81abb4aec1725bb916210c15224c407f0ccbda`)
- **Source-denied `SAM 3.1` masks:** `20261122T095722Z-14da4d11/full/published.tar` (`sha256: 994101c8d3466080777c262d92b9f15953c5636cc4addae06f75ad478d3594ee`)
- **Source video:** `rv102-rgb-2fd7f48841adb6ec/source.mkv` (`sha256: 2fd7f48841adb6ecc37e31856f2d616f50af79c988a37f63244826e5509d5b8a`)
- **DDIM-30 timing anchor:** `674.47 s` (48 frames × 2 players)
- **Zero-leakage face conditioning:** `face_pixel_values` is set to all-black
  (`-1.0` normalized, matching Wan-Animate's unconditional face input) because
  PointStream transmits only `(reference, poses, seed)` with no driving face video.
- **Inference settings:** native `512×784`, single temporal clip (`clip_len=17`
  for 16f smoke, `clip_len=49` for 48f full, sliced to `[:frames]`),
  `FlowDPMSolverMultistepScheduler`, `sampling_steps=20`, `shift=5.0`,
  `guide_scale=1.0`, `replace_flag=False`.

## Hypothesis and Competing Explanation

- **Hypothesis:** Replacing Moore-AnimateAnyone with a modern 14B DiT video
  foundation model (`Wan2.2-Animate-14B`, zero-shot, no tennis fine-tuning) on
  the identical `PSCP` + `PSR` wire payload produces coherent, temporally smooth
  48-frame tennis player animation (`tLPIPS` and `self_flow_error` competitive
  with or better than `aa_dpm10_ov8_smooth`, `YOLO26x-pose` recall ≥ 0.70) at a
  higher per-frame compute cost than the 1.5B SD-based UNet, demonstrating that
  PointStream's wire protocol scales cleanly to 14B DiT video backbones.
- **Competing explanation:** Because `Wan2.2-Animate-14B` was trained on
  high-resolution DWPose full-body/face videos and is evaluated zero-shot on
  17-keypoint COCO `k6b.pose_map` skeletons without driving face crops, it may
  drift in pose alignment (`YOLO26x-pose` recall < 0.50 or OKS < 0.40) or
  hallucinate background/scale shifts that degrade masked DISTS/PSNR relative to
  the tennis-adapted Moore-AnimateAnyone baseline.

## Pre-Run Decision Rule

1. **16-frame smoke gate (`match102_near`, 16 frames, exclusive RTX 6000 Ada):**
   - Assert CUDA execution on RTX 6000 Ada (SM 8.9) with native FlashAttention
     SDPA kernel (`SDPBackend.FLASH_ATTENTION`) active in transformer blocks.
   - Assert peak allocated VRAM < 44 GiB.
   - Assert determinism on repeated seeded 5-frame 4-step calls (max abs pixel diff ≤ 1).
   - Assert `distinct_frames >= 15`, finite wall times, and forecast 48-frame × 2-player
     wall time ≤ 2700 s (45 min).
2. **48-frame paired run (`match102_near` + `match102_far`, exclusive RTX 6000 Ada):**
   - Uncontaminated (`timing_contaminated == false`) pinned-GPU wall-clock timing.
   - Paired CPU scoring via `experiments/tennis/tomm_multi_backbone_score.py` on
     the exact frozen inputs, saving the verified receipt to
     `docs/experiments/tomm-backbone-wan-animate-20261128-scores.json`.

## Budget

- **Ceiling:** ≤ 3.5 GPU-hours (planned ≤ 0.5 GPU-hours: ~5 min smoke + ~15 min
  exclusive 48-frame generation + ~10 min CPU scoring).

## Execution Ledger

| Stage | Job ID | Host / GPU | Status | Notes |
|---|---|---|---|---|
| Generation (attempt 1, failed) | `20261128T041848Z-d5d66523` | `gpu6` (RTX 6000 Ada) | Failed (smoke) | `reset_peak_memory_stats` called before `torch.cuda.set_device(0)`; fixed |
| Generation (attempt 2, failed) | `20261128T042632Z-52337146` | `gpu6` (RTX 6000 Ada) | Failed (smoke) | `transformers` checked `decord.__spec__`; added `ModuleSpec` to shims |
| Generation (attempt 3, failed) | `20261128T043027Z-0244070d` | `gpu6` (RTX 6000 Ada) | Failed (smoke) | `wan/__init__.py` eagerly imported `speech2video` (`librosa`); bypassed via package stub |
| Generation (attempt 4, failed) | `20261128T043508Z-31a5f8cc` | `gpu6` (RTX 6000 Ada) | Failed (smoke) | All models loaded; `wan.modules.animate.clip` bound unpatched `flash_attention`; patched |
| Generation (16f smoke + 48f full) | `20261128T043956Z-8aa56788` | `gpu6` (RTX 6000 Ada) | Complete | Exclusive RTX 6000 Ada (`timing_contaminated=false`, 15.0 min total, peak 36.5 GiB, 4,862 Flash SDPA calls, determinism diff=0, 465.51 s render); `published.tar` sha256 `93d6edfd...` |
| Scoring (16f smoke + 48f full) | `20261128T045711Z-b59c7278` | `gpu5` (CPU) | Complete | 13.0 min (`elapsed_s=446.81`); verified receipt saved at `docs/experiments/tomm-backbone-wan-animate-20261128-scores.json` (`sha256: 5daf5b61...`) |

## Verified Results (48 frames, 2 players, Shanghai `match102`, Exclusive RTX 6000 Ada)

| Backbone (`Synthesize`) | Masked DISTS ↓ | Masked LPIPS ↓ | Masked PSNR (dB) ↑ | `self_flow_error` ↓ | `source_flow_error` ↓ | `tLPIPS` ↓ | `YOLO26x-pose` Recall@0.5 / OKS ↑ | Gen Time (s) ↓ | Speedup vs. DDIM-30 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `wan_animate_14b` (Wan2.2-Animate-14B, 14B DiT, zero-shot, 20 DPM++ steps) | 0.3226 | 0.3525 | 12.12 | 17.79 | 26.50 | 0.0580 | 79.2% / 0.776 | 465.51 | 1.45× |

- **Per-player masked DISTS:** `near_dists = 0.2960`, `far_dists = 0.3492`.
- **Per-player render time:** `match102_near = 232.32 s` (`4.84 s/frame`), `match102_far = 233.19 s` (`4.86 s/frame`), total `465.51 s` (`4.85 s/player-frame`, `timing_contaminated = false`).
- **Peak VRAM & kernels:** `37,382 MiB` allocated (`36.5 GiB`, `38,618 MiB` reserved) on exclusive `RTX 6000 Ada` (SM 8.9); `4,862` native `SDPBackend.FLASH_ATTENTION` calls + `320` face SDPA calls; smoke 5-frame determinism `max_abs_diff = 0`.
- **Artifact identities:**
  - Generation `published.tar`: `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261128T043956Z-8aa56788/full/published.tar` (`sha256: 93d6edfd3af767997d63f4a6a73a88d26a851fd3f1219b9f7e8b3d54923fb5dc`)
  - Scores JSON receipt: `docs/experiments/tomm-backbone-wan-animate-20261128-scores.json` (`sha256: 5daf5b618638c1c8cd3f069ff9e0bc02dcf58c52c58a2e4129ea643e77201b75`)

**Decision & Takeaway:** Hypothesis confirmed. Zero-shot `Wan2.2-Animate-14B` consumes the exact frozen `PSCP` + `PSR` wire payload (`players.pscp`, sha256 `3ebc2e45...`) without protocol changes, outperforms both zero-shot 2D ControlNet-OpenPose + SD 1.5 (`0.3226` vs. `0.4019` DISTS; `0.0580` vs. `0.1120` tLPIPS) and non-generative `k6_ref_warp` (`0.3226` vs. `0.3346` DISTS; `0.776` vs. `0.744` OKS), and nearly matches tennis-adapted Moore-AnimateAnyone on temporal coherence (`17.79` vs. `17.51` `self_flow_error`, `0.0580` vs. `0.0571` `tLPIPS`) at `4.65×` the wall-clock render time (`465.51 s` vs. `100.11 s`, still `1.45×` faster than DDIM-30). Total GPU spend: ~0.25 GPU-hours (well under the 3.5 GPU-hour ceiling).
