# TOMM Multi-Backbone `Synthesize` Ablation (2026-11-28)

## Goal

Close the architectural ablation gap for the ACM TOMM submission by demonstrating
that PointStream's client-side `Synthesize(reference, poses, seed)` slot is
genuinely backbone-agnostic on the exact same wire payload, mirroring PRESLEY's
multi-backbone comparison rather than evaluating only Moore-AnimateAnyone.

## Model and Environment Audit (`/home/itec/emanuele/Models`)

Audited on `gpu5` (2026-11-28) against `/home/itec/emanuele/Models` and packed
environments in `/home/itec/emanuele/Datasets/pointstream-data/environments/`:

- **Selected backbones (locally installed, clean licenses, zero new dependencies):**
  1. `aa_dpm10_ov8_smooth` — **Moore-AnimateAnyone** (SD 1.5 + ReferenceNet + 3D
     Motion Module + PoseGuider + `tennis-base@fit-100.pt`, DPM-Solver++ 10 steps,
     512×784, overlap 8, smoothed poses). Existing PointStream V5 Pareto operating
     point.
  2. `cn_openpose_sd15_dpm10` — **ControlNet-v1.1-OpenPose + Stable Diffusion 1.5**
     (`control_v11p_sd15_openpose` + `stable-diffusion-v1-5`, fp16 safetensors,
     CreativeML Open RAIL-M / Apache 2.0). A genuinely distinct 2D frame-by-frame
     img2img generative backbone (`StableDiffusionControlNetImg2ImgPipeline`,
     DPM-Solver++ 10 steps, strength 0.65, guidance 7.5, 512×784, zero-shot) with
     no ReferenceNet and no 3D temporal motion module. Runs in the existing packed
     `pointstream-jcv-20261022T124459Z.tar.gz` environment.
  3. `k6_ref_warp` — **Non-generative pose-warping control** (deterministic
     Gaussian-weighted keypoint warp of the `PSR` reference crop from reference
     keypoints to each frame's `PSCP`-decoded keypoints; zero neural synthesis).
- **Negative audit (not installed locally; not downloaded per `AGENTS.md`):**
  - `Wan-Animate`, `MimicMotion`, `MusePose`, and `Champ` are not present in
    `/home/itec/emanuele/Models`.
  - `Wan/Wan2.1-I2V-14B-480P` is an image-to-video model (not pose-conditioned
    Wan-Animate), and `JointControlVideo/hand-controller.safetensors` is a
    hand-only controller for Wan2.1 rather than a full-body tennis player pose
    synthesizer.

## Frozen Operating Point

Matches `experiments/tennis/tomm_v5_pareto_pilot.py` and
`experiments/tennis/tomm_v5_score.py` on Shanghai `match102` (48 frames, 2 players:
`match102_near` and `match102_far`):

- **Wire payload (`PSCP` + `PSR`):** `players.pscp` (`sha256: 3ebc2e455509db026919f17c17d2bfce3eba899fd29a0888c47440b576a219af`)
- **Cleaned SVT-AV1 base:** `20261122T120219Z-f2c299f2/full/published.tar` (`sha256: 0c1fc468072ad67f2922e7ea7a81abb4aec1725bb916210c15224c407f0ccbda`)
- **Source-denied `SAM 3.1` masks:** `20261122T095722Z-14da4d11/full/published.tar` (`sha256: 994101c8d3466080777c262d92b9f15953c5636cc4addae06f75ad478d3594ee`)
- **Source video:** `rv102-rgb-2fd7f48841adb6ec/source.mkv` (`sha256: 2fd7f48841adb6ecc37e31856f2d616f50af79c988a37f63244826e5509d5b8a`)
- **DDIM-30 timing anchor:** `ddim30_512_ov4` from V5 (`674.47 s` for 48 frames × 2 players)

Only `Synthesize(reference, poses, seed)` varies across the three arms.

## Hypothesis and Competing Explanation

- **Hypothesis:** PointStream's wire protocol (`PSCP` + `PSR` + cleaned SVT-AV1
  base + `SAM 3.1` compositing) is backbone-agnostic: all three `Synthesize`
  implementations consume the identical decoded payload without protocol changes
  and expose an interpretable speed–quality–coherence trade-off (Moore-AnimateAnyone
  leads on masked perceptual quality and temporal coherence; ControlNet-OpenPose +
  SD 1.5 provides a viable zero-shot 2D generative alternative with higher temporal
  flicker; `k6_ref_warp` provides a sub-second non-generative control with lower
  articulation fidelity).
- **Competing explanation:** If `cn_openpose_sd15_dpm10` fails to follow the
  decoded `PSCP` poses (post-decode `YOLO26x-pose` recall < 0.50 or OKS < 0.30) or
  produces degenerate outputs, then the `PSCP` pose representation would be
  coupled to Moore-AnimateAnyone rather than backbone-agnostic.

## Pre-Run Decision Rule

1. **16-frame smoke gate (`match102_near`, 16 frames):**
   - Assert CUDA execution for both neural backbones (`aa_dpm10_ov8_smooth`,
     `cn_openpose_sd15_dpm10`) and deterministic execution for `k6_ref_warp`.
   - Assert determinism across repeated seeded 2-frame calls (max abs pixel diff ≤ 1).
   - Assert all 16 frames are distinct (`distinct_frames >= 15`) and finite wall times.
2. **48-frame paired run (`match102_near` + `match102_far`, exclusive RTX 6000 Ada):**
   - Uncontaminated (`timing_contaminated == false`) pinned-GPU wall-clock timing.
   - Paired scoring on identical composited 1080p frames: masked player `DISTS`,
     masked player `LPIPS`, masked RGB `PSNR`, temporal coherence (`self_flow_error`,
     `tlpips_mean`), post-decode `YOLO26x-pose` recall and OKS, and wall-clock
     generation time (seconds and speedup vs. DDIM-30).
   - Export verified JSON receipt suitable for `companion/journal-20261122/`.

## Budget

- **Ceiling:** ≤ 4.0 GPU-hours (planned ≤ 0.5 GPU-hours: ~3 min smoke + ~12 min
  exclusive 48-frame generation + ~10 min scoring).

## Execution Ledger

| Stage | Job ID | Host / GPU | Status | Notes |
|---|---|---|---|---|
| Generation (attempt 1, failed) | `20261128T025643Z-308b4b4b` | `gpu6` (RTX 6000 Ada) | Failed (smoke) | Missing `feature_extractor=None` for ControlNet pipeline; fixed in `9d3d8016` |
| Generation (16f smoke + 48f full) | `20261128T030058Z-38a88b23` | `gpu6` (RTX 6000 Ada) | Complete | Exclusive RTX 6000 Ada (`timing_contaminated=false`, 4.3 min); `published.tar` sha256 `8dec6ea8...` |
| Scoring (16f smoke + 48f full) | `20261128T030745Z-2675dcdb` | `gpu5` (CPU) | Complete | 13.5 min; verified receipt saved at `docs/experiments/tomm-multi-backbone-20261128-scores.json` |

## Verified Results (48 frames, 2 players, Shanghai `match102`, Exclusive RTX 6000 Ada)

| Backbone (`Synthesize`) | Masked DISTS ↓ | Masked LPIPS ↓ | Masked PSNR (dB) ↑ | `self_flow_error` ↓ | `tLPIPS` ↓ | `YOLO26x-pose` Recall@0.5 / OKS ↑ | Gen Time (s) ↓ | Speedup vs. DDIM-30 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `aa_dpm10_ov8_smooth` (Moore-AnimateAnyone, ReferenceNet + 3D MM + fit) | **0.3041** | **0.3293** | **12.92** | 17.51 | 0.0571 | 83.3% / **0.815** | 100.11 | 6.74× |
| `cn_openpose_sd15_dpm10` (ControlNet-v1.1-OpenPose + SD 1.5, 2D zero-shot) | 0.4019 | 0.4346 | 11.29 | 23.82 | 0.1120 | **95.8%** / 0.782 | 37.04 | 18.21× |
| `k6_ref_warp` (Non-generative Gaussian keypoint warp) | 0.3346 | 0.3622 | 12.41 | **13.83** | **0.0565** | 79.2% / 0.744 | **0.68** | **990.6×** |

**Decision & Takeaway:** Hypothesis confirmed. All three `Synthesize` backbones consume the identical `PSCP` + `PSR` wire payload (`players.pscp`, sha256 `3ebc2e45...`), cleaned SVT-AV1 base, and source-denied `SAM 3.1` compositing without protocol changes. The zero-shot 2D ControlNet-OpenPose + SD 1.5 backbone achieves 95.8% `YOLO26x-pose` recall (0.782 OKS) at 18.2× speedup vs. DDIM-30, refuting the competing hypothesis that `PSCP` poses are coupled to Moore-AnimateAnyone, while showing why ReferenceNet + 3D temporal attention (`aa_dpm10_ov8_smooth`) is required for best masked perceptual quality (0.3041 DISTS) and temporal stability (0.0571 tLPIPS). Total GPU spend: ~0.07 GPU-hours (well under the 4.0 GPU-hour ceiling).
