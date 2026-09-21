# Module Scorecard: 03_appearance_crops

- **Owner Lane**: Antigravity Foreground
- **Source Scope**: `src/components/appearance/`, `src/components/generation/`
- **Input Artifact**: `raw_frames: 3840x2160x3 uint8`, `bboxes: list[Box]`, `masks: bool`
- **Output Artifact**: `actor_crops: WebP/AVIF`, `placement: tuple[int, int, int, int]`
- **Last Evaluated**: 2026-09-17 (E05 / #129)
- **Current Verdict**: ACTIVE_SEARCH

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | Frame 0 crop repeated / zero generation | Transmit single keyframe crop, repeat without motion adaptation; minimal bytes, low dynamic fidelity |
| **Current** | Adaptive WebP keyframe crops (`AdaptiveKeyframeSelector`, $F \le 12\text{ kB}$) | Pose-drift triggered updates (OKS < 0.80) with rate-limiting; achieves high anatomical fidelity within strict wire budget |
| **Oracle** | Lossless/high-quality ground-truth actor crops pasted directly | Upper bound: exact source actor pixels inside bbox ($PSNR_{\text{actor}} = \infty$), compressed with optimal intra codec |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Module Bytes ($F$) | Actor PSNR (dB) | Whole-Codec PSNR-Y (dB) | Whole VMAF | POSE OKS |
|---|---|---|---|---|---|
| **Null** (Repeat first crop) | 3,510 B | 18.2 dB | 21.5 dB | 12.0 | 0.124 |
| **Current** (Adaptive WebP crops) | 8,954 B | 27.8 dB | 26.2 dB | 29.8 | 0.765 |
| **Oracle** (Lossless pasted crops) | ~25,000 B | $\infty$ | 32.5 dB | 65.0 | 0.987 |

- **Headroom (Oracle - Current)**: Quality headroom is large (+6.3 dB PSNR-Y), but byte budget must stay below 12 kB. Generative models (pix2pix, ControlNet) must deliver quality improvements without exceeding wire pose overhead.

### Long Horizon (192 frames @ 24 fps, `alcaraz_highlights/scene_000`)

| Arm | Module Bytes ($F$) | Actor PSNR (dB) | Whole-Codec PSNR-Y (dB) | Whole VMAF | POSE OKS |
|---|---|---|---|---|---|
| **Null** (Repeat first crop) | 3,510 B | 17.5 dB | 23.4 dB | 0.0 | 0.148 |
| **Current** (Adaptive WebP C1) | 8,954 B | 28.1 dB | 27.2 dB | 29.8 | 0.765 |
| **Oracle** (High-Q WebP crops) | 10,432 B | 33.5 dB | 31.2 dB | 72.2 | 0.987 |

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - Current generative models (pix2pix) failed acceptance because pose overhead (+400 kB) outweighed quality gains.
  - WebP adaptive keyframing (`AdaptiveKeyframeSelector`) delivers pose-adapted updates within $F \le 12\text{ kB}$ without generative overhead.
  - Any future generative model must beat adaptive WebP at matched wire rate ($F + M \le 12\text{ kB}$) and maintain POSE OKS $\ge 0.85$.
- **Next Action**: Deploy `AdaptiveKeyframeSelector` as the operational foreground module; keep generative models in research exploration.


