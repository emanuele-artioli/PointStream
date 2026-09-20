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
| **Current** | WebP appearance crops (pasted first reference, generation-OFF) | Shipped operational baseline ($F \approx 8\text{--}12\text{ kB}$ on 192f; pix2pix failed with -0.6 to -1.0 dB PSNR deficit and +400 kB pose wire overhead) |
| **Oracle** | Lossless/high-quality ground-truth actor crops pasted directly | Upper bound: exact source actor pixels inside bbox ($PSNR_{\text{actor}} = \infty$), compressed with optimal intra codec |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Module Bytes ($F$) | Actor PSNR (dB) | Whole-Codec PSNR-Y (dB) | Whole VMAF |
|---|---|---|---|---|
| **Null** (Repeat first crop) | 3,510 B | 18.2 dB | 21.5 dB | 12.0 |
| **Current** (WebP crops, gen-OFF) | 8,954 B | 27.8 dB | 26.2 dB | 29.8 |
| **Oracle** (Lossless pasted crops) | ~25,000 B | $\infty$ | 32.5 dB | 65.0 |

- **Headroom (Oracle - Current)**: Quality headroom is large (+6.3 dB PSNR-Y), but byte budget must stay below 12 kB. Generative models (pix2pix, ControlNet) must deliver quality improvements without exceeding wire pose overhead.

### Long Horizon (192 frames @ 24 fps, `alcaraz_highlights/scene_000`)

| Arm | Module Bytes ($F$) | Actor PSNR (dB) | Whole-Codec PSNR-Y (dB) | Whole VMAF |
|---|---|---|---|---|
| **Null** (Repeat first crop) | 3,510 B | 17.5 dB | 23.4 dB | 0.0 |
| **Current** (WebP crops C1) | 8,954 B | 28.1 dB | 27.2 dB | 29.8 |
| **Oracle** (High-Q WebP crops) | 12,118 B | 33.4 dB | 32.3 dB | 72.2 |

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - Current generative models (pix2pix) failed acceptance because pose overhead (+400 kB) outweighed quality gains.
  - WebP pasted-reference remains the conservative frozen baseline.
  - Any new generative model must beat WebP pasted-reference at matched wire rate ($F + M \le 12\text{ kB}$).
- **Next Action**: Maintain generation-OFF as operational default; evaluate compact pose conditioning before any new generative training.

