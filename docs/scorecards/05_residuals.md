# Module Scorecard: 05_residuals

- **Owner Lane**: Codec / Codex
- **Source Scope**: `src/pipeline/residual/`
- **Input Artifact**: `source_frames: uint8`, `predicted_composite: uint8`
- **Output Artifact**: `residual_stream: one VVC video`
- **Last Evaluated**: 2026-09-22 (measured 48-frame federer scene 007)
- **Current Verdict**: ACTIVE_SEARCH

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | Residual OFF ($R=0$) | Base PointStream composite delivered to client; 0 residual bits |
| **Current** | One VVC video of the C1 error, preset medium, QP 40, plus foreground-only and background-only ablations | Inter-frame residual. Preset `faster` emits an empty file on this signal. |
| **Oracle** | Transparent VVC intra coded residual at matched visual quality | Maximum fidelity restoration |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Residual bytes | Total bytes | Overall PSNR | FG / BG PSNR |
|---|---|---|---|---|
| **Before, C2** per-frame actor WebP | 401,018 B | 930,936 B | 20.57 dB | 35.89 / 20.56 dB |
| **Before, C3** per-frame background WebP | 14,094,766 B | 14,624,684 B | 33.40 dB | 35.89 / 33.39 dB |
| **After, full video** VVC medium QP 40 | 293,747 B | 759,913 B | 31.32 dB | 34.03 / 31.32 dB |
| **After, foreground-only video** | 5,249 B | 471,415 B | 20.51 dB | 36.81 / 20.50 dB |
| **After, background-only video** | 293,597 B | 759,763 B | 31.33 dB | 36.70 / 31.32 dB |

The full-video arm has weighted PSNR 33.21 dB, above VVC QP 46's 24.59 dB
and AV1 QP 54's 30.07 dB, but it loses on rate. The background-only arm is
35.08 dB weighted; the foreground-only arm is 31.91 dB weighted.

- **Short Horizon Diagnosis**: The per-frame background WebP was the 14 MB term. One background video does the same job at 294 kB and 31.3 dB, about 48× fewer residual bytes and about 6.8× the complete VVC anchor rate once the plate and crops are included. The foreground-only video is 5,249 B and adds 0.11 dB of foreground PSNR. The bytes in the unified residual are the background. After registration, the background residual of the warped plate is 107,005 B at QP 46 (background 29.56 dB) and 264,267 B at QP 40 (background 31.25 dB). At the same QP 40 the unregistered background residual was 293,597 B, so registration removed about 29 kB at matched quality and did not produce a Pareto win. VVC preset `faster` produced an empty residual file through the FFmpeg wrapper; direct `vvencapp` works, and medium is the current ladder preset. AV1 preset 10 at a matched corrected PSNR (QP 46, 33.58 dB) took 1,066,031 B against VVC medium QP 32 at 780,463 B and 33.59 dB.

### Long Horizon

Not remeasured. The 60.1% / 4.1 kB rows in the previous revision were the constant table.

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - The residual codec stays VVC medium unless a later sweep dominates it on this error video.
- **Next Action**: Do not sweep residual QP further on the still plate. The residual that matches the headroom argument is the player-region error after a conventional encode of plate-inpainted frames.
