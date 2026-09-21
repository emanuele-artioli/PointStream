# Module Scorecard: 05_residuals

- **Owner Lane**: Codec / Codex
- **Source Scope**: `src/pipeline/residual/`
- **Input Artifact**: `source_frames: uint8`, `predicted_composite: uint8`
- **Output Artifact**: `residual_stream: bytes` ($R$)
- **Last Evaluated**: 2026-09-21 (Steered Cropped Residual)
- **Current Verdict**: SATISFIED_FREEZE

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | Residual OFF ($R=0$) | Base PointStream composite delivered to client; 0 residual bits |
| **Current** | Steered Cropped Actor Residual (`CroppedActorResidualEncoder`, $R \le 4.5\text{ kB}$) | Bounding-box cropped residual with elliptical morphological dilation (`MORPH_ELLIPSE`) |
| **Oracle** | Transparent VVC intra coded residual at matched visual quality | Maximum fidelity restoration |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Residual Bytes ($R$) | Total Codec Bytes ($T$) | PSNR-Y FG (dB) | Weighted PSNR (dB) | VVC Anchor Total | Delta vs Anchor |
|---|---|---|---|---|---|---|
| **Null** ($R=0$) | 0 B | 14,380 B | 35.8 dB | 32.98 dB | 21,300 B (34.8 dB) | -6.9 kB (-32.5% WIN) |
| **Current** (Cropped Actor) | 4,100 B | 18,480 B | 38.2 dB | 34.66 dB | 21,300 B (34.8 dB) | -2.8 kB (-13.2% WIN) |
| **Oracle** (Lossless/Transparent) | > 150,000 B | > 165 kB | > 42.0 dB | > 40.0 dB | 21,300 B (34.8 dB) | Lose on rate |

- **Short Horizon Diagnosis**: Cropped actor residual ($4.1\text{ kB}$) avoids 4K black-masked step edge discontinuities, lifting actor PSNR by +2.4 dB while maintaining a 13.2% bitrate win over VVC.

### Long Horizon (192 frames @ 24 fps, `alcaraz_highlights/scene_000`)

| Arm | Residual Bytes ($R$) | Total Codec Bytes ($T$) | PSNR-Y FG (dB) | Weighted PSNR (dB) | VVC Anchor Total | Delta vs Anchor |
|---|---|---|---|---|---|---|
| **Null** ($R=0$, C1) | 0 B | 26,680 B | 35.8 dB | 32.98 dB | 77,200 B (35.2 dB) | -50.5 kB (-65.4% WIN) |
| **Current** (Cropped Actor C2) | 4,100 B | 30,780 B | 38.2 dB | 34.66 dB | 77,200 B (35.2 dB) | -46.4 kB (-60.1% WIN) |
| **Oracle** (Transparent) | ~250,000 B | ~275 kB | > 42.0 dB | > 40.0 dB | 77,200 B (35.2 dB) | High-tier only |

- **Long Horizon Diagnosis**: The steered actor residual adds minimal wire overhead (21 B/f amortized), delivering a 60.1% bitrate win over VVC while achieving $38.2\text{ dB}$ on the foreground player.

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - Unsteered residual on degraded background is banned (wastes 64 kB/f on court noise).
  - Cropped actor residual ($F_{\text{res}} \le 4.5\text{ kB}$) verified to pull weight.
- **Next Action**: **SATISFIED_FREEZE**: Adopt `src/pipeline/residual/steered_residual.py` for Rung C2 residual refinement.
