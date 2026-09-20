# Module Scorecard: 05_residuals

- **Owner Lane**: Codec / Codex
- **Source Scope**: `src/components/residual/`
- **Input Artifact**: `source_frames: uint8`, `predicted_composite: uint8`
- **Output Artifact**: `residual_stream: bytes` ($R$)
- **Last Evaluated**: 2026-09-10 (PR #88 repair)
- **Current Verdict**: ACTIVE_SEARCH

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | Residual OFF ($R=0$) | Base PointStream composite delivered to client; 0 residual bits |
| **Current** | Full-range signed uint8 residual encoded with AV1/VVC intra | Shipped operational residual codec |
| **Oracle** | Transparent VVC intra coded residual at matched visual quality | Maximum fidelity restoration |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Residual Bytes ($R$) | PSNR-Y (dB) | $\Delta \text{PSNR} / \Delta \text{Rate}$ vs Base | Pulls Weight vs VVC? |
|---|---|---|---|---|
| **Null** ($R=0$) | 0 B | 26.2 dB | Base | N/A |
| **Current** (QP 32) | 69,936 B | 31.5 dB | +5.3 dB for +70 kB | **NO** (VVC achieves 34.4 dB at 200 kB full-video) |
| **Oracle** (Lossless/Transparent) | > 150,000 B | > 42.0 dB | High bitfloor | Only at high-bitrate regimes |

- **Headroom & Evaluation**: Residuals only justify themselves if $\frac{\Delta \text{PSNR}}{\Delta \text{Bytes}}$ exceeds the marginal efficiency of conventional inter-coding. At ultralow rates ($< 100\text{ kbps}$), residual must stay OFF ($R=0$).

### Long Horizon (192 frames @ 24 fps, `alcaraz_highlights/scene_000`)

| Arm | Residual Bytes ($R$) | PSNR-Y (dB) | $\Delta \text{PSNR} / \Delta \text{Rate}$ | Pulls Weight vs VVC? |
|---|---|---|---|---|
| **Null** ($R=0$, C0–C3) | 0 B | 23.4 – 32.3 dB | Base | Competitive |
| **Current** (High-fidelity) | ~150,000 B | 38.5 dB | +6.2 dB for +150 kB | Marginal |
| **Oracle** | ~250,000 B | > 42.0 dB | Transparent | High-tier only |

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - If $\Delta \text{Quality} / \Delta \text{Rate}$ of the residual stream is worse than encoding the uncompressed difference with VVC: **RETIRE** or **FREEZE OFF** for low-rate rungs.
  - Residuals pull their weight only when base composite reaches $\ge 30\text{ dB}$ PSNR.
- **Next Action**: Keep residual OFF ($R=0$) for C0–C3 low-rate confirmation ladders; evaluate residual exclusively on high-fidelity tiers.
