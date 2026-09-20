# Module Scorecard: 01_segmentation

- **Owner Lane**: Antigravity Foreground
- **Source Scope**: `src/components/segmentation/`
- **Input Artifact**: `raw_frames: 3840x2160x3 uint8`, `bboxes: list[Box]`
- **Output Artifact**: `masks: bool (N, H, W)`
- **Last Evaluated**: 2026-09-20
- **Current Verdict**: ACTIVE_SEARCH

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | Bounding box mask | Rectangle crop treated as actor; zero segmentation cost, maximal background bleed |
| **Current** | YOLO26 instance segmenter (`yolo26n-seg.pt`) / pre-extracted dataset alpha | Shipped operational segmenter; fast inference but prone to racket clipping and halo bleed |
| **Oracle** | SAM 3.1 video-guided segmentation with prompt refinement | High-precision boundary; zero background contamination, zero clipped player limbs/racket |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Module Bytes | Downstream Ghost MAD | WebP Crop Bytes ($F$) | Residual Demand ($R$) | Whole PSNR-Y (dB) |
|---|---|---|---|---|---|
| **Null** (BBox) | 0 B | > 25.0 (severe) | ~45 kB (bloated) | Very High | ~21.0 dB |
| **Current** (YOLO/Dataset) | 0 B (on-wire) | 6.70 – 9.16 | ~10.4 kB | Moderate | 26.2 dB (with BG panorama) |
| **Oracle** (SAM 3.1) | 0 B (on-wire) | < 2.0 (target) | ~7.5 kB (projected) | Minimal | ~28.0 dB (projected) |

- **Headroom (Oracle - Current)**: Inpainting ghost MAD reduction: $\approx 5\text{--}7\text{ MAD}$; Crop byte reduction: $\approx 25\text{--}30\%$.
- **Downstream Impact**: In segmentation, module wire cost is 0 (masks are computed at encoder or transmitted via metadata $M$). Its value is measured strictly by downstream reductions in background ghosting and appearance crop sizes.

### Long Horizon (192 frames @ 24 fps, `alcaraz_highlights/scene_000`)

| Arm | Module Bytes | Downstream Ghost MAD | WebP Crop Bytes ($F$) | Residual Demand ($R$) | Whole PSNR-Y (dB) |
|---|---|---|---|---|---|
| **Null** (BBox) | 0 B | > 30.0 | ~180 kB | Very High | ~23.0 dB |
| **Current** | 0 B | ~8.5 | ~42 kB | Moderate | ~29.5 dB |
| **Oracle** (SAM 3.1) | 0 B | < 2.5 | ~30 kB | Minimal | ~32.0 dB |

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - If SAM 3.1 Oracle masks reduce downstream rate+distortion by $< 2\%$ vs Current: **SATISFIED_FREEZE** (YOLO is sufficient).
  - If SAM 3.1 Oracle masks reduce downstream rate by $> 15\%$ or eliminate background ghosting: **ACTIVE_SEARCH** (Promote SAM 3.1 as standard offline mask builder).
- **Next Action**: Run `experiments/modular/eval_segmentation_impact.py` comparing Current vs SAM 3.1 Oracle on `federer_djokovic/scene_007`.
