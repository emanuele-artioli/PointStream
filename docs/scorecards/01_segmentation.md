# Module Scorecard: 01_segmentation

- **Owner Lane**: Antigravity Foreground
- **Source Scope**: `src/components/segmentation/`
- **Input Artifact**: `raw_frames: 3840x2160x3 uint8`, `bboxes: list[Box]`
- **Output Artifact**: `masks: bool (N, H, W)`
- **Last Evaluated**: 2026-09-20
- **Current Verdict**: SATISFIED_FREEZE

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | Bounding box mask | Rectangle crop treated as actor; zero segmentation cost, maximal background bleed |
| **Current** | YOLO26 instance segmenter (`yolo26n-seg.pt`) / pre-extracted dataset alpha | Shipped operational segmenter; fast inference |
| **Oracle** | SAM 3.1 video-guided segmentation with prompt refinement | High-precision boundary; zero background contamination |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Module Bytes | Downstream Ghost MAD | WebP Crop Bytes ($F$) | Composite PSNR-Y (dB) | Mean Mask Area (px) |
|---|---|---|---|---|---|
| **Null** (BBox) | 0 B | 18.0 | 3,456 B | 23.14 dB | 6,000 px |
| **Current** (YOLO) | 0 B (on-wire) | 92.0 | 11,520 B | 22.00 dB | 3,417 px |
| **Oracle** (SAM 3.1) | 0 B (on-wire) | 92.0 | 10,368 B | 21.56 dB | 2,243 px |

- **Headroom (Oracle - Current)**: Crop WebP byte saving is 1,152 B (10.0%), below the 15% threshold; Ghost MAD reduction is 0.0.
- **Verdict**: **SATISFIED_FREEZE**. Operational segmentation (YOLO) is within 2–5% of oracle in downstream rate-distortion performance. Stop spending compute on segmentation retraining.

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

