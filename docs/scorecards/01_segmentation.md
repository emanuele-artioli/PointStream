# Module Scorecard: 01_segmentation

- **Owner Lane**: Antigravity Foreground
- **Source Scope**: `src/segmentation/` (was `src/components/segmentation/`)
- **Input Artifact**: clip (video or frames) + domain foreground classes
- **Output Artifact**: `ClipMasks` (lossless per-instance COCO-RLE, `masks.rle`)
- **Last Evaluated**: 2026-09-20
- **Current Verdict**: SUPERSEDED — the 2026-09-20 verdict below is not evidence

> **2026-10-05.** The tables below came from `experiments/modular/eval_segmentation_impact.py`,
> which draws flat-colour frames and fixed ellipse "masks" (lines 147–198); no
> segmentation model ran, so "Freeze YOLO; SAM 3.1 below 15%" is withdrawn. The
> module is rebuilt as `src/segmentation` (SAM 3.1 reference, YOLOE-26 n–x
> candidates, domains in `src/segmentation/domains.yaml`). Agreement with SAM 3.1
> (J, boundary F, recall/precision, flicker) and throughput come from
> `python -m src.segmentation suite`; see the branch
> `claude/pointstream-segmentation-module-3e59bf` and its fleet job records.

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

