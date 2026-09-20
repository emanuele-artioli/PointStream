# Module Scorecard: 04_motion_metadata

- **Owner Lane**: Cursor
- **Source Scope**: `src/runner/mask_wire.py`, `src/components/motion/`, `src/components/transport/`
- **Input Artifact**: `trajectories: list[Trajectory]`, `bboxes: list[Box]`, `masks: bool`
- **Output Artifact**: `wire_metadata_payload: bytes` ($M$)
- **Last Evaluated**: 2026-09-20
- **Current Verdict**: SATISFIED_FREEZE

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | Uncompressed raw structures (`np.savez` / JSON) | Baseline from Wave 2 pilot: 41.5 MB mask arrays, or 70.6 kB in legacy C0 |
| **Current** | E06 Lossless packed wire (mask RLE + thin placement) | Shipped operational transport: 17,581 B on 48f Federer 007 |
| **Oracle** | Keyframe delta-coded trajectories + entropy-coded RLE masks | Theoretical lower bound: $\le 3\text{ kB}$ total metadata over 48f ($\le 6\text{ kB}$ over 192f) |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Metadata Bytes ($M$) | Total Bytes ($T$) | Whole-Codec PSNR-Y | Notes |
|---|---|---|---|---|
| **Null** (Raw NumPy / JSON) | 70,609 B | ~100 kB | 20.68 dB | Exceeds entire VVC budget alone |
| **Current** (E06 Per-frame RLE) | 17,581 B | 30,297 B | 20.68 dB | Fits below VVC 21,288 B anchor budget for metadata |
| **Oracle** (Delta + Entropy wire) | ~3,200 B | ~15,900 B | 20.68 dB | Transparent lossless packing |

- **Predictor Headroom Finding**: Evaluated whether non-neural predictors (affine warp, first+last interpolation) can bridge the gap to VVC QP47 (24.6 dB). Even 48 ground-truth WebP crops per frame (`per_frame_crop_residual_off`) only reach **20.80 dB (+0.12 dB)**. Model-free predictor refinement is **disqualified** from attempting to bridge the 3.8 dB deficit.
- **Metadata Rate Finding**: E06 RLE/thin-metadata dropped metadata overhead from 70 kB to 17.5 kB, clearing the wire floor goal.

### Long Horizon (192 frames @ 24 fps, `alcaraz_highlights/scene_000`)

| Arm | Metadata Bytes ($M$) | Amortized Bytes/frame | Whole-Codec PSNR-Y |
|---|---|---|---|
| **Null** | ~120,000 B | 625 B/f | 27.2 dB |
| **Current** (E06) | ~40,343 B | 210 B/f | 27.2 dB |
| **Oracle** | ~10,500 B | 54.7 B/f | 27.2 dB |

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - Metadata lossless packing is verified at 17,581 B on 48f and ~40 kB on 192f.
  - Predictor refinement headroom is bounded at $+0.12\text{ dB}$; the deficit to VVC is in background plate fidelity ($B$), not actor placement.
  - Verdict: **SATISFIED_FREEZE**.
- **Next Action**: Freeze E06 per-frame RLE wire packing as the standard transport for all candidate configurations.

