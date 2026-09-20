# Module Scorecard: 04_motion_metadata

- **Owner Lane**: Cursor
- **Source Scope**: `src/runner/mask_wire.py`, `src/components/motion/`, `src/components/transport/`
- **Input Artifact**: `trajectories: list[Trajectory]`, `bboxes: list[Box]`, `masks: bool`
- **Output Artifact**: `wire_metadata_payload: bytes` ($M$)
- **Last Evaluated**: 2026-09-17 (E06 floor arms)
- **Current Verdict**: ACTIVE_SEARCH

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

| Arm | Metadata Bytes ($M$) | % of Total Codec Rate | Whole-Codec PSNR-Y | Notes |
|---|---|---|---|---|
| **Null** (Raw NumPy / JSON) | 70,609 B | > 70% | 26.2 dB | Exceeds entire VVC budget alone |
| **Current** (E06 Per-frame RLE) | 17,581 B | ~28% | 26.2 dB | Fits below VVC 21,288 B anchor, but leaves little margin for B and F |
| **Oracle** (Delta + Entropy wire) | ~3,200 B | ~5% | 26.2 dB | Transparent lossless packing |

- **Headroom (Oracle - Current)**: ~14.3 kB can be saved strictly by optimizing lossless transport without affecting image quality.

### Long Horizon (192 frames @ 24 fps, `alcaraz_highlights/scene_000`)

| Arm | Metadata Bytes ($M$) | Amortized Bytes/frame | Whole-Codec PSNR-Y |
|---|---|---|---|
| **Null** | ~120,000 B | 625 B/f | 27.2 dB |
| **Current** (E06) | ~40,343 B | 210 B/f | 27.2 dB |
| **Oracle** | ~10,500 B | 54.7 B/f | 27.2 dB |

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - Metadata is strictly lossless: quality delta is always 0, metric is purely bytes saved.
  - Current E06 reduced metadata from 70 kB to 17.5 kB on 48f.
  - If delta-coded predictor arms achieve $\le 5\text{ kB}$ metadata on 48f, mark **SATISFIED_FREEZE**.
- **Next Action**: Execute pending Cursor card `evaluation_20260917_e06_floor_predictor_probe.json` to lock in delta-coded metadata.
