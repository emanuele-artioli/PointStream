# Module Scorecard: 02_background

- **Owner Lane**: Antigravity Background
- **Source Scope**: `src/components/background/`
- **Input Artifact**: `raw_frames: 3840x2160x3 uint8`, `masks: bool (N, H, W)`
- **Output Artifact**: `canvas: VVC/WebP`, `homographies: float32 (N, 3, 3)`
- **Last Evaluated**: 2026-09-17 (E04B)
- **Current Verdict**: ACTIVE_SEARCH

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | `still_frame0` | Single unwarped keyframe (frame 0); fails on camera pans |
| **Current** | `registered_panorama` (removal-OFF / removal-ON) | Homography-stitched canvas encoded with VVC intra + float32 homographies |
| **Oracle** | Ideal clean-court canvas / `cleaned_video` | Perfect dynamic background without occlusion artifacts, encoded at highest quality |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Module Bytes ($B$) | Total Codec Bytes ($T$) | PSNR-Y Vis (dB) | SSIM Vis | VVC Anchor Total | Delta vs Anchor |
|---|---|---|---|---|---|---|
| **Null** (`still_frame0` QP47) | 31,814 B | ~60 kB | 20.06 dB | 0.8158 | 21,288 B (24.6 dB) | +39 kB / -4.5 dB (LOSE) |
| **Current** (`registered_panorama` QP47) | 32,368 B | ~62 kB | 26.22 dB | 0.9541 | 21,288 B (24.6 dB) | +41 kB / +1.6 dB (LOSE on rate) |
| **Oracle** (`cleaned_video` QP47) | 74,188 B | ~104 kB | 31.18 dB | 0.9824 | 21,288 B (24.6 dB) | +83 kB / +6.6 dB (LOSE on rate) |

- **Short Horizon Diagnosis**: At 48 frames, the fixed background plate (~32 kB) alone exceeds the entire VVC anchor budget (21,288 B). No background module polishing can win at 48 frames without amortizing over longer clips.

### Long Horizon (192 frames @ 24 fps, `alcaraz_highlights/scene_000` / Gate A)

| Arm | Module Bytes ($B$) | Amortized Bytes/frame | PSNR-Y Vis (dB) | SSIM Vis | VVC Anchor Total | Delta vs Anchor |
|---|---|---|---|---|---|---|
| **Null** (`still_frame0` C0) | 6,034 B | 31.4 B/f | 23.4 dB | 0.833 | 31,746 B (24.5 dB) | -25 kB / -1.1 dB |
| **Current** (`registered_panorama` C1) | 17,051 B | 88.8 B/f | 27.2 dB | 0.889 | 77,228 B (29.1 dB) | -60 kB / -1.9 dB (COMPETITIVE) |
| **Oracle** (`clean_canvas_vvc` C2) | 40,397 B | 210.4 B/f | 30.4 dB | 0.936 | 77,228 B (29.1 dB) | -37 kB / +1.3 dB (WIN) |

- **Long Horizon Diagnosis**: Amortization across 192 frames reduces the plate cost to 88–210 bytes/frame, opening a significant rate-distortion window against VVC inter-coding.

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - Background module is verified structurally unable to beat VVC at 48 frames due to fixed plate overhead.
  - At 192 frames, `registered_panorama` is competitive and Oracle shows a clear win window.
- **Next Action**: Lock background evaluation to the $\ge 192$-frame horizon for competitive claims; maintain 48-frame runs strictly as fast diagnostics.
