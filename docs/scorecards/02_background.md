# Module Scorecard: 02_background

- **Owner Lane**: Antigravity Background
- **Source Scope**: `src/components/background/`
- **Input Artifact**: `raw_frames: 3840x2160x3 uint8`, `masks: bool (N, H, W)`
- **Output Artifact**: `canvas: WebP q40`, `homographies: float32 (N, 3, 3)`
- **Last Evaluated**: 2026-09-23 (four background arms, federer scene 007, 48 frames)
- **Current Verdict**: ACTIVE_SEARCH

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | `still_frame0` | Single unwarped keyframe (frame 0); fails on camera pans |
| **Current** | WebP q40 plate, registration off | Current PointStream choice; full-resolution VVC intra replacement is deferred. |
| **Oracle** | Ideal clean-court canvas / `cleaned_video` | Perfect dynamic background without occlusion artifacts, encoded at highest quality |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Module Bytes ($B$) | Total Codec Bytes ($T$) | Overall PSNR | FG / BG PSNR | VVC anchor QP 46 | Delta vs anchor |
|---|---|---|---|---|---|---|
| **Before** (WebP q40 plate + WebP crops) | 129,452 B | 528,958 B | 20.51 dB | 35.59 / 20.50 dB | 112,295 B (31.26 dB) | larger, and 10.8 dB lower overall |
| **Current** (WebP q40 plate + AV1 crops) | 129,452 B | 466,166 B | 20.51 dB | 36.70 / 20.50 dB | 112,295 B (31.26 dB) | 62,792 B smaller from crop codec; plate quality unchanged |

- **Short Horizon Diagnosis**: The current PointStream plate remains WebP q40. Background PSNR stays at 20.5 dB because registration is off while the camera moves. The full-resolution sweep found a VVC intra point, but replacing this plate is deferred. The 32.5% win in the old table was the constant-table runner and is withdrawn.

Registration with the current WebP q40 plate raises BG PSNR from 20.50 to
23.85 dB, but charges 166,680 B for the plate plus 1,728 B for float32
homographies. The deferred VVC QP 32 control is 119,558 B plus the same maps
at 23.88 dB BG. Neither registered static arm clears the VVC video anchor.

The warp-residual probe then coded that registered plate as VVC intra QP 40:
58,814 B, plus 1,728 B of maps. Background PSNR of the warped plate alone is
23.62 dB at 62,900 B total with one crop. Adding the warp-error residual
(107,005 B, QP 46) raises background to 29.56 dB at 169,905 B total. That is
under the old unregistered residual and still short of the anchor on both
bytes and background PSNR.

### Long Horizon (192 frames, `alcaraz_highlights/scene_000`)

Not remeasured after the codec change. The 65.4% row in the previous revision of this scorecard was the constant table. The earlier measured WebP run, before this codec change, had C0 at 148,060 B with foreground PSNR 13.3 dB, which beat VVC on rate and was not a usable point.

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - A plate win requires both fewer bytes than the anchor and background PSNR that is not stuck near 20 dB on a moving camera.
- **Next Action**: See the [background campaign](../workflow/session/evaluation-campaign/20260923-background-campaign.md). Perricard scene 002, inpainted VVC QP 46, is 86,894 B versus a 104,482 B source, court matched, 17,588 B left. Alcaraz scene 000’s panorama leaves 40,501 B at 1.5 dB under the source court. Federer scene 007’s inpainted video leaves 4,103 B. Production plate remains WebP q40 until a later step replaces it.
