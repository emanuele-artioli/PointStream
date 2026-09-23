# Module Scorecard: 03_appearance_crops

- **Owner Lane**: Antigravity Foreground
- **Source Scope**: `src/components/appearance/`, `src/components/generation/`
- **Input Artifact**: `raw_frames: 3840x2160x3 uint8`, `bboxes: list[Box]`, `masks: bool`
- **Output Artifact**: `actor_crops: AV1 intra QP 42`, `placement: tuple[int, int, int, int]`
- **Last Evaluated**: 2026-09-22 (measured 48-frame federer scene 007)
- **Current Verdict**: ACTIVE_SEARCH

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | Frame 0 crop repeated / zero generation | Transmit single keyframe crop, repeat without motion adaptation; minimal bytes, low dynamic fidelity |
| **Current** | AV1 intra QP 42, new crop when foreground MSE exceeds 50 | The crop winner in the useful PSNR band. The keyframe rule is what sets the byte count. |
| **Oracle** | Lossless/high-quality ground-truth actor crops pasted directly | Upper bound: exact source actor pixels inside bbox ($PSNR_{\text{actor}} = \infty$), compressed with optimal intra codec |

---

## 2. Whole-Codec Rate-Distortion Impact

### Short Horizon (48 frames @ 24 fps, `federer_djokovic/scene_007`)

| Arm | Module Bytes ($F$) | FG PSNR | Overall PSNR | Notes |
|---|---|---|---|---|
| **Before** (WebP q75, 48 keyframes) | 398,546 B | 35.59 dB | 20.51 dB | larger than the whole VVC anchor |
| **After** (AV1 intra QP 42, 48 keyframes) | 335,754 B | 36.70 dB | 20.51 dB | 62,792 B smaller and +1.1 dB FG; total C1 is 466,166 B |

- **Short Horizon Diagnosis**: The codec change helped. The rate failure is the keyframe rule: foreground MSE above 50 fires on every frame of this window (metadata is 960 B, 48 boxes). The old 12 kB budget was not what this ladder sent. `pose_oks` is null on this run.

The single-appearance motion control was also measured with the current WebP
plate. Bbox motion costs 8 B/frame and reaches 14.34 dB FG / 16.19 dB
weighted at 131,810 B total. COCO-17 keypoints cost 102 B/frame and reach
15.33 dB FG / 16.88 dB weighted at 136,228 B. The pose backend found all
48 frames and all 17 joints. These are classical-warp controls; no generator
has been evaluated on this arm.

On the registered VVC plate, a 12 kB appearance budget admitted six AV1 crops
(frames 0–4 and 7) and suppressed 42. Appearance bytes were 11,993 B.
Foreground PSNR moved from 14.34 dB to 17.70 dB. The paste covered every
foreground pixel, so the remaining error is the crop itself.

### Long Horizon

Not remeasured after the codec change. On the earlier WebP run, 192-frame appearance was 2,754,794 B.

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - Appearance bytes have to come down near the anchor, not only beat WebP.
- **Next Action**: Keep AV1 intra QP 42. The 12 kB budget is measured and is not enough. The open lever is a generator or a larger appearance budget, not another motion packing pass.


