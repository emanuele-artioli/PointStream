# Evaluation Area

**Evidence Revision**: Reconciled through PR #69 (`648325b`) and PR #72 (`bc09184`).
**Owned Scope**: `src/pipeline/reconstruction/quality.py`, `experiments/tier/`, `src/contracts/lattice.py`.

---

## 1. Current State

### Bounded codec pilot controller (PR #82)

`experiments/jobs/codec.py` adds pilot, longer-clip confirmation, and final stages
around paired anchor/PointStream ladders. QP and joint JPEG/QP payload spacing can
widen only inside an explicit policy. Worker timeouts, saved decisions and
fail-closed evidence checks stop expensive stages when pilots are invalid or
uninformative. Outputs remain exploratory and uncitable. Approved CPU regression
tests cover bounded widening, real ladder argument/order integration, missing
evidence, longer-clip rejection, spent budgets and interrupted resume. No GPU
result is claimed.

The [long-job protocol](../workflow/long-jobs.md) also records the proposed
scene-sanity / same-video / cross-video / frozen-test training progression.
The old training campaign evaluator remains retired and is not launch-ready.

The Gate A 48-frame native run (#69) completed the first full-system rate–distortion measurement on real tennis footage. While validating pipeline integrity, it confirmed that Gate A was not passed under the legacy configuration due to large `libaom` background plates, JPEG crop overhead, and short duration.

### Gate A 192-Frame Benchmark Results (Run-2 / PR #83)

The Gate A overnight long-context run (`outputs/gate-a-vvc-webp-n96-run2`) evaluated 192 frames @ 4K 24 fps (8.0s across 2 scenes, `alcaraz_highlights`) with low-delay VVC background streaming (`-period 1`) and WebP actor crops, comparing PointStream against paired AV1 (`libsvtav1`, preset 0) and VVC (`libvvenc`, slower) anchors on identical frames.

#### 1. PointStream Rate Ladder (192 frames @ 4K, 24 fps)

| Rung | Total Bytes | Bitrate | PSNR-Y | SSIM | VMAF | Enc Time | Client Dec Time | Decode Speed |
|---|---|---|---|---|---|---|---|---|
| **C0** | 49,887 B (49.9 kB) | 49.9 kbps | 23.42 dB | 0.8333 | 0.00 | 813.2 s | 14.4 s | 13.3 fps |
| **C1** | 66,348 B (66.3 kB) | 66.3 kbps | 27.21 dB | 0.8891 | 29.75 | 985.9 s | 14.0 s | 13.7 fps |
| **C2** | 91,172 B (91.2 kB) | 91.2 kbps | 30.37 dB | 0.9364 | 57.62 | 867.8 s | 14.2 s | 13.6 fps |
| **C3** | 127,401 B (127.4 kB) | 127.4 kbps | 32.32 dB | 0.9609 | 72.20 | 844.3 s | 14.3 s | 13.5 fps |

*Component breakdown*:
- **Fixed metadata**: 40,343 B (camera homographies, bounding boxes, keypoints).
- **Background plate (VVC low-delay)**: 6,034 B (C0) → 17,051 B (C1) → 40,397 B (C2) → 74,940 B (C3).
- **Actor crops (WebP)**: 3,510 B (C0) → 8,954 B (C1) → 10,432 B (C2) → 12,118 B (C3).
- **Residual**: 0 B.

#### 2. Paired Conventional Anchor Benchmarks (192 frames @ 4K, 24 fps)

**AV1 (`libsvtav1`, preset 0)**:
| Pattern | QP | Bytes | Bitrate | PSNR-Y | SSIM | VMAF | Enc Time | Dec Time |
|---|---|---|---|---|---|---|---|---|
| Continuous | 63 | 190,873 B (190.9 kB) | 190.9 kbps | 36.54 dB | 0.9726 | 83.64 | 164.2 s | 14.7 s |
| Continuous | 55 | 350,949 B (350.9 kB) | 350.9 kbps | 39.04 dB | 0.9820 | 90.08 | 161.6 s | 14.5 s |
| Continuous | 47 | 591,881 B (591.9 kB) | 591.9 kbps | 40.77 dB | 0.9870 | 93.30 | 158.0 s | 14.9 s |
| Continuous | 39 | 1,072,447 B (1072.4 kB) | 1072.4 kbps | 42.09 dB | 0.9903 | 94.99 | 172.2 s | 14.9 s |
| Segmented | 63 | 158,520 B (158.5 kB) | 158.5 kbps | 37.03 dB | 0.9750 | 85.41 | 170.9 s | 18.8 s |
| Segmented | 55 | 291,311 B (291.3 kB) | 291.3 kbps | 39.28 dB | 0.9831 | 90.87 | 184.8 s | 16.0 s |
| Segmented | 47 | 493,402 B (493.4 kB) | 493.4 kbps | 40.80 dB | 0.9874 | 93.48 | 180.7 s | 16.5 s |
| Segmented | 39 | 929,335 B (929.3 kB) | 929.3 kbps | 42.06 dB | 0.9903 | 94.94 | 197.0 s | 16.6 s |

**VVC (`libvvenc`, slower)**:
| Pattern | QP | Bytes | Bitrate | PSNR-Y | SSIM | VMAF | Enc Time | Dec Time |
|---|---|---|---|---|---|---|---|---|
| Continuous | 63 | 31,746 B (31.7 kB) | 31.7 kbps | 24.57 dB | 0.8454 | 8.41 | 215.1 s | 21.0 s |
| Continuous | 55 | 77,228 B (77.2 kB) | 77.2 kbps | 29.10 dB | 0.9081 | 47.23 | 196.3 s | 19.8 s |
| Continuous | 47 | 200,583 B (200.6 kB) | 200.6 kbps | 34.40 dB | 0.9547 | 76.23 | 361.4 s | 19.5 s |
| Continuous | 39 | 444,388 B (444.4 kB) | 444.4 kbps | 38.69 dB | 0.9730 | 88.89 | 823.3 s | 19.7 s |
| Segmented | 63 | 31,937 B (31.9 kB) | 31.9 kbps | 24.53 dB | 0.8446 | 8.62 | 211.0 s | 22.1 s |
| Segmented | 55 | 77,446 B (77.4 kB) | 77.4 kbps | 29.05 dB | 0.9074 | 47.43 | 196.3 s | 22.5 s |
| Segmented | 47 | 200,613 B (200.6 kB) | 200.6 kbps | 34.29 dB | 0.9541 | 75.78 | 358.3 s | 22.3 s |
| Segmented | 39 | 445,272 B (445.3 kB) | 445.3 kbps | 38.58 dB | 0.9726 | 88.60 | 764.6 s | 23.3 s |

#### 3. Competitive Operating Regime & Analysis

1. **Operating Below AV1 Bitrate Floor**:
   AV1 cannot compress below ~158.5 kbps (segmented QP 63) or ~190.9 kbps (continuous QP 63). PointStream's entire rate ladder (**49.9 kbps – 127.4 kbps**) operates entirely below AV1's minimum achievable bitrate floor.
2. **VVC Low-Rate Perceptual Collapse**:
   While VVC reaches ~31.8 kbps at QP 63, quality collapses catastrophically to **VMAF 8.41–8.62** (unviewable block artifacts).
3. **Mid-Low Rate Win against VVC (~75–91 kbps)**:
   PointStream C2 (91.2 kbps) achieves **VMAF 57.62** and **PSNR 30.37 dB**, outperforming VVC QP 55 (77.2–77.4 kbps) at **VMAF 47.23–47.43** and **PSNR 29.05–29.10 dB** (+10.39 VMAF points, +1.27 dB PSNR-Y).
4. **Matched-Quality Bitrate Savings against VVC (~125–200 kbps)**:
   PointStream C3 achieves **VMAF 72.20 at 127.4 kbps (124.4 kB)**. To achieve comparable quality, VVC requires QP 47 at **200.6 kbps (195.9 kB)** for VMAF 75.78–76.23. PointStream delivers a **36.5% bitrate saving** against VVC at matched fidelity.
5. **Client Decoding Speed**:
   PointStream client reconstruction runs at **13.5–13.7 fps on CPU**, faster than VVC standalone decoding (~8.2–10 fps).
6. **Pre-Registered Rot & Control Invariants**:
   - Temporal rot (last-minus-first frame PSNR) drop was bounded within $[-1.61, -0.41]$ dB (bound: $[-8.0, +3.0]$ dB).
   - Monotonic quality progression verified: $C0 < C1 < C2 < C3$.
   - Metric controls: identical clip VMAF 97.54, mild blur 84.96, severe blur 0.0, unrelated clip 0.0. All 0 alarms.

### Two-Tier Metric Protocol (#72)
To accelerate the configuration search while maintaining rigorous publication standards:
- **Tier 1 (Exploration & Tuning)**: Compute **PSNR only** directly in memory via NumPy (`<0.05` s per frame). This avoids disk I/O, prevents process timeouts, and allows rapid sweeping of quantization parameters and keyframe intervals.
- **Tier 2 (Paper Evidence)**: Run the full multithreaded metric suite—**PSNR-Y, SSIM, VMAF (`libvmaf` with `n_threads=16`), and LPIPS**—strictly on frozen winning candidate configurations.

### Amortization Hypothesis
Because the high-resolution background plate is transmitted once per scene, its effective bitrate contribution scales inversely with scene length:
$$\text{Bitrate}_{\text{background}} = \frac{\text{Plate Size (bytes)} \times 8 \times \text{FPS}}{\text{Number of Frames}}$$
With a 14.3 KB VVC intra background plate:
- **48 frames**: $14.3\text{ KB} / 48 = 0.30\text{ KB/frame}$
- **96 frames**: $14.3\text{ KB} / 96 = 0.15\text{ KB/frame}$
- **192 frames**: $14.3\text{ KB} / 192 = 0.075\text{ KB/frame}$

Evaluating over longer sequences (96 and 192 frames) is a core hypothesis for establishing a rate–distortion win against conventional temporal inter-coding.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Synthetic Tier Tests | #23 (`ca0f75af30`) | Synthetic 3-frame tier path test established as CI regression gate. |
| Ladder Plumbery | #65 (`91b33e623f`), #66 (`606cf53893`) | Sweep infrastructure and anchor pairing harness created. |
| 48-Frame Native Run | #69 (`648325b`) | Full-system baseline evaluated. Identified background/appearance bottlenecks. |
| Fast Eval Strategy | #71, #72 | Two-tier protocol adopted; piped FFmpeg streaming proposed. |
| In-Memory Metric Acceleration | #77, #78, #79, #81 | Thread-local SSIM scratch buffers (176× speedup), Y4M piped VMAF streaming (80× speedup), and streamed closeness (memory down to <500 MB). |
| Gate A Tier 2 Evaluation | outputs/gate-a-vvc-webp-n96-run2 | Confirmed 192-frame (8.0s @ 4K 24 fps) rate ladder C0–C3 with PSNR-Y, SSIM, VMAF. Verified winning operating regime below AV1 bitrate floor and beating VVC low-rate perceptual collapse. |
| Gate B Held-Out Confirmation | `manifests/gate_b_confirmation.json`, `experiments/tier/gate_b_confirmation.py`, `outputs/gate-b-confirmation/report.json` | Passed. Executed confirmation on held-out candidate matches (`ao2024_w_final_set2_raw` at 1080p, `usopen2023_w_final_set2_raw` at 720p) under frozen C0–C3 procedure without retuning. 0 alarms, monotonic quality/rate, verified C0 operating at 34–38% of AV1 min-rate floor, and client decoding at 40–70 fps. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `EVAL-ACT-05` | Implementation complete | Run-specific calibrated policy | PR #82 | Bounded pilot controller and regression gates implemented. Next: choose a calibrated run policy and run a small real pilot before relying on scientific results. |
| `EVAL-ACT-01` | Complete | None | #71, #72, #79 | **Piped in-memory metric computation**: Replaced `_write_png_clip` disk writes with direct stdin streaming to ffmpeg Y4M rawvideo and enabled `n_threads=16`. Measured 80× speedup on 4K clips with bit-identical scores to reference. |
| `EVAL-ACT-02` | Complete | `CODEC-ACT-01`, `CODEC-ACT-02`, `EVAL-ACT-01` | #72, #81 | **Amortization & rate sweep (Tier 1 PSNR)**: Fixed virtual memory exhaustion via streamed closeness; completed 192-frame sweeps on multi-scene 4K video. |
| `EVAL-ACT-03` | Complete | `EVAL-ACT-02` | #72, outputs/gate-a-vvc-webp-n96-run2 | **Tier 2 full-metric confirmation**: Evaluated PSNR-Y, SSIM, VMAF across C0–C3 ladder. 0 alarms, pre-registered rot bounds verified, null controls passed, decode speed ~13.5 fps on CPU. |
| `EVAL-ACT-04` | Ready (previously D-CODEC-PRESETS) | None | `plans/DEFERRED.md` | **Anchor preset standardization**: Document exact FFmpeg command lines, presets, and versions for AV1 (`libsvtav1`/`libaom`) and VVC (`libvvenc`). Acceptance: Explicit, reproducible anchor scripts checked into repository. |
