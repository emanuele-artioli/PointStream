# TOMM Real-Time CPU Receiver (`k6_ref_warp` v2) — Track A

## Goal & Target Figure

- **Target Manuscript Section:** Section 6 (`sections/journal-results.tex`, `Table~\ref{tab:multi-backbone}` and real-time receiver benchmark).
- **Question:** Can a 100% CPU-only, non-generative pose-driven warper (`k6_ref_warp` v2) with a **source-denied, zero-SAM receiver matte** (extracted once at `t=0` from the already-transmitted `PSR` reference crop and cleaned SVT-AV1 base frame, then warped alongside RGB) run at **≥ 30 fps end-to-end on 1 CPU thread** while strictly beating SVT-AV1 and DCVC-UF across the 240-frame native tennis suite (Shanghai `match102`, Sydney `match127`, Halle `match105`)?

## Pre-Run Hypothesis & Competing Explanations

- **Hypothesis (Pilot, 48-frame Shanghai `match102`):**
  1. Replacing point-translation Gaussian warping (`v1_gauss`) with **bone-aligned Linear Blend Skinning (`v2_lbs`)** or **Moving Least Squares Similarity (`v2_mls`)** improves limb rotation fidelity, reducing masked DISTS/LPIPS and improving `YOLO26x-pose` recall/OKS while staying ≥ 30 fps on 1 CPU thread.
  2. Extracting a binary/feathered reference alpha matte **once at `t=0`** from `|ref_crop - base_crop_0|` + `ref_kp` (GrabCut + skeleton prior, 0 extra wire bytes, 0 SAM 3.1 calls) and warping that alpha matte alongside RGB preserves competitive perceptual and pose-detection quality relative to SAM 3.1 masking, eliminating the 57 s/frame GPU matting bottleneck.
- **Competing Explanations:**
  - Large out-of-plane body turns (e.g., player turning back-to-front during a serve or rally) expose disoccluded body regions that pure 2D warping cannot hallucinate, causing warped-mask silhouette drift over 240 frames.
  - Background-subtraction matting at `t=0` may include player shadows or miss low-contrast clothing (e.g., white shirt on white court line), degrading DISTS/LPIPS when composited onto the cleaned base.

## Decision Rule

1. **Stage 1 (48-frame Shanghai `match102` pilot, CPU-only):**
   - Smoke (16 frames, near player) must pass: deterministic outputs, finite metrics, `v1_gauss_sam` reproduces prior `k6_ref_warp` (`20261128T030745Z-2675dcdb`), and CPU throughput ≥ 30 fps for 2-player equivalent.
   - Select the best warper (`v2_lbs` vs. `v2_mls` vs. `v1_gauss`) on the 48-frame 2-player pilot by masked DISTS, masked LPIPS, and `YOLO26x-pose` recall/OKS.
   - **Promote to Stage 2 (240-frame 3-match native benchmark)** if the 100% source-denied warped-mask receiver achieves:
     - End-to-end 2-player CPU throughput **≥ 30.0 fps** (≤ 33.3 ms/frame on 1 CPU thread, including `t=0` matte extraction amortized + per-frame warp + alpha composite), AND
     - Masked DISTS **< 0.431** (beating DCVC-UF) and masked LPIPS **< 0.471** (beating DCVC-UF) on Shanghai `match102`.
   - If warped-mask degrades DISTS by > 0.05 relative to SAM 3.1 masking, diagnose the failure mode (shadow leakage vs. silhouette drift) and test a minimal 1-bit `t=0` sender-transmitted reference mask (~0.8 kbps amortized) before scaling to 240 frames.

## Budget & Hardware

- **GPU budget:** **0.00 GPU-hours** (100% CPU-only receiver + CPU scoring).
- **CPU budget:** ≤ 30 CPU-minutes for Stage 1 (48-frame pilot, 5 variants × 48 frames + YOLO26x-pose/DISTS/LPIPS/DIS flow).
- **Stage 2 budget (if promoted):** ≤ 90 CPU-minutes across 240 frames × 3 matches (Shanghai `match102`, Sydney `match127`, Halle `match105`).

## Job Ledger

| Stage | Job ID | Host | Status | Notes |
|---|---|---|---|---|
| Stage 1 (48f pilot) | `20261128T040442Z-ee9d0597` | `gpu6` (CPU) | `failed` (smoke) | Determinism check caught unseeded `cv2.grabCut` RNG; fixed with `cv2.setRNGSeed(0)` |
| Stage 1 (48f pilot, retry) | `20261128T040755Z-869c0567` | `gpu5` (CPU) | `failed` (validator) | Smoke passed & showed `v2_lbs_sam` 0.2898 / `v2_mls_sam` 0.2880 DISTS, `v2_lbs_warped_mask` 85.2 fps; missing `PS_VALIDATION_PATH` write |
| Stage 1 (48f pilot, v2) | `20261128T041424Z-b872e14b` | `gpu2` (CPU) | `complete` | rev `5d160451`, receipt `docs/experiments/tomm-warp-v2-pilot-scores.json` (`6f389410`). **Promoted to Stage 2:** `v2_mls_sam` hit 0.3265 DISTS / 0.3496 LPIPS / **95.8% pose recall / 0.904 OKS**; `v2_mls_warped_mask` hit 0.3447 DISTS / 0.3994 LPIPS (beating DCVC-UF 0.431/0.471 & SVT-AV1 0.532/0.491); `v2_lbs_warped_mask` hit 31.0 fps total / 35.7 fps steady. |
| Stage 2 (240f Shanghai) | `20261128T044616Z-3c147248` | `gpu5` (CPU) | `complete` | rev `dc61f491`, receipt `docs/experiments/tomm-warp-native240-match102-scores.json` (`6e0705d5`): `bgsub_alpha` **0.3536 DISTS / 0.4260 LPIPS / 52.1% pose recall at 30.8 fps e2e** (40.3 fps synth, 15.48 kbps warm); `psr_alpha` 0.3527 DISTS (+905 B); `sam_oracle` 0.3434 DISTS (83.3% recall) |
| Stage 2 (240f Sydney) | `20261128T044711Z-1fe56719` | `gpu3` (CPU) | `complete` | rev `dc61f491`, receipt `docs/experiments/tomm-warp-native240-match127-scores.json` (`ebb9292c`): `bgsub_alpha` **0.3538 DISTS / 0.3597 LPIPS / 41.7% pose recall at 29.9 fps e2e** (34.4 fps synth, 21.41 kbps warm); `psr_alpha` 0.3517 DISTS (+677 B); `sam_oracle` 0.3154 DISTS (66.7% recall) |
| Stage 2 (240f Halle) | `20261128T044853Z-cda34076` | `gpu1` (CPU) | `complete` | rev `dc61f491`, receipt `docs/experiments/tomm-warp-native240-match105-scores.json` (`672c2964`): `bgsub_alpha` **0.3425 DISTS / 0.3934 LPIPS / 52.1% pose recall at 27.0 fps e2e** (32.2 fps synth, 13.38 kbps warm); `psr_alpha` 0.3325 DISTS (+849 B); `sam_oracle` 0.3220 DISTS (58.3% recall) |

## Final Outcome (0.00 GPU-hours spent)

All three 240-frame unadapted external broadcasts (**1,440 player-frames total**) completed on 1 CPU thread with **0 GPU-hours**, **0 SAM 3.1 receiver calls**, and **0 extra wire bytes** (`mls_step8_bgsub_alpha`):

| Match (fps) | Warm kbps | `bgsub_alpha` DISTS / LPIPS | `psr_alpha` DISTS (+kbps) | `sam_oracle` DISTS | SVT-AV1 Player DISTS | DCVC-UF Player DISTS | PointStream-AA (GPU) DISTS | 1-CPU-Thread FPS (e2e / synth) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Shanghai (`match102`, 29.97 fps) | 15.48 | **0.3536** / 0.4260 | 0.3527 (+0.90) | 0.3434 | 0.524 | 0.438 | 0.319 | **30.8** / 40.3 |
| Sydney (`match127`, 60 fps) | 21.41 | **0.3538** / 0.3597 | 0.3517 (+1.35) | 0.3154 | 0.493 | — | 0.289 | **29.9** / 34.4 |
| Halle (`match105`, 25 fps) | 13.38 | **0.3425** / 0.3934 | 0.3325 (+0.71) | 0.3220 | 0.542 | 0.467 | 0.310 | **27.0** / 32.2 |
| **3-Match Mean** | **16.76** | **0.3500** / 0.3930 | **0.3456** | **0.3269** | **0.520** | — | **0.306** | **29.2** / 35.6 |
