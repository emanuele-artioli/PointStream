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
| Stage 1 (48f pilot, retry) | _pending_ | CPU | — | `experiments.tennis.tomm_warp_v2_pilot` (seeded GrabCut) |
