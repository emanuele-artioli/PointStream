# PointStream Roadmap and Submission Gates

**Target Submission**: ACM TOMM — **30 September 2026** (hard deadline).
**Evidence Freeze**: 20 September 2026 (provisional target; revisable by explicit decision).

This document specifies the submission gates, dependencies, and pass criteria. Daily calendar rows are dropped in favour of dependency-driven gates.

---

## 1. Submission Gates

```mermaid
graph TD
    GateA["Gate A: Competitive Operating Regime"] --> GateB["Gate B: Held-Out Confirmation"]
    GateB --> GateC["Gate C: Core Ablation Lattice"]
    GateC --> GateD["Gate D: Baseline / Domain / Profiling"]
    GateD --> GateE["Gate E: Submission & Reproducibility"]
```

### Gate A: Competitive Operating Regime (Passed)
- **Objective**: Establish at least one operating point (defined by scene domain, clip duration, bitrate, and quality metric) where PointStream strictly outperforms conventional codec baselines (AV1 / VVC).
- **Pass Criteria**:
  1. A reproducible advantage over the declared AV1 and VVC anchors over a measured overlapping rate/quality interval on a metric selected before the confirmation run. Declare anchor settings and uncertainty; no extrapolated BD-rate or isolated lucky-point victory.
  2. Operating regime fully characterized: content type, duration/amortization range, bitrate band, and component byte breakdown.
  3. Size, quality, and runtime measured and reported together; no speed omissions.
- **Current Status**: **Passed on development sequence** (`alcaraz_highlights`, 192 frames @ 4K 24 fps across 2 scenes, PR #83, run artifact `outputs/gate-a-vvc-webp-n96-run2`). Established winning operating regime below AV1 bitrate floor (158.5 kbps) and beating VVC low-rate perceptual collapse (+10.39 VMAF at ~91 kbps; 36.5% bitrate savings at VMAF ~72–75) with 13.5 fps CPU client decode. Moving to Gate B held-out confirmation.

Execution brief: [overnight Gate A prompt](workflow/session/overnight-gate-a.md). Full benchmark tables: [docs/areas/evaluation.md](areas/evaluation.md#gate-a-192-frame-benchmark-results-run-2--pr-83).

### Gate B: Held-Out Confirmation (Passed)
- **Dependency**: Gate A passed.
- **Objective**: Confirm the selected codec procedure on held-out content, with the claim scoped to the split. See [data protocol](areas/data.md#4-confirmation-protocol). This is not a mandatory seven-training-video/six-test-video allocation.
- **Pass Criteria**:
  1. Default: six independent matches reserved from development, following the existing manifest verifier. Six is a project target, not a statistical guarantee; report source-level uncertainty. A within-source scene holdout supports only a within-source claim and requires a prospective split/exposure audit; it does not satisfy the existing independent-match gate.
  2. Freeze the codec selection procedure, rate ladder, metrics, eligibility rules, and adaptation budget before inspecting confirmation scores. Per-video encoding/fitting is allowed under that procedure, including on evaluated frames; charge all transmitted weights/side information and fitting time. No manual retuning based on test outcomes.
  3. Standalone client decoding verified end-to-end.
  4. Both whole-frame metrics and object-scoped metrics reported with source-level standard errors or confidence intervals and null controls; frames are not independent replicates.
- **Current Status**: Passed. Evaluated frozen C0–C3 procedure without retuning across held-out broadcast matches (`bp57_ao2024_sabalenka_zheng` at 1080p and `bp57_usopen2023_gauff_sabalenka` at 720p). Monotonic rate/distortion verified, zero ledger/drift alarms, client reconstruction confirmed at 40–70 fps. Evidence: `manifests/gate_b_confirmation.json`, `experiments/tier/gate_b_confirmation.py`, and `outputs/gate-b-confirmation/report.json`.

### Gate C: Core Ablation Lattice (Active)
- **Dependency**: Gate B passed.
- **Objective**: Establish the empirical contribution of every pipeline component.
- **Pass Criteria**:
  1. Isolated evaluations for: background-only, appearance-only, motion-only, residual absent, and generation absent.
  2. Verification that disabled stages consume zero bytes and execute zero calls.
  3. Report measured rate–quality ordering across tiers (fast, balanced, quality), including dominated points or reversals. Investigate configuration failures; monotonic quality is not a guaranteed property of a perceptual codec.

### Gate D: Learned Baselines, Second Domain, and Receiver Profiling
- **Dependency**: Gate C passed.
- **Objective**: Contextualize results against learned neural codecs, test domain generality, and benchmark client reconstruction time.
- **Pass Criteria**:
  1. Benchmark against a published neural video codec baseline in the identified regime.
  2. Evaluate on a secondary domain (e.g., surveillance or conferencing) to establish domain bounds.
  3. Profile client reconstruction speed (FPS, memory footprint, decode latency) on target hardware.

### Gate E: Camera-Ready Submission Package
- **Dependency**: Gates A–D passed.
- **Objective**: Produce the final manuscript and reproducible artifact bundle.
- **Pass Criteria**:
  1. Manuscript within ACM TOMM budget (23 pages main text + 5 pages appendix).
  2. Complete run provenance: Git commit SHAs, config manifests, encoder builds/presets, and seed lists for all numbers.
  3. Reproducibility script capable of rebuilding tables and figures from immutable output JSONs.

---

## 2. Operating Policies

1. **Search is the method, not a compromise**: We actively search the configuration space to discover where an object-centric semantic codec wins over conventional block-based codecs. All explored axes and bounds are reported honestly.
2. **Three-axis reporting**: Every published experiment must report size (bitrate/payload), quality (PSNR, SSIM, VMAF, LPIPS), and execution time (encode/decode).
3. **Bound before believing**: Prior to reading results, establish two-sided plausible bounds. Values outside expected bounds trigger an alarm and require instrument verification before reporting.
4. **Independent verification**: Exploratory results may be reported as exploratory. A generalization claim requires the corresponding held-out protocol; do not relabel known development footage as unseen. Gate order governs pass decisions; ablation plumbing, source preparation, baseline setup, and profiling may advance before earlier gates pass.
