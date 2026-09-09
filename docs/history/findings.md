# Historical Findings, Invariants & Retractions

This document records scientific invariants, retracted conclusions, and experimental boundaries established across prior research waves. These constraints govern what may be claimed in the manuscript and prevent reviving flawed assumptions.

---

## 1. Invalid Early LPIPS & Self-Image Roster Results
- **Issue**: Early evaluations compared generated frames against identical reference keyframes using self-image LPIPS, yielding spuriously near-zero distances that did not reflect temporal synthesis quality.
- **Finding**: A metric evaluated without calibration against known anchors (mild degradation, severe blur, unrelated image) is uninterpretable. All subsequent quality evaluations enforce calibrated instruments and standard two-sided bounds.
- **Provenance**: PR #24 (`0d1ade6`), `plans/done/RESEARCH-HISTORY.md` §2.5.

---

## 2. BP10 Paste-Certifying Gate
- **Finding**: On whole-frame and crop PSNR/SSIM, pasting the reference keyframe consistently outperforms every evaluated generative neural model (ControlNet, IP-Adapter, Animate-Anyone).
- **Rule**: Generative models are optional experimental points; the default competitive codec pipeline keeps generation OFF and relies on reference pasting and residual coding.
- **Provenance**: PR #28 (`66da545`), `plans/done/RESEARCH-HISTORY.md` §2.10.

---

## 3. Synthetic vs Real Headroom Discrepancy
- **Finding**: Synthetic moving-block test clips demonstrated massive compression gains that completely vanished when evaluated on real 4K broadcast footage with fine textures and camera pans.
- **Rule**: Synthetic clips serve strictly as integration smoke tests (`tests/runner/test_tier_end_to_end.py`); they cannot be cited as rate–distortion evidence.
- **Provenance**: PR #22 (`7cf8e89`), PR #31 (`764e9d9`).

---

## 4. Withdrawn VVC Gap & Cross-Plate Subtraction Conclusions
- **Issue**: Early reports claimed PointStream had beaten VVC intra based on naive cross-plate residual subtraction.
- **Retraction**: The comparison subtracted plates across mismatched coordinate spaces, artificially depressing residual energy. When correctly aligned with homography compensation, the apparent crossover vanished. The claim was formally retracted.
- **Provenance**: PR #37 (`a00c6fb`), PR #45 (`ecebd9b`).

---

## 5. BP43 Circular Client Background Retraction
- **Issue**: BP43 proposed client-side background plate reconstruction from decoded stream frames.
- **Retraction**: Found to be logically circular: client-side plate synthesis required decoding the foreground, which itself depended on the client-side plate. Retracted in PR #48. The architecture strictly maintains canonical server-transmitted plates or sidecars.
- **Provenance**: PR #48 (`a66995c`).

---

## 6. BP53 Identity and Timing Limitations
- **Finding**: Measurement scripts previously contaminated pipeline execution times with filesystem sync latency and unisolated background recovery state.
- **Rule**: Encoder and decoder execution timings must be measured with disjoint timers, excluding disk I/O and warmup phases.
- **Provenance**: PR #61 (`f63019c`), direct commit `ee17a8a`.

---

## 7. Source Contamination & Confirmation Protocol
- **Finding**: Broadcast tennis sequences used during early exploration were repeatedly observed during feature engineering and plate stitching tuning.
- **Rule**: Cross-match generalization claims require the independent-match Gate B protocol. Within-source holdouts support a narrower claim; per-video fitting under a frozen encoding procedure is allowed with complete byte/time accounting. See [data protocol](../areas/data.md#4-confirmation-protocol).
- **Provenance**: PR #56 (`09727a4`), PR #57 (`77f30ec`), PR #60 (`f6f4f72`).

---

## 8. AVC QP ROI No-Op Finding (Deferred D3)
- **Finding**: Driving the conventional AVC encoder with region-of-interest QP maps in ffmpeg produced no rate reduction because the underlying encoder build silently ignored the ROI side-data.
- **Rule**: A flag existing is not a feature working. Any encoder knob must be driven and measured to verify bitrate modulation before relying on it in an ablation.
- **Provenance**: `plans/DEFERRED.md` (D3), paper appendix `roi_verification.tex`.

---

## 9. Gate A 48-Frame Checkpoint Alarms (#69)
- **Finding**: In the 48-frame native run (#69), PointStream failed to cross the AV1/VVC rate–distortion curve because:
  1. The tested `libaom` intra configuration produced a large plate; a codec-wide “artificial bitfloor” was not established.
  2. JPEG actor crops consumed 4–25 KB each.
  3. Short sequence duration (48 frames) provided insufficient amortization for 4K background plates.
- **Usability Limit**: No BD-rate calculation is citable from the #69 configuration. Gate A remains open pending VVC/SVT background sidecars, WebP appearance crops, and extended amortization (96/192 frames).
- **Provenance**: PR #69 (`648325b`), PR #70–#72.

---

## 10. Gate A 192-Frame Resolution & Winning Operating Regime (Run-2 / PR #83)
- **Finding**: Evaluated on 192 frames @ 4K 24 fps (8.0s across 2 scenes, `alcaraz_highlights`) with low-delay VVC background streaming (`-period 1`) and WebP actor crops, PointStream establishes a clear winning regime against conventional anchors:
  1. **Below AV1 Bitrate Floor**: AV1 (`libsvtav1`, preset 0) cannot compress below ~158.5 kbps (segmented QP 63) or ~190.9 kbps (continuous QP 63). PointStream's entire rate ladder (48.7 kB – 124.4 kB / 49.9 – 127.4 kbps) operates strictly below AV1's minimum floor.
  2. **Superiority over VVC at Mid-Low Rates**: At 91.2 kB (91.2 kbps), PointStream C2 achieves **VMAF 57.62** and **PSNR 30.37 dB** vs VVC QP 55 (75.4 kB / 77.2 kbps) at **VMAF 47.23** and **PSNR 29.10 dB** (+10.39 VMAF, +1.27 dB PSNR-Y).
  3. **Bitrate Savings over VVC at Quality**: PointStream C3 achieves **VMAF 72.20 at 124.4 kB**, whereas VVC requires QP 47 at **195.9 kB** for comparable quality (VMAF 75.78), delivering a **36.5% bitrate saving**.
  4. **Client CPU Decoding Viability**: Client reconstruction for 192 frames @ 4K took **14.0s–14.4s** (~13.5 fps) on commodity CPU without GPU acceleration.
- **Rule**: Claims of conventional codec superiority are valid within the 50–130 kbps operating band on broadcast sports with stationary or panning cameras; high-motion unmodeled scenes fall back to conventional coding.
- **Provenance**: PR #75, #79, #80, #81, #83; run artifact `outputs/gate-a-vvc-webp-n96-run2/report.json`.

## 10. Gate A/B pass interpretations superseded (2026-09-09)

PR #83/#84 compared unequal rates/qualities and generalized a sampled SVT-AV1 endpoint into a codec-wide floor. PR #85 declared Gate B passed whenever its alarm list was empty, without requiring competitive comparisons or the six-source protocol. The underlying reports contain unfavorable VVC comparisons and no AV1 quality overlap. Keep the original artifacts for provenance, but their pass labels are not citable. The [evaluation audit](../areas/evaluation.md) records evidence, instrument gaps, and the repair/search sequence. No raw video metric is asserted fabricated; the scientific verdict and claimed scope were wrong.
