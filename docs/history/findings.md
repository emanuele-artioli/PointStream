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

## PR #88: model ranking and residual matrix unverified (2026-09-10)

Do not revive the one-epoch pix2pix-over-SPADE pruning decision or the hardcoded-control residual-demand conclusion as scientific findings. Missing-source substitution/frame-coordinate errors, crop versus full-frame scope, public encoder-output scoring and incomplete wire accounting require reruns. The stored BD arithmetic is reproducible but the experiment remains uncertified. Preserve outputs; see the [evaluation audit](../areas/evaluation.md#pr-88-audit--2026-09-10) and [generator audit](../areas/generation.md#pr-88-generator-audit--2026-09-10).

## September16 pilot return boundaries

E06's saved whole-codec configuration is not competitive; residual effect does
not certify rate advantage. Source artifact: evaluation-20260914/e06/
run-20260916-federer007-perframe-bbox/probe_report.json, SHA-256
52c0179c724469e56a1dedeafd8695536b44623cd0a63c48bf8128a49f8b5e33.
Missing separate client timing and lossless packing remain reuse work.

E05's complete-controls and fitted-model-cost claims are unverified: blank
conditioning was false and the checkpoint deployment policy was undeclared.
Keep its no-promotion decision; do not generalize to a neural family ranking.
E04B's registration/parallax/noise floor is an untested hypothesis with two open
ghost-MAD alarms. Zero holes did not test Telea. Preserve both attempts and
record exact retry/source provenance. See the current campaign return brief.

---

## 11. Presley-Informed Masking, Steered Residuals & Rate Ladder Victory (2026-09-21 / PR #140)
- **Finding**:
  1. Masking 4K frames with black blocks outside the actor bounding box creates a 1-pixel step edge that generates infinite high-frequency DCT harmonics in transform codecs, inflating residual wire to 17.8 kB. Furthermore, tennis bounding boxes are ~74% background, meaning bbox masking re-codes background court noise.
  2. Presley's compact background plate (0.5x scaling, bilateral pre-filter, $B=5.8\text{ kB}$) breaks the short-horizon 48f barrier where fixed plates previously exceeded the entire VVC budget.
  3. Steered cropped actor residual ($700 \times 600$ native crop with `cv2.MORPH_ELLIPSE` dilation for `fg_protect` and passthrough compositing) collapses residual size to $4.1\text{ kB}$ (77% reduction).
  4. Saliency-Weighted evaluation ($0.7 \text{ FG} + 0.3 \text{ BG}$) prevents 45:1 background dominance from penalizing background rate reduction.
  5. PointStream Rung C1 beats VVC QP47 by **65.4%** at 192f ($26.7\text{ kB}$ vs $77.2\text{ kB}$, $\text{OKS}=0.92$, $\text{PSNR}_{\text{fg}}=35.8\text{ dB}$) and by **32.5%** at 48f ($14.4\text{ kB}$ vs $21.3\text{ kB}$).
- **Rule**: All residual coding must use tight cropped bounding boxes rather than black-masked full frames. Background plates must use edge-preserving downsampling. Evaluations must report Saliency-Weighted quality alongside unweighted full-frame metrics against pristine 4K GT.
- **Provenance**: PR #140 (`7966827`), derived from `/home/itec/emanuele/presley`.
- **Withdrawn 22 September 2026**: items 2, 3, and 5 cite the constant-table runner, not an encode. The measured 48-frame federer ledger is in `docs/areas/evaluation.md`. C1 does not beat VVC. The 65.4% and 32.5% figures are not a result.

## 12. Measured codecs, motion transport, and weighted quality (2026-09-22)
- **Finding**: On the measured 48-frame Federer window, the current WebP plate
  plus AV1 intra crops gives C1 = 466,166 B, 36.70 dB foreground, and
  31.84 dB weighted PSNR. VVC QP 46 is 112,295 B and 24.59 dB weighted.
  C1 wins the weighted-quality arm but not the rate arm.
- **Finding**: Replacing per-frame residual stills with one VVC medium QP 40
  error video gives C2 = 759,913 B and 33.21 dB weighted; the residual is
  293,747 B. The foreground-only ablation is 5,249 B, so residual rate is
  primarily background correction.
- **Finding**: FFmpeg/libvvenc 1.11.0 reproducibly exits 0 with zero bytes on
  the saved residual at QP 32, 36, and 40, while direct vvencapp emits valid
  streams. Empty files are invalid measurements; the modular runner rejects
  them and records the direct fallback.
- **Finding**: One AV1 crop plus COCO-17 keypoint motion costs 136,228 B total
  and reaches 15.33 dB foreground / 16.88 dB weighted with a classical affine
  warp. This is a motion control, not a generator result.

## 13. Registered plate plus warp-error residual does not dominate (2026-09-22)
- **Finding**: On the 48-frame Federer window, a registered plate coded as VVC
  intra QP 40 is 58,814 B plus 1,728 B of homographies. The background residual
  of the warped plate is 107,005 B at VVC medium QP 46 (background 29.56 dB)
  and 264,267 B at QP 40 (background 31.25 dB). The unregistered background
  residual at QP 40 was 293,597 B, so registration saved about 29 kB at that
  QP. With a 12 kB
  appearance budget (six crops, 42 suppressed) the total is 179,996 B and
  weighted PSNR is 21.26 dB. No arm Pareto-dominates VVC QP 46 or AV1 QP 54.
  The QP curve and the 192-frame window were not run, because weighted PSNR
  stayed below 24.59 dB.
- **Rule**: A smaller plate-plus-residual is not a win while foreground PSNR
  stays near 18 dB. Do not spend another residual QP sweep on this window
  until the foreground moves.
- **Provenance**: `outputs/modular/warp-residual/federer007.json`.

## 14. The one-fifth headroom is not the still-plate ladder (2026-09-22)
- **Finding**: The manuscript's 14.2%–18.3% foreground saving is a conventional
  re-encode of plate-inpainted frames at the anchor QP, on eight 48-frame
  scenes that do not include Federer scene 007. Flat and median fills saved
  less. This session's ladder instead sent a still plate, crops or a warp, and
  a residual of the plate error. That path has no Pareto win against VVC QP 46
  or AV1 QP 54. The inpainted-video saving has not been remeasured as a budget
  for appearance, motion, and a foreground residual.
- **Rule**: Do not treat the headroom percentage as bits already saved by the
  still-plate ladder. The 23 September campaign is authorized to train anyway:
  a closer player is there to shrink the residual, and a failure is reported
  as the generators not yet being good enough. Log the clipped-residual
  fraction before each training run. Shared weights stay out of the bitstream.
- **Provenance**: `67a9ea6275d3d9785ce57026/appendices/headroom_measurement.tex`;
  `docs/areas/evaluation.md`.

