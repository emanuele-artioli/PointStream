# Where does joint foreground coding pay off?

The evaluation needs a rate–quality figure that supports a choice between complete streams. Use the two external native rally banks already scored on twelve frozen source instants per rally. Plot a separate panel per rally, actual transmitted rate and equal-class foreground DISTS, with separate supplemental player and ball endpoints. Keep each rally's cache cost, support and eligible population explicit.

**Decision before analysis.** Count a measured gain only when a physical PointStream packet fits the given byte ceiling and improves on the best affordable physical comparator. Test whether the higher-rate selection differs across rallies or endpoints. A remaining gap in a tested bank only supports a bracket conditional on that bank. A better full-size or neural control can move it. The low-rate body advantage may persist where perceptual loss on the background or temporal motion still favors video. Report measured choices at equal ceiling and separately compute PCHIP BD-rate on the shared quality interval against each physical family’s upper concave frontier. Report the interval alongside each estimate and retain the unsupported cases. Never interpolate between matches or automatically choose a codec for playback from its test-set scores.

**Resources.** Reuse sealed native score and byte receipts. No new reconstruction or GPU on this analysis card. Use 3 local CPU-minutes for support checks, a single-rally smoke and the final two-rally plots. The companion Shanghai native parity and wider-rung run has its own card and resource budget.

## Shanghai mid-rate ordinary controls

The existing 1080p control bank jumps in scored quality between 118.46 and 146.81 kbps. The receiver elevation would be misleading if compared only with the earlier low-rate controls. Search native full-HD first at ceilings of 90, 120 and 140 kbps. The prior 1080p rate receipt already brackets these prices near legacy quarter-CRFs 253 and 246. Score the largest affordable verified stream at each ceiling with the same source and registered twelve instants. The existing calibrated SVT-AV1 preset 4 and dav1d worker binds the source, 24-frame smoke and all 240 deliveries at 29.97 fps. Budget 38 aggregate CPU-minutes including its validation and no GPU. If the 1080p controls still leave the rank unclear where the new joint points land, consider the same three ceilings on a 960-wide control as a separately recorded follow-up. Continue only after the original source-matched controls remain valid.

The full-HD control pilot is fleet job `20261123T115627Z-64bfa092`. Its inputs, budget and native toolchain are in the [registered specification](shanghai-av1-1920-spec.json). [Submission receipt](shanghai-av1-1920-registration.json)

## Shanghai neural anchor

The saved 480-wide focused DCVC-UF bank stops short of the new mid-rate range. Reuse the validated 48-frame setting at 960×544 Q42 to anchor that gap, once on an idle RTX 6000 Ada under the current budget. The earlier shared-GPU attempt `20261122T105300Z-7f7cd2fa` passed smoke and failed during the full receiver decode and supplies no score. This exclusive attempt changes the resource condition, keeps the production entry point and independent decoder checks, and must pass the same timestamp and paid-byte validation. Cap the job at 9 GPU-minutes. If it fails again, retain the missing neural point and its error. If it succeeds, inspect its achieved price before deciding whether a quantizer neighbor could change the ranking.

## How the object priorities and receiver state change the choice

At fixed physical byte ceilings, find the full interval of body weights for which a particular PointStream packet is no worse than every affordable conventional and neural packet. For each pair constrain the linear combination of body DISTS and ball DISTS. This brackets an interpretation of the equal-class point using the measured object scores; it never weights the background or claims calibrated viewer preferences. Reuse the genuine native score and price receipts. For each cached PointStream rung, find the least expensive measured baseline of equal or better DISTS, without score interpolation. Where that baseline has a higher rate, calculate the active seconds needed to amortize the physical initialization bill. If the better baseline is already cheaper, repeat reuse cannot favor this PointStream rung on that metric. This is a measured-bank sensitivity, conditional on known cache validity and stationary running cost, with no claim of reuse across actual cuts.

The exclusive neural control `20261123T115906Z-88158745` stopped before encoding. The production worker has a qualified shared-allocation resource contract and rejected the exclusive resource profile. Keep the resource checks intact. This leaves the 960-wide Shanghai bank without a qualified neural result from this worker. [Terminal receipt](shanghai-neural-960-q42-status.json)

The ROI bank also lacks adequate mid-rate coverage. Reuse the native 640-wide +40-outside map and hold the source-mask, transmitted-ball and time supports fixed. Target 90/120/140 kbps with the same three ceilings as the full-HD ordinary job. Admit only physically sent ROI streams whose same-CRF no-ROI twin confirms that the allocated map changes the decoded luma. One registered job, up to 21 campus CPU-minutes and no GPU.

The 640-wide ROI job is `20261123T120820Z-35db68b8`. [Submission receipt](shanghai-roi-640-registration.json)

## Native-control scores and metric cross-check

Both control jobs reached full validation. The six real 29.97-fps controls use the same twelve source instants. The full-HD AV1 control was recovered as paid IVF files and locally decoded by dav1d. All 240 decoded YUV frame bytes agree with the saved campus digest for each file. The ROI scores are extracted from the verified full decoded archive. Both paths retain source hashes, native comparator speed/profile and actual packet bytes. For the nearby rate switches, rescore the ordinary 89.34 and 118.90 kbps controls and the 118.75 kbps ROI on those same decoded frames with the existing LPIPS, luma SSIM, RGB PSNR and complementary background mask. This asks whether the first foreground DISTS reversal also changes spatial metric ranking. One CPU run under the existing five-minute matched-metric cap. This does not measure temporal agreement at the wider rates.

At the first Shanghai reversal, the complete 83.194 kbps PointStream stream scored 0.25960 foreground DISTS, 0.28478 LPIPS and 20.09 dB foreground RGB PSNR. AV1 at 89.344 kbps scored 0.25840, 0.24345 and 24.85 dB. Background PSNR was 23.98 and 34.11 dB. The 109.770 kbps PointStream packet at the identical coded width was almost unchanged on each foreground metric. These observations show when raising the same base quality buys little. They use actual native streams and a common spending ceiling, with no claim of a dense sequence of exact bitrate matches. [Spatial cross-check](shanghai-midrate-metrics.json)

A 640×384 PointStream packet regained the foreground-DISTS lead at 113.128 kbps (0.24221). Ordinary AV1 took it back at 118.904 kbps (0.23483). At that ceiling the 640-wide PointStream packet retained the lower ball DISTS (0.17149 vs 0.18776), and AV1 had the lower body DISTS (0.28190 vs 0.31294) and LPIPS (0.19052 vs 0.24098 on the combined region). The observed winner can change more than once across a rate ladder. The missing high-rate neural control limits the Shanghai comparison. [640-wide PointStream result](../shanghai-wider-20261123/q640-metrics.json) · [Physical results and BD-rates](results.json)

## Reproduce the figures

From the root with the project's SciPy and Matplotlib environment:

```sh
card=docs/experiments/joint-operating-range-20261123
shanghai=docs/experiments/shanghai-wider-20261123
PYTHONPATH=. python "$card/compare.py" OUTPUT_DIRECTORY \
  --shanghai-wider "$shanghai/q23-metrics.json" "$shanghai/q17-metrics.json" "$shanghai/q640-metrics.json" \
  --ordinary "$card"/ordinary-*.json --roi "$card"/foreground-roi-*.json
```

The relative rates in the JSON use a higher-is-better internal orientation for DISTS. The budget steps show only physical choices. PCHIP BD-rate is a secondary interpolation-based estimate and is left empty for pairs without a shared quality interval. The sparse joint ladder is especially important when interpreting the estimated savings.
