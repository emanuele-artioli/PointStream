# Conventional controls for the first combined rally

## Decision registered before encoding

Use the opened Nadal–Kyrgios 000@120 development window, 240 native 50 Hz frames. Can local native AV1 and a player-region bit-allocation control bracket the cold player packet at the same frame cadence? Ordinary allocation is the competing explanation for a regional gain. This pilot measures stream sizes and whether the map changes the allocation. A scientific quality comparison also needs the full combined stream, independently reviewed foreground and ball locations, and the neural anchor.

Use the receipt-bound source and two conditioning files from the November 16 player qualification. Reuse their encoder-side player crop boxes to define one reproducible control. In both modes resize limited-range YUV420 to 960×544 with an area filter. Encode the plain source and the same source with the project’s +20 outer-block QP map, SVT-AV1 preset 4 at CRFs 36, 48 and 60. Keep the stronger result when comparing anchor families later. Each map covers both player boxes through the shot. No ball hint is available to this control.

Test the first 24 frames at CRF 48 with both modes. Require both native streams to decode every frame, ordered IVF timestamps, a nontrivial region on every frame, two different decoded pixel streams and a reduced ROI file size. Compare luma error inside the boxes and on their complement as an allocation diagnostic. Use all pixels with their true counts. For the full pilot report all six points and whether each family brackets the current JPEG player carrier rates; retain all outcomes. Allow 20 wall minutes and 60 local CPU-minutes for smoke and full combined. Start full only if the smoke projection fits. No campus/GPU budget. Raw outputs are private because they contain broadcast footage.

## Measured controls on 240 native timestamps

| CRF | Plain, kbps | Player ROI, kbps | Box luma PSNR, plain / ROI (dB) | Complement luma PSNR, plain / ROI (dB) |
|---|---:|---:|---:|---:|
| 36 | 637.44 | 565.62 | 38.36 / 38.37 | 44.33 / 43.68 |
| 48 | 303.43 | 288.23 | 35.06 / 35.05 | 42.12 / 41.61 |
| 60 | 125.99 | 124.85 | 31.49 / 31.46 | 38.76 / 38.23 |

Every rate includes the complete IVF file. Both families bracket the 152.14 kbps Q70 cold player carrier. The 112.06 kbps Q40 player carrier is below both families on these points. Recheck support using the final combined rate after adding the ball and verifying receiver assets. The control lowers bytes at comparable box PSNR within each CRF pair, with lower complement PSNR. These broad player boxes include background and leave the ball untargeted. They provide a tested allocation control. A body-and-ball ROI and common perceptual scoring remain necessary for the scientific comparison.

The 24-frame smoke took 4.80 s wall and 12.59 CPU-s. Even charging twice the smoke time per frame for each full point, the projected totals fit the registration. All six encodes and decodes covered the 240 ordered timestamps. The full run took 108.01 s wall and 371.43 CPU-s on the Mac using SVT-AV1 4.0.1. The campus baselines used different executable builds. The [smoke](smoke.json) and [full results](full.json) contain command, tool, input, output, geometry, per-arm timing, and region measurements. The read-back hashes of all eight streams and raw decodes matched after copying to `/Users/manu/Datasets/pointstream-submission-20261118/`. To reproduce, run `run.py` with `PYTHONPATH=.` and the source, conditioning directory, config, fresh output directory and frame count from the report or registered player release.

## Matched-placement foreground ROI extension, registered before its encodes

With both source-only input signals now available, also give SVT-AV1 the same two player boxes and the actual paid ball patch footprint at every sender location, with no receiver-side mask charge for conventional encoder decisions. Hold the source, six-cell experiment's three CRFs, map policy, encoder and decoder fixed. Compare each new map to the same-CRF plain encode by authenticating the saved source and plain stream; verify every resulting frame and measure quality on exactly the same player-box/ball union for both arms. Rerun the 24-frame smoke using the full shot's sender schedule (sender inference used offline context). Use the unused local CPU budget. This tests whether these stronger conventional streams bracket the measured player-input-plus-real-ball transport rate. The signal-derived boxes/patches are forward-looking anchor inputs and diagnostic support. They do not supply the independent scientific foreground reference.

All three matched-placement ROI streams decoded at 240/240 frames. The complete rates and diagnostic quality measurements are below. The native rows were rescored on the same union.

| CRF | Plain / ROI kbps | Union luma PSNR, plain / ROI (dB) | Complement luma PSNR, plain / ROI (dB) |
|---|---:|---:|---:|
| 36 | 637.44 / 564.37 | 38.33 / 38.34 | 44.34 / 43.69 |
| 48 | 303.43 / 288.21 | 34.99 / 34.97 | 42.15 / 41.63 |
| 60 | 125.99 / 124.74 | 31.40 / 31.35 | 38.79 / 38.25 |

Both plain and matched-placement ROI bracket the 184.00 kbps paid player-input and ball carrier. Further comparison needs reconstructed players, quality support and a neural-codec curve in this rate range. [Full matched-placement record](full-player-ball.json) · [Smoke](smoke-player-ball.json)
