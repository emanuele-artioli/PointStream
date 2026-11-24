# Do the twelve rate-matched ball results agree across metrics?

**Figure and decision.** Report the twelve prospectively enrolled point estimates for DISTS, LPIPS, luma SSIM and RGB PSNR on the same 64-pixel crop at every independently marked visible source instant. Use the existing tight-price ball-plus-video and plain SVT-AV1 pair at the original coded geometry on all twelve matches. Preserve the released visibility denominator and actual rates. Also show the saved whole-display RGB PSNR tradeoff separately. This checks whether the local gain extends beyond DISTS and whether foreground enhancement shifts aligned pixel fidelity. If the metrics reverse, describe the specific endpoint and observed per-match patterns. The larger BD-rate bank and route qualification are separate comparisons.

**Run and promotion.** Start on Shanghai (known source and calibrated decode). Use a six-visible-tick smoke stratified over the original 240-frame source, never an early prefix with few ball marks. Verify original frame, complete packet and authorized decode hashes and the prior ball DISTS samples. Run all registered visible instants only if the smoke passes these checks and its measured 1.5× forecast fits the minute cap. Repeat the exact path on the other eleven matches using saved native packets. No new video encoding, GPU work or reference changes. Forecast at most seven CPU minutes per match including staging, no more than 84 total. Preserve the original measured scope when any arm is unavailable.

Shanghai pilot `20261123T120443Z-9a4a6d8c` was submitted on the original scorer's host. [CPU spec](match102-spec.json) · [Registration](match102-registration.json)

The first Shanghai smoke computed valid matched metrics and a 46.92-second forecast, but its validator emitted checks to stdout instead of the fleet's validation file. No full stage ran. The corrected validator explicitly publishes its checks at the required path. [First terminal receipt](match102-smoke-terminal.json)

The corrected Shanghai attempt is `20261123T121031Z-739155a9`. [Registered job](match102-retry-registration.json)

The corrected Shanghai pair completed. Both complete receiver RGB hashes and all per-tick original DISTS scores match. The lower local DISTS also accompanies lower LPIPS, higher luma SSIM and lower RGB error on the same crops. The full smoke forecast was under one minute and native work used one CPU. Promote the remaining eleven previously enrolled matches using their existing completed CPU score receipts and identical metrics. [Shanghai native result](match102-result.json) · [Terminal receipt](match102-terminal.json)

The overnight run produced eight complete native pairs. Marseille and Miami reached the identity check but stopped at the original combined LPIPS/SSIM assertion. Isolate LPIPS's exact self-check and allow at most two parts per million for float32 SSIM rounding. This does not change how any live score is computed or compared. Halle passed its smoke but remaining wall budget could not support the full reservation. Munich exceeded the initial smoke wall cap under concurrent demand. Give the four missing pairs one revised CPU-only attempt, with the original source, parameters and labels frozen. No new video encoding or GPU time.

On the diagnostic retry, Marseille and Miami returned exact LPIPS and MSE identities. Their float32 luma self-SSIM values were 0.99998820 and 0.99999535. Admit an identity roundoff tolerance of 0.00002 and preserve the exact zero checks for LPIPS and MSE. This recovery changes only an assertion, leaving all scored pixels, regions and metric arithmetic intact. Run the two established CPU cases once more.

## Completed twelve-match comparison

All twelve matches and 402 independently marked visible instants were scored. Each stream kept the original physical packet, coded size, source frames and all-frame RGB signature. The two codec rates were within 0.5% in each pair. Numbers below are equal-match means. RGB PSNR transforms pooled squared error over the visible crops within each match before averaging matches. The whole-picture value uses every native tick.

| Endpoint | Ordinary AV1 | PointStream | Matches improved |
|---|---:|---:|---:|
| Ball DISTS | 0.2684 | 0.2527 | 12/12 |
| Ball LPIPS | 0.2621 | 0.1992 | 12/12 |
| Ball luma SSIM | 0.7801 | 0.7856 | 10/12 |
| Ball RGB PSNR | 24.469 dB | 24.430 dB | 7/12 |
| Whole-picture RGB PSNR | 24.658 dB | 24.394 dB | 1/12 |

The paired match bootstrap interval for ball RGB PSNR difference is −0.181 to +0.072 dB. Rendering the ball can recover local perceptual features while making small aligned errors in the streak or its background. The unweighted complete frame reflects the bits spent on the rest of the picture. This fixed-resolution control is distinct from an ordinary codec choosing any tested width and from the separate routed variant. [All matches and intervals](summary.json)

From the repository root, run `python docs/experiments/ball-metric-suite-20261123/summarize.py OUTPUT.json docs/experiments/ball-metric-suite-20261123/match*-result.json` to reproduce the aggregate. The registered job receipt for each match identifies its frozen code and hashed data. Use the recheck registration for Marseille and Miami and the retry registration for Munich and Halle. Every result matches the earlier DISTS and whole-display PSNR; the local comparison uses the same common label and rate contract.
