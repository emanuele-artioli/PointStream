# Do the twelve rate-matched ball results agree across metrics?

**Figure and decision.** Report the twelve prospectively enrolled point estimates for DISTS, LPIPS, luma SSIM and RGB PSNR on the same 64-pixel crop at every independently marked visible source instant. Use the existing tight-price ball-plus-video and plain SVT-AV1 pair at the original coded geometry on all twelve matches. Preserve the released visibility denominator and actual rates. Also show the saved whole-display RGB PSNR tradeoff separately. This checks whether the local gain extends beyond DISTS and whether foreground enhancement shifts aligned pixel fidelity. If the metrics reverse, describe the specific endpoint and observed per-match patterns. The larger BD-rate bank and route qualification are separate comparisons.

**Run and promotion.** Start on Shanghai (known source and calibrated decode). Use a six-visible-tick smoke stratified over the original 240-frame source, never an early prefix with few ball marks. Verify original frame, complete packet and authorized decode hashes and the prior ball DISTS samples. Run all registered visible instants only if the smoke passes these checks and its measured 1.5× forecast fits the minute cap. Repeat the exact path on the other eleven matches using saved native packets. No new video encoding, GPU work or reference changes. Forecast at most seven CPU minutes per match including staging, no more than 84 total. Preserve the original measured scope when any arm is unavailable.

Shanghai pilot `20261123T120443Z-9a4a6d8c` was submitted on the original scorer's host. [CPU spec](match102-spec.json) · [Registration](match102-registration.json)

The first Shanghai smoke computed valid matched metrics and a 46.92-second forecast, but its validator emitted checks to stdout instead of the fleet's validation file. No full stage ran. The corrected validator explicitly publishes its checks at the required path. [First terminal receipt](match102-smoke-terminal.json)

The corrected Shanghai attempt is `20261123T121031Z-739155a9`. [Registered job](match102-retry-registration.json)
