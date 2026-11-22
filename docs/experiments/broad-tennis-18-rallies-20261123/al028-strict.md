# Alcaraz–Hurkacz 028 at the strict 20.4 kbps ceiling

Select the largest paid joint stream within 24,480 bytes from the earlier CRF grid, without looking at its quality. CRF54 cost 24,257 bytes. Compare its 240 actual frames with the scored ordinary and neural controls under the same ceiling on the previously reviewed 12 body times and 11 visible ball times. The original 24,489-byte joint and 24,494-byte Q24 neural miss this ceiling. Keep the 157,517 B appearance initialization as explicit warm state.

The old gpu5 primary attempt `20261122T050913Z-eb4d3664` stopped on a one-level CPU raster difference; same-host gpu3 successor `20261122T051404Z-22c675e2` reproduced all 240 original frame hashes. Restored the original source, portable KPS/reference carrier, fitted ball bytes and 53.5 paid packet from four members of the prior private inputs (3,210,890 bytes). OpenCV 4.10 produced a different clean background. OpenCV 5.0.0 / NumPy 2.0.0 recovered every expected clean-input, 320×192 raw-input, CRF53.5 positive, CRF54 physical packet/IVF and native decode digest with the pinned original SVT-AV1 4.0.1 and dav1d 1.5.4. A fresh private staging archive binds these three new files (8,423,388 compressed bytes).

CPU-only native gpu3 job `20261122T222047Z-8ded33a8` uses the two archived generated RGBs and masks. The price-only target is 20.214167 kbps. Its mandatory 24-tick test checks the original low-rate paid receiver before completing all 240 ticks and preserves one PNG per reviewed time. The study caps this attempt at 15 minutes wall time, zero GPU. [Paid byte grid](../al028-neural-display-20261122/common-budget.json) · [Original roles and CPU profile](al028-strict-spec.json)

The first registration expired without a host because the full declared 680-second allowance exceeded the remaining window. The comparable high-rate replay took 107.44 seconds including its audit. The replacement registration `20261122T222252Z-34de1411` reduces the total cap to 360 seconds, with 24 native smoke frames, 265 seconds maximum for the full stage and the original deadline unchanged. [Bounded job](al028-strict-bounded.json)


The gpu3 CPU receiver finished in 94.91 seconds including its audit. All 240 original control frame hashes matched. The new complete display had RGB PSNR 24.044 dB and the 12 frozen samples reached the evaluator losslessly. The measured full-wire rate is **20.2142 kbps warm**, or 151.4783 kbps if the 157,517 B initialization travels with this 9.6-second rally. [Price and byte identities](al028-strict-carrier.json) · [240-frame receiver](al028-strict-display.json) · [Per-frame foreground scores](al028-strict-scores.json)

| Display | Actual kbps | Player DISTS | Ball DISTS | Equal-class foreground DISTS |
|---|---:|---:|---:|---:|
| PointStream, 320×192 clean court | 20.214 | 0.3037 | 0.2403 | **0.2720** |
| Best affordable ordinary AV1 of 30 | 19.930 | 0.4580 | 0.3309 | 0.3944 |
| Best affordable ROI AV1 of 8 | 20.056 | 0.5010 | 0.3319 | 0.4165 |
| Archived DCVC-UF Q18 producer display | 14.572 | 0.3918 | 0.3243 | 0.3581 |

Table scores use the eleven times with a marked visible ball and two accepted player silhouettes. Both player silhouettes at all twelve times have mean body DISTS 0.3027. The best conventional control in each family was selected on the foreground endpoint among affordable streams; the fixed-price display inventory uses the same bytes-only rule across metrics. The archived Q18 producer result has a scored display. A prior independent replay did not reproduce its native packet. The better validated 20.4117 kbps Q24 stream exceeds the strict cap.

At the common 20.4 kbps ceiling the two equally weighted development rallies (Nadal and Alcaraz) have foreground DISTS **0.2950** against **0.4353** for their respective best tested affordable ordinary AV1 streams, a 32.2% reduction in the mean. The two physical joint rates were 20.362 and 20.214 kbps. [Nadal receiver and controls](nad000-width.md)

## Four measures on the same complete displays

The strongest affordable ordinary foreground-DISTS point used 640×384 CRF62.25. Its 23,916-byte physical stream and all 240 decoded portable native frames reproduced the recorded hashes. Both displays were then scored for all four metrics on the same original source supports. DISTS and LPIPS are better when lower, luma SSIM and RGB PSNR when higher. The eleven common instants give the following means.

| Region | Codec | DISTS | LPIPS | Luma SSIM | RGB PSNR, dB |
|---|---|---:|---:|---:|---:|
| Players | PointStream | 0.3037 | 0.3093 | 0.2422 | 13.595 |
| Players | AV1 | 0.4580 | 0.6098 | 0.3508 | 16.126 |
| Ball | PointStream | 0.2403 | 0.2728 | 0.7236 | 27.547 |
| Ball | AV1 | 0.3309 | 0.5328 | 0.7148 | 25.551 |
| Equal-class mean | PointStream | 0.2720 | 0.2910 | 0.4829 | 20.571 |
| Equal-class mean | AV1 | 0.3944 | 0.5713 | 0.5328 | 20.839 |

Both perceptual metrics favor PointStream for players and ball. Ball reconstructions also improve by 0.0088 luma SSIM and 2.00 dB. Conventional AV1 more closely preserves player intensities as measured by SSIM and PSNR. The PSNR column averages the individual region dB values and differs from whole-frame PSNR. [All paired per-region records and scoring identities](al028-strict-multimetric.json)
