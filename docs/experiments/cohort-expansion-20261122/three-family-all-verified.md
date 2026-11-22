# Three codec families under the same per-match bitrate ceiling

The available bitrate constrains every match. Rate and DISTS are then averaged with equal weight per match. Every selected point is an actual native stream. The neural envelope includes every verified native score available on each source. The companion records the tested indices per source, including a Q48 pilot on one match.

| Matches | Available kbps | AV1 delivered / DISTS | PointStream delivered / DISTS | DCVC-UF delivered / DISTS | PS − DCVC DISTS, 95% interval |
|---:|---:|---:|---:|---:|---:|
| 12 | 35 | 29.51 / 0.2488 | 29.76 / 0.2318 | 25.16 / 0.2475 | -0.0157 (-0.0305, -0.0020) |
| 12 | 60 | 45.10 / 0.2206 | 43.96 / 0.2017 | 33.69 / 0.2379 | -0.0362 (-0.0477, -0.0245) |
| 12 | 100 | 64.69 / 0.2016 | 62.16 / 0.1818 | 39.45 / 0.2339 | -0.0521 (-0.0635, -0.0403) |
| 7 | 15 | 13.17 / 0.3031 | 13.00 / 0.3001 | 9.92 / 0.2758 | +0.0243 (+0.0059, +0.0440) |
| 7 | 20.4 | 19.86 / 0.2516 | 19.99 / 0.2383 | 19.81 / 0.2467 | -0.0084 (-0.0177, -0.0007) |
| 7 | 35 | 27.37 / 0.2298 | 28.26 / 0.2124 | 24.65 / 0.2398 | -0.0274 (-0.0449, -0.0124) |
| 7 | 60 | 46.38 / 0.2045 | 42.94 / 0.1905 | 27.18 / 0.2361 | -0.0456 (-0.0587, -0.0342) |
| 7 | 100 | 58.09 / 0.1950 | 58.98 / 0.1769 | 31.74 / 0.2332 | -0.0563 (-0.0675, -0.0431) |

The paired interval resamples whole matches. The table reports a finite tested neural ladder. The companion JSON records the selected quantizer, native byte count, and packet hash for every underlying event and budget.

## Delivered bitrate at equal ball-crop fidelity

Against the DCVC-UF envelope at quantizer indices 24, 36, with the additional measured indices listed for each source, 12/12 matches had overlapping quality. The geometric mean per-match PointStream BD-rate was +11.10% (95% paired match interval -0.06% to +23.50%). PointStream required fewer bits on 3/12 comparable matches.

Against the DCVC-UF envelope at quantizer indices 12, 24, 36, plus the source-specific byte-calibrated index, with the additional measured indices listed for each source, 7/7 matches had overlapping quality. The geometric mean per-match PointStream BD-rate was +20.17% (95% paired match interval +6.47% to +36.04%). PointStream required fewer bits on 1/7 comparable matches.

Per-event common-quality intervals and rate-domain behavior are in the numerical companion. Each reported rate reduction is bounded by the tested neural operating points.

## The seven matches at 20.4 kbps

Each cell reports achieved kbps and original-label ball-crop DISTS. The calibration index is the closest affordable price found independently of image quality. The DCVC-UF column uses the strongest affordable measured index, which can differ.

| Match | AV1 | PointStream | DCVC-UF | Calibrated index | Selected DCVC-UF index |
|---|---:|---:|---:|---:|---:|
| match108 | 20.23 / 0.2437 | 20.27 / 0.2388 | 19.68 / 0.2407 | 27 | 27 |
| match102 | 18.83 / 0.2251 | 19.98 / 0.2021 | 19.38 / 0.2219 | 37 | 36 |
| match105 | 19.73 / 0.2717 | 20.34 / 0.2531 | 20.23 / 0.2565 | 34 | 34 |
| match116 | 20.27 / 0.2396 | 20.00 / 0.2333 | 20.24 / 0.2323 | 34 | 34 |
| match131 | 20.04 / 0.2383 | 20.06 / 0.2156 | 20.18 / 0.2474 | 27 | 27 |
| match130 | 20.05 / 0.2743 | 19.13 / 0.2695 | 19.27 / 0.2748 | 27 | 27 |
| match126 | 19.92 / 0.2684 | 20.18 / 0.2558 | 19.72 / 0.2533 | 32 | 32 |
