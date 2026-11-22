# Ball fidelity under a fixed available bitrate

Each family selects its lowest measured crop DISTS among its affordable encoded files at 320 and 640 pixels wide. The same original set of matches contributes every point of the common-support curve. The two lower-budget columns include only matches for which both measured banks span that budget. A paired bootstrap resamples matches. The final column describes the potential gain from choosing the lower-loss family separately on each source with knowledge of the marks.

| Scope | Budget, kbps | Coverage | AV1 DISTS | PS DISTS | PS − AV1 (95% interval) | PS wins | Oracle gain vs best fixed family |
|---|---:|---:|---:|---:|---:|---:|---:|
| Common support | 35 | 12/12 | 0.2488 | 0.2318 | -0.0170 (-0.0235, -0.0108) | 11/12 | 0.0004 |
| Common support | 60 | 12/12 | 0.2206 | 0.2017 | -0.0189 (-0.0273, -0.0113) | 11/12 | 0.0005 |
| Common support | 100 | 12/12 | 0.2016 | 0.1818 | -0.0198 (-0.0298, -0.0097) | 11/12 | 0.0017 |
| Low-rate eligible | 15 | 7/12 | 0.3031 | 0.3001 | -0.0030 (-0.0175, +0.0171) | 5/7 | 0.0077 |
| Low-rate eligible | 20.4 | 7/12 | 0.2516 | 0.2383 | -0.0133 (-0.0190, -0.0077) | 7/7 | 0.0000 |
| Same low-rate cohort | 15 | 7/12 | 0.3031 | 0.3001 | -0.0030 (-0.0175, +0.0171) | 5/7 | 0.0077 |
| Same low-rate cohort | 20.4 | 7/12 | 0.2516 | 0.2383 | -0.0133 (-0.0190, -0.0077) | 7/7 | 0.0000 |
| Same low-rate cohort | 35 | 7/12 | 0.2298 | 0.2124 | -0.0174 (-0.0239, -0.0118) | 7/7 | 0.0000 |
| Same low-rate cohort | 60 | 7/12 | 0.2045 | 0.1905 | -0.0140 (-0.0236, -0.0054) | 6/7 | 0.0008 |
| Same low-rate cohort | 100 | 7/12 | 0.1950 | 0.1769 | -0.0181 (-0.0256, -0.0118) | 7/7 | 0.0000 |

## Actual selected deliveries

| Available kbps | Matches | Range of selected AV1 kbps | Range of selected PointStream kbps |
|---:|---|---:|---:|
| 35 | match108, match102, match120, match127, match105, match116, match107, match131, match119, match104, match130, match126 | 22.43–34.81 | 23.19–34.91 |
| 60 | match108, match102, match120, match127, match105, match116, match107, match131, match119, match104, match130, match126 | 34.04–59.51 | 30.38–59.85 |
| 100 | match108, match102, match120, match127, match105, match116, match107, match131, match119, match104, match130, match126 | 43.44–99.58 | 44.20–81.79 |
| 15 | match108, match102, match105, match116, match131, match130, match126 | 9.69–14.96 | 10.61–14.98 |
| 20.4 | match108, match102, match105, match116, match131, match130, match126 | 18.83–20.27 | 19.13–20.34 |

## Rate change at equal ball-crop DISTS

| Original match | Common quality range, DISTS | PointStream BD-rate vs AV1 | Rate-domain comparison |
|---|---:|---:|---|
| match108 | 0.1355–0.3468 | -9.15% | crosses the anchor envelope |
| match102 | 0.1482–0.3096 | -21.17% | crosses the anchor envelope |
| match120 | 0.1378–0.3130 | -12.66% | better at every common rate |
| match127 | 0.1342–0.2991 | -38.36% | crosses the anchor envelope |
| match105 | 0.1667–0.3281 | -22.89% | better at every common rate |
| match116 | 0.1438–0.3304 | -11.24% | crosses the anchor envelope |
| match107 | 0.1387–0.2941 | -43.14% | better at every common rate |
| match131 | 0.1504–0.3123 | -36.38% | better at every common rate |
| match119 | 0.1472–0.3291 | -8.65% | crosses the anchor envelope |
| match104 | 0.1626–0.3241 | -19.55% | crosses the anchor envelope |
| match130 | 0.1515–0.3432 | -11.00% | crosses the anchor envelope |
| match126 | 0.1623–0.3323 | -14.65% | better at every common rate |

The geometric mean per-match rate change was **-21.68%** (match-bootstrap 95% interval -28.79% to -14.91%) across 12/12 matches with overlapping fidelity. PointStream required fewer bits on 12/12.

Per-match BD-rate uses monotone piecewise-cubic interpolation over common attainable quality. The anchor is its achievable concave envelope. A negative rate change means fewer delivered bits at equal quality. Full packet identities, selected rates and every original registered mark are in the numerical companion.
