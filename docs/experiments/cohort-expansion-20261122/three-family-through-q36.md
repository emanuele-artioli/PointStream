# Three codec families under the same per-match bitrate ceiling

The available bitrate constrains every match. Rate and DISTS are then averaged with equal weight per match. Every selected point is an actual native stream. The twelve-match table uses the neural indices measured on all twelve. The seven-match table can also use its registered lower-price neural anchor.

| Matches | Available kbps | AV1 delivered / DISTS | PointStream delivered / DISTS | DCVC-UF delivered / DISTS | PS − DCVC DISTS, 95% interval |
|---:|---:|---:|---:|---:|---:|
| 12 | 35 | 29.51 / 0.2488 | 29.76 / 0.2318 | 25.09 / 0.2478 | -0.0160 (-0.0313, -0.0021) |
| 12 | 60 | 45.10 / 0.2206 | 43.96 / 0.2017 | 33.93 / 0.2380 | -0.0363 (-0.0478, -0.0246) |
| 12 | 100 | 64.69 / 0.2016 | 62.16 / 0.1818 | 37.02 / 0.2357 | -0.0539 (-0.0644, -0.0425) |
| 7 | 15 | 13.17 / 0.3031 | 13.00 / 0.3001 | 9.92 / 0.2758 | +0.0243 (+0.0059, +0.0440) |
| 7 | 20.4 | 19.86 / 0.2516 | 19.99 / 0.2383 | 14.96 / 0.2592 | -0.0209 (-0.0272, -0.0148) |
| 7 | 35 | 27.37 / 0.2298 | 28.26 / 0.2124 | 24.53 / 0.2403 | -0.0279 (-0.0462, -0.0127) |
| 7 | 60 | 46.38 / 0.2045 | 42.94 / 0.1905 | 27.58 / 0.2363 | -0.0458 (-0.0587, -0.0345) |
| 7 | 100 | 58.09 / 0.1950 | 58.98 / 0.1769 | 27.58 / 0.2363 | -0.0594 (-0.0677, -0.0494) |

The paired interval resamples whole matches. The table reports a finite tested neural ladder. The companion JSON records the selected quantizer, native byte count, and packet hash for every underlying event and budget.

## Delivered bitrate at equal ball-crop fidelity

Against the DCVC-UF envelope at quantizer indices 24, 36, 12/12 matches had overlapping quality. The geometric mean per-match PointStream BD-rate was -2.95% (95% paired match interval -11.15% to +5.44%). PointStream required fewer bits on 6/12 comparable matches.

Against the DCVC-UF envelope at quantizer indices 12, 24, 36, 7/7 matches had overlapping quality. The geometric mean per-match PointStream BD-rate was +22.67% (95% paired match interval +9.54% to +37.09%). PointStream required fewer bits on 1/7 comparable matches.

Per-event common-quality intervals and rate-domain behavior are in the numerical companion. Each reported rate reduction is bounded by the tested neural operating points.
