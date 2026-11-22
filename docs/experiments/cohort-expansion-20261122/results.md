# Official tennis TEST results

The fixed panels contain one 240-native-frame passage from each match. The sender received only the original video. Source-only event identification and marks from the test release are documented in the [cohort card](README.md) and [event audit](test-events.md). Lower DISTS indicates a closer ball crop. Both families pay for the delivered stream.

| Match | Selection | PS / AV1 kbps | PS / AV1 DISTS | Best affordable AV1 kbps / DISTS | Visible hits within 10 px | Emitted / absent marks |
|---|---|---:|---:|---:|---:|---:|
| Munich 2024, Fritz, Draper | target | 20.32 / 20.23 | 0.2612 / 0.2699 | 20.23 / 0.2437 | 43 / 50 | 0 / 0 |
| Shanghai 2023, Sinner, Shelton | other-matched | 20.51 / 20.53 | 0.2455 / 0.2662 | 18.83 / 0.2251 | 36 / 40 | 0 / 1 |
| Gstaad 2022, Gaston, Thiem | other-matched | 39.44 / 39.35 | 0.2599 / 0.2695 | 38.84 / 0.2637 | 16 / 19 | 1 / 1 |
| Sydney, ATP Cup 2022, Shapovalov, Safiullin | other-matched | 33.97 / 34.06 | 0.2351 / 0.2686 | 31.33 / 0.2616 | 18 / 20 | 0 / 0 |
| Halle 2024, Sinner, Marozsan | other-matched | 20.62 / 20.52 | 0.2535 / 0.2762 | 20.46 / 0.2654 | 47 / 49 | 1 / 1 |
| Geneva 2023, O'Connell, Dimitrov | other-matched | 178.96 / 178.17 | 0.1337 / 0.1438 | 178.17 / 0.1438 | 35 / 48 | 0 / 1 |
| Miami 2024, Marozsan, Rune | other-matched | 24.31 / 24.29 | 0.2941 / 0.3310 | 24.29 / 0.3310 | 20 / 20 | 0 / 0 |
| Acapulco 2024, Ruud, Rune | other-matched | 20.06 / 20.04 | 0.2156 / 0.2383 | 20.04 / 0.2383 | 34 / 38 | 1 / 11 |
| Monte-Carlo 2024, de Minaur, Wawrinka | other-matched | 22.44 / 22.55 | 0.3291 / 0.3314 | 22.32 / 0.3284 | 12 / 18 | 0 / 2 |
| Marseille 2020, Herbert, Auger-Aliassime | other-matched | 33.56 / 33.46 | 0.2715 / 0.2783 | 33.46 / 0.2783 | 15 / 21 | 0 / 0 |
| Queen's Club 2024, Draper, Paul | other-matched | 20.08 / 20.05 | 0.2707 / 0.2743 | 20.05 / 0.2743 | 36 / 40 | 1 / 9 |
| Lyon 2024, Evans, Rinderknech | other-matched | 20.25 / 20.25 | 0.2628 / 0.2738 | 19.92 / 0.2684 | 36 / 39 | 7 / 11 |

The target classification requires a 0.5% match to the nominal budget as well as the receiver-to-receiver match. The other-matched classification compares the two delivered streams within 0.5% at another observed price. The sensitivity band permits 0.5–2%. The strongest affordable ordinary anchor is the measured 320- or 640-wide AV1 stream with lowest ball-crop DISTS under the larger cost of each paired comparison. The same-resolution anchor is independently byte-calibrated. The cross-resolution selection uses the held-out outcome and measures an achievable envelope.

| Paired comparison | Eligible matches | DISTS, PS / anchor | Relative decrease | Match wins | 95% interval on decrease |
|---|---:|---:|---:|---:|---:|
| target, same resolution | 1 | 0.2612 / 0.2699 | +3.23% | 1 / 1 | single match |
| target, across resolutions | 1 | 0.2612 / 0.2437 | -7.19% | 0 / 1 | single match |
| other-matched, same resolution | 11 | 0.2520 / 0.2683 | +6.09% | 11 / 11 | +3.69% to +8.57% |
| other-matched, across resolutions | 11 | 0.2520 / 0.2617 | +3.71% | 9 / 11 | +0.45% to +6.93% |
| matched-actual, same resolution | 12 | 0.2527 / 0.2684 | +5.86% | 12 / 12 | +3.62% to +8.22% |
| matched-actual, across resolutions | 12 | 0.2527 / 0.2602 | +2.86% | 9 / 12 | -0.49% to +6.23% |

The source-side selector admitted all 240 ticks on 11 matches, none on 1, and a mixture on 0. Full non-court routing occurred on match119. The unconditional object-mode table preserves all frozen sources, including shots the router would send via conventional video.

Direct positions from the paid stream hit 348 of 402 independently marked visible balls within 10 original pixels. Equal-match mean recall was 85.7%. The receiver emitted a location at 11 of 37 explicitly absent marks. The remaining native frames with no annotation make no contribution to either denominator. On these explicitly annotated ticks, localization precision was 92.6% and F1 was 0.895. An emitted location outside the 10-pixel gate counts as a false positive and misses the labeled ball. The 95% intervals resample complete matches together with their paired outcomes using 10,000 draws.

## Neural video control and receiver-side detector

| Match | DCVC-UF kbps / DISTS | Affordable at paired ceiling | Original detector hits / visible | Source qualified | AV1 / PS / best spatial AV1 detector hits |
|---|---:|---|---:|---|---:|
| Munich 2024 | 16.79 / 0.2485 | yes | 37 / 50 | no | — |
| Shanghai 2023 | 10.54 / 0.2425 | yes | 37 / 40 | yes | 0 / 34 / 0 |
| Gstaad 2022 | 31.83 / 0.2628 | yes | 15 / 19 | no | — |
| Sydney, ATP Cup 2022 | 22.45 / 0.2436 | yes | 18 / 20 | yes | 0 / 19 / 0 |
| Halle 2024 | 12.14 / 0.2808 | yes | 44 / 49 | yes | 0 / 44 / 0 |
| Geneva 2023 | 10.91 / 0.2584 | yes | 36 / 48 | no | — |
| Miami 2024 | 29.38 / 0.2441 | no | 19 / 20 | yes | 0 / 20 / 0 |
| Acapulco 2024 | 16.53 / 0.2504 | yes | 23 / 38 | no | — |
| Monte-Carlo 2024 | 23.29 / 0.2766 | no | 15 / 18 | yes | 0 / 12 / 0 |
| Marseille 2020 | — | pending | — | pending | — |
| Queen's Club 2024 | — | pending | — | pending | — |
| Lyon 2024 | 12.89 / 0.2744 | yes | — | pending | — |

Neural prices refer to the complete fixed 480-wide Q24 bitstream at each native cadence. Source qualification requires a hit within 10 pixels for at least 80% of visible marks on that original source. Detector comparisons on receiver video use only qualifying source matches. Every prematched decoded input passed pixel-hash verification across the scoring and detector hosts. The paid-coordinate outcome uses the full twelve.


### Localization on identical qualified sources

| Receiver readout | Matches | Hits / visible | Emits / explicitly absent | Marked-tick F1 |
|---|---:|---:|---:|---:|
| Original video + WASB | 5 | 133 / 147 | 2 / 4 | 0.920 |
| Matched AV1 + WASB | 5 | 0 / 147 | 0 / 4 | 0.000 |
| PointStream display + WASB | 5 | 129 / 147 | 1 / 4 | 0.905 |
| Direct PointStream coordinates | 5 | 133 / 147 | 1 / 4 | 0.933 |
| Best affordable spatial AV1 + WASB | 5 | 0 / 147 | 0 / 4 | 0.000 |
