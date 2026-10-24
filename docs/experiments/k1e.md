# Experiments: K1e

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-21 — K1e: K1d with hand-only boxes (and why two items fit badly)
- Question (user, 2026-10-21: "if it's cheap, try to box the hand first";
  "can we figure out why 2 pilots get good fit, and 2 don't?"). K1d's
  mask-box fits include the forearm and jitter (coded shift 24–64 px on
  new rows against 7–9 px on H2's). Does fitting on the hand alone remove
  that, and does the bound then pass?
- Change against K1d (nothing else changes). Each mask-box row is fitted
  twice. WiLoR on the mask's box gives a wrist and middle knuckle. H1's
  `wrist_split` cuts the mask there, and WiLoR runs again on the hand
  part's box (the first box if the part is under 16 px a side). The
  second fit is the row.
- Measured beside, from the run's arrays (CPU, in `k1c report`):
  - Jitter: the median frame-to-frame acceleration of the uncoded joints
    (px, mean over 21 joints, consecutive frames of a side), for H2's rows
    and the new rows, for K1d's fits (K1e's first pass, the same box and
    network, saved beside) and for K1e's second fits. This tests the competing explanation
    directly: K1d's new rows should jitter more than H2's, and K1e's less
    than K1d's.
  - Scene predictors per item, measurable at the encoder: the share of the
    hand frames H1's detector boxed (H2's rows over all rows), the mean
    hand-region size, and the median joint speed. They are measured on
    the pilot and named as candidates only. A scope chosen from 4 items
    would have to be tested on items it was not chosen from (B2's other
    30) before any headline uses it.
- Decision rule (fixed before any run). K1c's: the bound passes when
  LPIPS on R is below the hull at the item's rate on ≥ 3 of 4 items. Then
  hands get K6b's ladder with the two-pass fit as the coverage path.
  Otherwise hands are closed, and the outcome goes to the user if LPIPS
  over S or DISTS on R crosses 3 differently.
  - The user's proposed criterion (2026-10-21, made after K1c's and K1d's
    results): pass when the mean over {LPIPS on R, DISTS on R} of the
    share of items below the hull is ≥ 75%. K1c and K1d already meet it
    (2 of 4 and 4 of 4). It is reported beside, and the user decides which
    governs. The concerns: it was proposed after the results, it would
    turn both earlier runs into passes, and DISTS compares texture
    statistics, so it is lenient on the smooth fill and misplaced texture
    that are this bound's failure modes. The paper reports both metrics
    either way.
- **Amendment (user, 2026-10-21, before K1e's results; after K1c's and
  K1d's).** The user's criterion governs K1e and every later run. It
  passes when the mean over {LPIPS on R, DISTS on R} of the share of items
  below the hull is ≥ 75%, so LPIPS must win on at least half the items.
  It also applies **retroactively** to K1c and K1d. That is a post-hoc
  change and the paper must disclose it: both runs failed the rule written
  before them (LPIPS 2 of 4) and pass under this one (2/4 and 4/4, mean
  75%). Consequence: hands go to K6b's ladder. The held-out test (K1f)
  is judged by this criterion, fixed before it runs.
- Hypothesis. The hand-only fits jitter about as little as H2's rows, so
  P09_106 (0.003 short in K1d) passes and P02_02 (0.07 short, rubber
  gloves) still fails: 3 of 4.
- Competing explanations. (1) The jitter comes from the hands themselves
  (blur, gloves, occlusion), not the box, so the second fit jitters as
  much as the first. (2) The rate, not the fit, decides P02_02: doubling
  the rows reads the hull at 3.8 kbps, where it is 0.162.
- Budget. One fleet job, K1d's with a second WiLoR pass (≤ 45 min; K1d
  took 17 min). Ceiling 1 GPU-hour.
- Job `20261021T205023Z-a7bda6e6` (gpu6, RTX 6000 Ada, shared; `be8251e`
  from `ready.json`). Smoke passed the fleet validator. The full stage
  (about 17 min) passes all 59 of `k1c validate`'s checks locally. Report
  `visor/k1e-2026-10-21/report/k1c-report.json` (`7e6f8f69…d769`, on
  `full/published.tar` `c9287130…1524`).
- Outcome (on R, each item at its own rate; LPIPS bound / hull):

  | item | kbps | LPIPS | DISTS | K1c → K1d → K1e LPIPS | joint acceleration, px: H2's rows / new rows, mask box / new rows, hand box |
  |---|---:|---:|---:|---|---:|
  | P01_107 | 1.95 | **0.086** / 0.257 | **0.069** / 0.222 | 0.087 → 0.087 → 0.086 | 5.1 / 9.0 / 9.1 |
  | P02_02 | 3.81 | 0.196 / 0.162 | **0.142** / 0.234 | 0.272 → 0.232 → 0.196 | 20.3 / 56.0 / 40.9 |
  | P03_10 | 2.00 | **0.083** / 0.152 | **0.097** / 0.201 | 0.093 → 0.083 → 0.083 | 6.5 / 28.2 / 28.5 |
  | P09_106 | 3.55 | 0.139 / 0.138 | **0.175** / 0.197 | 0.211 → 0.141 → 0.139 | 8.0 / 20.9 / 20.1 |

  - LPIPS: 2 of 4 below the hull (P09_106 misses by 0.001). DISTS: 4 of 4.
    **The amended criterion passes (mean 75%). The rule written before
    K1e fails (LPIPS 2 of 4).**
  - Jitter. The new rows jitter 2–9× more than H2's rows on the same item.
    The hand-only box cuts P02_02's jitter from 56 to 41 px, and P02_02's
    LPIPS from 0.232 to 0.196. It changes nothing on P01_107, P03_10 or
    P09_106. So the competing explanation (1) holds in part: the jitter
    there comes from the hands, not the box. P02_02's own H2 rows jitter
    20 px, against 5–8 px on the other items (rubber gloves).
  - The hypothesis (3 of 4 by LPIPS) is refuted by 0.001 on P09_106.
- Decision. Under the user's criterion, hands go to K6b's ladder with
  the two-pass fit as the coverage path. K1f, the held-out test, uses the
  same path.
