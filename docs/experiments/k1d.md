# Experiments: K1d

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-21 — K1d: K1c's bound with every masked hand fitted (coverage fix)
- Question (user, 2026-10-21, after K1c: "if we still have a chance to win
  on hands, let's try it"). K1c lost on P02_02 and P09_106 because H2 has
  no fit for 40% and 27% of R. On P02_02, 53 hand frames have no fit and
  131 have one. On P09_106 it is 66 and 114. Whole hands were never boxed
  by H1's detector, so H2b's hybrid (same boxes) cannot help. The encoder
  has the hand masks, though. If every masked hand is fitted, does the
  bound pass?
- Change against K1c (nothing else changes: items, warp, judges, hull,
  bars). For every window frame and side whose fill-set class mask is
  non-empty (box sides ≥ 16 px) and that has no H2 WiLoR row, WiLoR (H2's
  `Regressor`, same checkpoint) is run on that mask's bounding box, with
  the side from the mask's class. The fit becomes a row as in `h2.run_visor`
  (rotation, shape, full-frame translation at H2's focal). It joins H2's
  rows, and the item's WiLoR tracks are coded again with H2's 100 ms choice
  (`CodingSet`, so a new row joins its side's track). The mask box includes
  the forearm, so these fits can be worse than H1-boxed ones. That cost is
  charged through the coding error of the track.
- Rate. H2's pooled rate (1.927 kbps for H2's rows) scaled per item by the
  ratio of coded rows after to before. This assumes the new rows cost what
  H2's rows cost on average, and leaves out extra track starts. Both new
  counts are reported, and the hull is read at each item's own rate.
- Decision rule (`k1c.DECISION_K1D`, fixed before any run). This is K1c's
  rule on the covered fits. The bound passes on an item when LPIPS on R is
  below the hull at that item's rate.
  - ≥ 3 of 4: hands get K6b's ladder (PLAN K6b: audit, zero-shot, overfit,
    match fit), with the mask-box fit as the coverage path.
  - < 3: hands are closed for every renderer.
  - LPIPS over S or DISTS on R crossing 3 differently: to the user.
- Hypothesis. Passes on 4 of 4. With full coverage, `warp@uncoded` falls
  to about 0.01 everywhere (as on P01_107 and P03_10). The coding error
  of the new rows keeps P02_02 and P09_106 below their hulls (0.204,
  0.184), as it did at 0.087–0.093 on the fitted items.
- Competing explanation. Mask boxes include the forearm, so WiLoR's fits
  there jitter. H2's Gaussian smoothing then moves the coded joints far
  from the uncoded ones, and the coding error eats the gain. The mean coded
  shift of new rows against H2's rows is reported to show this.
- Budget. One fleet job, the K1c job with WiLoR added (WiLoR verified on
  Ada/A6000 by H3b/H3c2): smoke ≤ 600 s (P01_107, 48 frames), full ≤ 45
  min. Ceiling 1 GPU-hour.
- Job `20261021T200019Z-e6b9c107` (gpu6, RTX 6000 Ada, shared; `1aa1b43`
  from `ready.json`; inputs H3c2's). Smoke passed the fleet validator. The
  full stage took 1,019 s and passes all 59 of `k1c validate`'s checks
  locally, including that under 1% of R is left unfitted on every item
  (0.0% on all four). Report `visor/k1d-2026-10-21/report/k1c-report.json`
  (`6427ea8f…7c68`, on `full/published.tar` `a8249947…1f42`).
- Outcome. On R, each item at its own rate:

  | item | new rows (H2's) | kbps | `warp` LPIPS | hull LPIPS | result | DISTS `warp` / hull | coded shift, H2's rows / new rows |
  |---|---:|---:|---:|---:|---|---:|---:|
  | P01_107 | 6 (474) | 1.95 | **0.087** | 0.257 | saves 96% | 0.070 / 0.222 | 7.4 / 9.3 px |
  | P02_02 | 237 (243) | 3.81 | 0.232 | 0.162 | loses (−214%) | 0.163 / 0.234 | 20.8 / 64.2 px |
  | P03_10 | 15 (416) | 2.00 | **0.083** | 0.153 | saves 96% | 0.097 / 0.201 | 8.9 / 16.1 px |
  | P09_106 | 123 (146) | 3.55 | 0.141 | 0.138 | loses (−3%) | 0.177 / 0.197 | 15.6 / 24.2 px |

  - LPIPS on R: below the hull on **2 of 4**. Over S: 2 of 4. DISTS on R:
    4 of 4. **By the rule, the outcome goes to the user again.**
  - Coverage worked: `warp@uncoded` is ≤ 0.0001 on every item (it was
    0.192 and 0.138). P02_02 improved from 0.272 to 0.232 and P09_106
    from 0.211 to 0.141.
  - The competing explanation holds. The mask-box fits jitter: their coded
    joints move 24–64 px from the uncoded ones. They also corrupt the H2
    rows they join (P02_02's H2 rows moved 10.9 px in K1c and 20.8 px
    here). Doubling the rows doubles the rate, too, so the hull is read at
    3.5–3.8 kbps, where it is lower. P09_106 misses by 0.003; P02_02 by
    0.07.
- Reading. Hands win where the hand is boxed well (P01_107 and P03_10, 96%
  each, at H2's rate). They lose where a good fit is missing, now through
  bad fits instead of no fits. Hypothesis (4 of 4) refuted.
- Scope. As K1c. The rate scaling assumes a new row costs an average H2
  row. Mask boxes include the forearm (the stated limitation).
- Budget: 1,019 s plus the smoke. K1c and K1d together used under 1
  GPU-hour.
