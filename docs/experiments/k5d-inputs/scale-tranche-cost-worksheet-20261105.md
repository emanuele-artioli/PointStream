# K5d next-stage cost worksheet — 2026-11-05

**Review only; no job, allocation or full-study release.** The current first-tranche **remaining** envelope is 9,000 GPU / 10,500 workload-CPU seconds, as stated by root from the live ledger. It already reflects prior tranche charges; do not subtract them again. The prospective native PID repair has **180 CPU seconds still to spend**, nested within the 10,500 (50 preparation, 100 one-core campaign, 30 audit). Its full 220-second repair subcap includes the two 20-second capability probes already charged. No GPU is reserved here. Root owns the authoritative ledger and any per-job allocations.

The candidate order is (1) native/PID repair; (2) actual 16-frame producer plus six-point GPU replay/parity on an exclusive RTX A6000, proposed **1,000 GPU / 1,200 CPU seconds inclusive**; (3) four sequential confirmation jobs, preparation and score for game9/Clip5 (292 frames), then preparation and score for game10/Clip12 (586 frames). The 16-frame calibration clip is game10/Clip3 and is not an independent sample. Proceed only after the preceding proof, source bridge, immutable inputs, fresh host checks, inclusive reservation and primary review pass. No result or authorization is assumed.

## Numerical baseline

The original A6000 preparation job `20261105T025241Z-fe451f9e` took 168.218345 and 234.938960 seconds for 16 and 48 frames; coalesced scoring `20261105T032433Z-1423f92a` took 110.028818 and 201.415560 seconds. Its [independent audit](coalesced-score-1423-independent-audit-20261105.json) establishes the stated six-point original-source parity. These are single-development-clip command walls, not CPU consumption or proven upper bounds on other clips. Applying the required 1.5× margin to the linear fixed-plus-frame fit gives:

| Confirmation command | Raw fit, s | 1.5×, s | With illustrative 20 s stage reserve, s |
| --- | ---: | ---: | ---: |
| Prepare 292 | 743.684 | 1,115.525 | 1,135.525 |
| Score 292 | 898.239 | 1,347.359 | 1,367.359 |
| Prepare 586 | 1,356.679 | 2,035.019 | 2,055.019 |
| Score 586 | 1,737.855 | 2,606.783 | 2,626.783 |
| **Four commands** | **4,736.458** | **7,104.686** | **7,184.686** |

The 586 score's padded command plus this example reserve leaves only 43.217 seconds below the 2,670-second full-stage cap; separately prove the readiness, staging, validator and campaign limits. The 16-frame parity pair's original command walls sum to 278.247 seconds, or **417.371 seconds at 1.5×**. The proposed parity 600-second smoke, 800-second campaign and 1,000/1,200 inclusive ceilings are prospective bounds, not measured costs for the changed wrapper. Its cache copies, staging, validator, no-op full stage, publication, namespace postlude, audit and failure reserve all belong **inside** that inclusive 1,000/1,200 ceiling, not added a second time below.

## Four confirmation jobs: a conservative illustrative GPU-reservation scenario

These are accounting assumptions for review, not measured bounds or an approved spec. The eight stages are four readiness smokes and four full commands. To avoid claiming a GPU saving before the fleet's claim lifecycle is measured, this illustration charges all listed elapsed allowances as GPU reservation time, including potentially CPU-only work before or after the claim. The same allowances are **not** measured CPU occupancy. They are additional to the full-command timers in the table.

| Item | Illustrative allowance | GPU s |
| --- | --- | ---: |
| Four full commands, 1.5× | Above | 7,104.686 |
| Full-stage reserve | 4 × 20 s | 80.000 |
| Readiness | 4 × 20 s | 80.000 |
| Asset/environment staging | 8 × 20.357149 s × 1.5 | 244.286 |
| Validators | 4 × 30 s | 120.000 |
| Namespace postlude | 8 × 5 s | 40.000 |
| Publication/sealing | 8 × 15 s | 120.000 |
| Per-job cached source copies | 4 × 30 s | 120.000 |
| Independent external audits | 4 × 40 s | 160.000 |
| Failed-start / unexpected-work reserve | one tranche allowance | 200.000 |
| **Four confirmation jobs and their allowances** | | **8,268.972** |
| **Plus parity inclusive ceiling** | 1,000 GPU s | **9,268.972** |
| **Compared with remaining 9,000 GPU s** | | **268.972 over** |

The staging proxy is the *larger* of the two warm-cache sums from CPU-readiness job `20261105T050105Z-e9104e5e`; its [audit](readiness-probe-e9104e5e-audit-20261105.json) records 20.350204/20.357149 seconds per pass, a 6.370-second full header/label/old-checkpoint readiness command, and a 0.222-second smoke validator. That readiness probe used the old 48-frame checkpoint, no models or image decoding, and cached inputs; it does not establish cold staging or a full-length checkpoint cost. The 20-second readiness allowance exceeds 1.5 × 6.370 seconds but is not an observed production bound. The 30-second source-copy line is a proposed per-job cap, not a measured cost for these jobs. Publication, postlude, audit and failed-start figures are explicit placeholders for review, not guarantees; add or change them only with evidence and without reducing the 1.5 margin. The one-time full-source cache upload belongs to CPU preparation before any GPU claim, not to every confirmation; its latest observed 431.655-second wall and 1.5× plus 30-second extrapolation are not CPU measurements.

The arithmetic does **not** prove 9,000 is impossible: the 1,000 parity reservation may close below its ceiling, and real source copy, validation or external audit may occur outside GPU ownership. It also does **not** demonstrate that 9,000 fits: the scenario is already 268.972 seconds over, source/asset staging may be slower, and full-path timings are not yet measured at 292/586 frames. Even before the illustrative operational lines, 9,000 − 1,000 − 7,104.686 leaves only **895.314 GPU seconds** for all four confirmations' stage reserves and operational costs. Determine actual claim intervals and conservative reservation treatment, then reforecast after parity and each confirmation; do not silently omit a cost or treat elapsed wall as actual GPU use.

## CPU and later full cohort

Within the 10,500 remaining CPU seconds, subtracting the nested 180-second future native repair and the proposed 1,200-second inclusive parity ceiling leaves **9,120 CPU seconds** for the four confirmation jobs, all their staging, namespace work, validation, publication, independent audit and failure reserves. There is no defensible complete CPU estimate yet: old fleet child counters are cumulative, readiness covered only part of the path, real PID-tree teardown/accounting remains unqualified, and no CPU measurements exist for the 292/586 full commands. A one-core wall or the 1.5× GPU estimate is not a measured process-tree CPU total. Stop if reviewed CPU reservations plus the repair and parity caps exceed 10,500, or if qualification or the actual post-parity forecast fails.

The historical **55,619.046 GPU-second** 30-clip illustration already contains both 292/586 clips' four full commands and 60 jobs' readiness/staging/validation assumptions; it is neither a current complete forecast nor an additional charge. If and only if the new full-context confirmation outputs are accepted after exact source, checkpoint, schema, native/wire/pixel, judge and metric joins, count those four commands once: rebuild the remaining 28-clip schedule and its actual outstanding operational work, rather than adding the four-job spend to the untouched 30-job forecast. For orientation only, subtracting their 7,104.686-second numerical fit from the old 51,022.734-second science fit leaves **43,918.048 seconds** of old-fit science; that figure excludes remaining readiness, staging, validators, publication, audit, source transfers, failure reserves and any change in cost model. Do not subtract any operational allowance without identifying the matching already-completed job row. If a confirmation fails or is not admitted, do not credit it as a valid cohort result. All 30 registered clips, including the three historical source failures, remain in the coverage denominator.
