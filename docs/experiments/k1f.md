# Experiments: K1f

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-21 — K1f: the hand bound's scope on held-out items
- Question (user, 2026-10-21). On the pilot the bound wins big on two
  items (96%) and loses narrowly on the two where H1's detector boxed
  about half of the hand frames. Is "scenes where the detector tracks the
  hands" a scope where PointStream wins? Does it hold on items the scope
  was not chosen from?
- Items. B2's other 30 items, in a content-blind order (sha256 of the id;
  `tools/datasets/k1f_inputs.py`), 240 frames each. Two come from videos
  in G's training footage (P07_110, P22_107). They are reported apart,
  and the clean subset is reported beside the pooled one.
- Bound. K1e's path, unchanged: H2's WiLoR rows plus the two-pass
  hand-box fit for every boxable hand without one, coded with H2's 100 ms
  choice, K1c's warp, judged on R. Rate: K1d's per-item scaling.
- Envelope, coded per item as the pilot's was (`h3d.code_point`): H3b2's
  seven SVT-AV1 low points, DCVC-UF HT-L off the shelf at QP 0 at full
  size and QP 0, 21 and 42 at ½ and ¼, and H3d's G-L network on each ½
  and ¼ point. Points above about 20 kbps are left out. On the pilot, the
  hull with and without them is identical at every rate up to 5 kbps
  (checked on H3d's report). Savings that need a higher rate are capped.
  B2's CRF 41–62 streams are not used (all ≥ 35 kbps).
- Scope, fixed now from the pilot. An item is in scope when H1/H2's
  coverage, H2's rows over the side-frames with a boxable hand mask, is
  ≥ 0.8. On the pilot the two winners have 0.99 and 0.97, and the two
  losers 0.51 and 0.54. The encoder can measure it.
- Decision rule (`k1f.DECISION`, the user's criterion). A group passes
  when the mean over {LPIPS on R, DISTS on R} of the share of its items
  whose bound is below the hull at its rate is ≥ 75%.
  - In scope passes and out of scope fails: the scope is confirmed.
    Headline: on scenes in scope (their share of items and of hand frames
    reported), the hand bound saves a median X% of the envelope's rate at
    about 2 kbps. It is a bound (appearance granted), and K6b's ladder
    decides what a renderer keeps.
  - All items pass: no scope is needed, and the claim is over all items.
  - In scope fails: the scope claim is dropped, and hands' place on K6b's
    ladder goes back to the user.
- Hypothesis. In scope passes (both metrics ≥ 75%). Out of scope sits
  near the bar, as P09_106 does.
- Competing explanations. (1) The pilot's two winners are atypical
  (still, close views), and coverage ≥ 0.8 does not predict a win
  elsewhere. Then in scope fails. (2) Coverage is a proxy for motion. The
  joint speed per item is reported beside to tell the two apart.
- Budget. Parts of 2 items, each a fleet job of ≤ 45 min (the first one
  times an item; a smoke of 48 frames with two DCVC-UF points). Ceiling 8
  GPU-hours (user, 2026-10-21) for all of K1f. If it runs out, the items
  done in the fixed order are reported, and so is the shortfall.
- Attempts (infrastructure, no score read). Part 0
  `20261021T211220Z-3d431acd` (`eca0d32`, gpu6): the smoke passed (159 s),
  then the full stage overran its declared 36,000 MiB at 38,200 (PyTorch's
  cache held across points, with WiLoR and G resident) and was stopped
  before finishing an item. Fixed: the cache is released after each point,
  and 40,000 MiB is declared.
  Part 6 `20261021T220155Z-2e7fdeb3` ran past the 2,700 s stage limit
  (two slow items). Its `partial.tar` (`d34d41fd…7619`) keeps P06_03
  finished and P26_108's envelope. Rerun as part 6b with `--restore`:
  units of the same items and window length, sha256-checked, are kept, and
  only P26_108's bound is computed. Parts 8–10 had not started. They were
  cancelled and resubmitted as one item per job, so no part nears the
  limit. Items and order are unchanged. Part 8 (`20261021T220208Z-fda99e47`)
  had begun its smoke when it was cancelled, and its two items were
  resubmitted the same way.
- Reproduction check (added after the held-out scores were read, before
  they are recorded as evidence). K1f's envelope path has not been shown
  to reproduce H3d's recorded points. One job runs `k1f run` on the pilot
  item P01_107 (same flags, same inputs). Every point it shares with
  H3d's component report (SVT-AV1's seven low points, DCVC-UF's seven,
  G-L's six) must match within 2% in rate on R and 0.003 in LPIPS on R.
  If any point fails, the held-out result is not evidence until the
  difference is explained.
  Job `20261022T013208Z-9e048ecf` (gpu6, `3f530f1`): all 20 points pass,
  18 exactly. G-L at QP 0 ½ differs by 1.05% in rate and 0.0011 in LPIPS,
  and at QP 42 ½ by 0.34% and 0.0001. The bound reproduces K1e's 0.0865.
- Jobs (RTX 6000 Ada, shared, gpu6). Parts 0–5 and 7 (`d1a47c3`), 6b and
  the six single-item parts (`b9e1d58`), each revision from its
  `ready.json`: `20261021T211923Z-08179c83`, `20261021T220120Z-f7cbc6ed`,
  `20261021T220127Z-381c7366`, `20261021T220134Z-c0b4a121`,
  `20261021T220141Z-eff4e8be`, `20261021T220148Z-9e99d6a7`,
  `20261022T000642Z-3f9256ae`, `20261021T220201Z-9337c526`,
  `20261022T000649Z-adac9f11`, `20261022T000656Z-1aea7c30`,
  `20261022T000702Z-d16bf68f`, `20261022T000709Z-f303e32f`,
  `20261022T000742Z-2efab10c`, `20261022T000749Z-e723338c`. Every smoke
  passed the fleet validator, and every full stage passes `k1f validate`
  locally (44 checks for two-item parts, 26 for single, 46 for 6b).
  Report `visor/k1f-2026-10-21/report/k1f-report.json` (`2f6e4291…888e`),
  with the validation receipts in `k1f-validation.tar`. GPU time: 22,488 s
  of stage execution, plus the failed and cancelled attempts and the
  reproduction check, about 7.6 GPU-hours, within the 8.
- Outcome. 22 of the 30 held-out items were run (the budget's
  content-blind order; positions 22–29 were not run). On R, each item at
  its own keypoint rate (1.93–2.72 kbps):
  - **The bound is below the hull on 22 of 22 items by LPIPS and 22 of 22
    by DISTS.** Every group passes the criterion at 100%: all (22), in
    scope (19), out of scope (3), clean of G's training (21), and in
    scope and clean (18).
  - LPIPS on R: the bound's median is 0.084 (0.059–0.194), and the hull's
    at the same rate is 0.175 (0.076–0.410). On 14 items the bound beats
    every pixel point coded (up to about 45 kbps, SVT-AV1 CRF 63 at full
    size), so their saving is censored. On the 8 others the median
    saving is 84% (9–91%). The narrowest margin is P04_24, at 0.075
    against 0.076.
  - Scope. 19 of 22 items (86% of items and of hand frames) have coverage
    ≥ 0.8. The 3 below (0.71–0.73) also win once the hand-box fit fills
    them.
- Decision (by the rule). **All items pass, so no scope is needed. The
  claim is over all held-out items:** with appearance granted, egocentric
  hands sent as 21 coded keypoints (about 2 kbps) beat SVT-AV1, DCVC-UF
  and DCVC-UF + G on the hand region, on every held-out item, on both
  metrics.
- Reading. The pilot was harder than the held-out set. With the
  hand-box path, P02_02 (rubber gloves, jittery fits) and P09_106 lose
  by LPIPS, and on the held-out set nothing does. Pooled with the pilot,
  the bound wins 24 of 26 items by LPIPS and 26 of 26 by DISTS. The
  pilot's coverage predictor was not needed. Coverage is fixed by fitting
  every masked hand, and what is left on P02_02 is fit quality, not
  missing fits.
- Hypothesis. In scope passed as predicted. "Out of scope sits near the
  bar" is refuted: all 3 win clearly.
- Scope and caveats. This is a bound: the hand's own pixels are moved,
  so appearance and outline are granted. K6b's ladder decides what a
  renderer keeps. 22 of 30 held-out items were run. The rate is K1d's
  per-row scaling (new rows assumed to cost an average H2 row). G's
  training overlaps one item (P22_107). Judges are LPIPS and DISTS on R;
  pose fidelity was not judged. Savings are censored above about 45
  kbps.
- Completion (user, 2026-10-22: "yes", about 2.6 more GPU-hours). The 8
  items not run (positions 22–29 of the fixed order) run as single-item
  jobs, unchanged. The result is read over all 30.
  Jobs (`ffd6ccd`, gpu6, RTX 6000 Ada): `20261022T072405Z-ab09c0af`,
  `20261022T072414Z-a0071b24`, `20261022T072430Z-d94d8b6e`,
  `20261022T072439Z-e7c8ed8c`, `20261022T072448Z-b850bbac`,
  `20261022T072458Z-9cc87e86`, `20261022T072508Z-128ba164`,
  `20261022T072522Z-e7c0f8bc`. Each passes all 26 checks locally (9,378
  s together). Report over all 30:
  `visor/k1f-2026-10-21/report/k1f-report-30.json` (`d31b1fc2…0eb8`), with
  `k1f-validation-30.tar`. K1f's total: about 10.2 GPU-hours (8 approved,
  then 2.6 more, user, 2026-10-22).
- **Outcome over all 30 held-out items.** The bound is below the hull on
  **30 of 30 by LPIPS and 30 of 30 by DISTS**, and every group passes at
  100%: all (30), in scope (26), out of scope (4), clean of G (28), and
  in scope and clean (24). LPIPS on R: the bound's median is 0.086, the
  hull's 0.174. On 17 items the bound beats every pixel point coded (up
  to about 45 kbps). On the 13 others the median saving is 83% (10–93%).
  The narrowest margin is still P04_24 (0.075 against 0.076). In scope:
  87% of items and of hand frames. Pooled with the pilot (K1e's path),
  the bound wins 32 of 34 items by LPIPS and 34 of 34 by DISTS.
