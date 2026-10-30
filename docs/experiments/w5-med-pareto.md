# Experiments: W5

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md), [W1](w1-roi-pareto.md).

### 2026-10-29 — W5: the W1 test on the Medvedev window (rule)
- Question. W1 found no SVT allocation reaching PointStream's (whole-frame,
  player) quality pair on Al. Does the same hold on Med (Medvedev–Struff
  shot 014 from frame 194, 240 frames at 50 fps, 4.8 s), with every bit
  paid as on Al, and against DCVC-UF too?
- Step 1, the hybrid point. Score Med's actual-base hybrid with the frozen
  real-base recipe of job `20261027T211451Z-f6a754eb` (scale 4 paid base,
  saved generator crops, receiver masks; the Med controls already
  reproduced there), on its 80 scored ticks, whole frame and fixed
  foreground R, LPIPS and DISTS. Rate: base + side terms as on Al, with the
  keypoint packet billed as KPS2 (2193 B) for the primary rate and KPS1
  (3542 B) reported beside it.
- Step 2, the anchors. Plain SVT-AV1 and SVT with +20/+40 QP outside R
  (the working convention, W1) at half and quarter resolution, CRFs chosen
  on rate alone to bracket the hybrid's rate, same judge and controls as
  W1. DCVC-UF points come from W3/W3b (Med triple); if they do not bracket
  the hybrid's rate, the DCVC verdict is unresolved, not extrapolated.
- Decision rule: W1's, per anchor family. A policy dominates the hybrid on
  a metric if whole ≤ hybrid + 0.001 AND foreground ≤ hybrid + 0.001 at
  the hybrid's rate (log-rate interpolation between bracketing points
  only). Outcome A: nothing dominates on LPIPS or DISTS. Outcome B: some
  policy dominates on either. A narrowing on B: report which metric and
  family, and the foreground margin left.
- Hypothesis: as on Al; Med's players are larger in the frame, so the
  player gain is larger and the whole-frame cost smaller.
- Competing explanation: at 50 fps the base costs more per second, the
  side share is smaller, and plain SVT at half resolution dominates.
- Budget: ≤1.5 GPU-hours including smokes and failures; CPU encodes via
  the fleet; every stage ≤45 min.
- Jobs: (none yet)

### 2026-10-29 — W5 implementation notes (written before any run; no deviation from the rule above)
- Step 1 reuses the frozen judge unchanged except for one switch, `--no-early-stop`
  (default off, so every earlier invocation behaves as before): job `20261027T211451Z-f6a754eb`
  stopped after Al by the registered arithmetic stop, so Med was never scored. The same spec
  (inputs, hosts reduced to gpu3, RTX A6000, `separated-v1` cost model) is re-run with the switch;
  it scores both windows, so Al doubles as a regression check against the recorded Al means.
  Spec: `docs/experiments/w5-specs/judge.json`.
- Med's hybrid rate = the scale 4 plain base file (20484 B, 34.140 kbps over 4.8 s) + side terms,
  with the keypoint packet as KPS1 (3542 B, 5.903 kbps) in the judge's `rates`; the KPS2 rate
  (primary) subtracts (3542 - 2193) B * 8 / 4.8 s = 2.2483 kbps. Same quality for both (pixels
  unchanged).
- Step 2 reuses W1's code parametrised by `--window med` (`w1_roi.Window`; Al is the default and
  its constants are unchanged): masks S (`w1_masks`), CRF search and encodes (`w1_encode`),
  scoring (`w1_score`). The CRF search brackets the KPS1 rate (the higher one); the usual three
  selected CRFs (bracketing pair + the next-lower-rate neighbour) then also bracket the KPS2
  rate, exactly as W1 did for Al's re-billed rate. First probes: quarter 60, half 70 (half-
  resolution plain CRF 63 is already 80.3 kbps, above the hybrid; if CRF 70 is still above the
  target the half-resolution family is unresolved there, never extrapolated).
- Grid: plain (offset 0, inside = no ROI) and outside-R +20/+40 (W1's convention honoured by
  SVT-AV1 4.2.0 CRF mode; the negative-inside convention was shown inert) at half and quarter
  resolution; the three plain CRF 63 bases of scale 8/4/2 are re-encoded and must match the
  base manifest's bytes (11734/20484/48152) and re-score within 1e-4 of the step 1 archive.
- Rule application: primary rate KPS2, KPS1 beside; hybrid quality from step 1's `quality.json`
  (scale 4 method means). DCVC: judged on W3b's Med triple if present; otherwise pending.
- Jobs (recorded at submission, 2026-10-29): step 1 judge `20261029T192002Z-9f6a6e45` (gpu3, A6000, ready.json git_head `ee3623a4`; client publish timed out, status pending afterwards); masks S `20261029T192450Z-bc8195a6` (CPU, started on gpu2; git_head from ready.json, to be confirmed). Encode and scoring job IDs follow below.
- Masks job `20261029T192450Z-bc8195a6` COMPLETE, validator passed (Med R/S rebuilt and matched the K6 hashes at all 80 ticks). Step 1 job `20261029T192002Z-9f6a6e45` stayed pending: gpu3 had 25 GB held by a foreign process, so the declared 20000 MiB + 8192 MiB sharing margin could not fit. I cancelled it (own job, never started) and resubmit with `gpu_memory_mib` 12000 (actual peak of the earlier job: 3.52 GiB; the in-code cap is unchanged). No scientific content changes.
- Step 1 judge, resubmissions: `20261029T193013Z-1f570343`, `20261029T193850Z-a81e4a31`, `20261029T194652Z-afcf76dd` stayed unpublished (snapshot extraction on gpu2 timed out after 300 s; no spec, no request; no GPU time). Published: **`20261029T224434Z-adf807af`** (gpu3 only, A6000, spec `docs/experiments/w5-specs/judge.json`, 12000 MiB declared).
- Step 1 smoke of `20261029T224434Z-adf807af` PASSED (18 checks, projection 1159 s / 1294 s with publication under 2000 s). Its rates (rate-only, independent of ticks): Med scale 4 total = **42.520872 kbps (KPS1, 3542 B)**, base 34.140 + side 8.381; KPS2 (2193 B) = 42.520872 - 2.248333 = **40.272539 kbps** (primary). Al unchanged at 22.737962. The CRF searches bracket 42.520872; the selected pair + next-lower-rate neighbour then also bracket 40.272539 (as W1 did). The full stage FAILED (state failed, partial.tar only; reason being read next).
- Step 1 full stage of `20261029T224434Z-adf807af` FAILED by the fleet memory guard, not by the code: the job used 15880 MiB (torch peak allocated 3.5 GiB, the rest reserved/context) against my declared 12000 MiB + 1024 (the declaration I had lowered so it would fit beside a foreign 25 GB process). About 814 GPU-s charged (claim 1793316513 to 1793317264 plus 63 s salvage; running total of the 5400 s budget: 814). Fix: declare 16500 MiB and a 4096 MiB sharing margin (needs 20.6 GB free; gpu3 has 23.6 GB beside the foreign process); no code change. Resubmitting; the four encode searches (CPU) are submitted in parallel at the now known rate.
- Status (2026-10-30, stopped by me): NO W5 job is running. After the failed full stage, the step 1 judge (spec now 16500 MiB / 4096 MiB margin) and the four CPU encode searches could not be published: about 25 attempts over about 3 h failed either with "worker gpuN has no fresh heartbeat" (gpu3/gpu6, intermittent) or with the snapshot extraction on gpu2 timing out after 300 s (unpublished inbox dirs, no spec, no GPU time; e.g. `20261030T004049Z-26d9a94a`, `20261030T005046Z-2fd9d48a`, `20261030T010056Z-2016dc06`, `20261030T011052Z-4ef049e4`, earlier `20261029T193013Z-1f570343`, `...193850Z-a81e4a31`, `...194652Z-afcf76dd`, `...234831Z-962535a8`). All unpublished; none to cancel. Spent: 814 of 5400 GPU-s (the failed full stage). Resume needs a working extraction host or a smaller snapshot; helper `docs/experiments/w5-specs/submit_retry.sh`, specs in `docs/experiments/w5-specs/`. Step 2 scoring spec is generated by `w5_specs.score_spec` once the encode archives exist.

- Jobs after the fleet fix (code at merge `3ad7a1cc`, ready.json git_head `3ad7a1cc` for all): step 1 judge `20261030T072024Z-10c9606e` (gpu3, A6000, COMPLETE, validator passed, worker 816 s, archive sha256 `4e2a337d…aa6d`); encodes (CPU, all COMPLETE, validators passed, rate-only): half plain + controls `20261030T072109Z-eaff2f20` (gpu6; controls match 11734/20484/48152 B), quarter plain `20261030T072138Z-4c6ec4e5` (gpu6), half outside +20/+40 `20261030T072250Z-581e1ff7` (gpu6), quarter outside `20261030T072402Z-837f2dac` (gpu2). All 6 searches `complete` (bracketing pairs of adjacent CRFs). Selected CRFs: half plain 68/69/70 (47.7/44.2/39.8 kbps), quarter plain 61/62/63 (43.0/39.3/34.1), half +20 68/69/70, half +40 68/69/70, quarter +20 62/63/64, quarter +40 61/62/63; each triple brackets 42.521 and 40.273.
- Step 1 result (Med hybrid, scale 4 method, 80 ticks; Al regression reproduces the recorded 0.2758/0.3459/0.1696/0.2526): whole LPIPS 0.4020, fg LPIPS 0.3748, whole DISTS 0.1866, fg DISTS 0.2529; its plain base 0.4066/0.6782/0.1992/0.4427. Rates: 42.5209 kbps (KPS1), 40.2725 kbps (KPS2, primary). The player gain over the plain base is large (fg LPIPS -0.30), the whole-frame gain small (-0.005 LPIPS, -0.0126 DISTS).
- Scoring job spec `docs/experiments/w5-specs/score.json` (gpu3 A6000, 80 ticks, step 1 archive as the hybrid input, controls re-scored and compared with the step 1 archive within 1e-4).
