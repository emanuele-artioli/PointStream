# Experiments: W1

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md).

### 2026-10-29 — W1: PointStream against every SVT bit allocation (rule)
- Question. At equal total rate, does PointStream reach a pair (whole-frame
  quality, player quality) that SVT-AV1 cannot reach by moving bits between
  players and background with a region-of-interest (ROI) quantizer map? This
  is the fair comparison for a codec whose claim is "spend the bits on what
  the viewer watches"; whole-frame BD-rate alone cannot credit it.
- Point under test: the existing audited hybrid on Al (Alcaraz–Hurkacz shot
  031 from frame 129), job `20261027T211451Z-f6a754eb`: total 22.737962 kbps,
  whole LPIPS 0.275787 / DISTS 0.169612, fixed-foreground LPIPS 0.345876 /
  DISTS 0.252551. No new PointStream output is generated in W1.
- Baseline grid (fixed before any quality is seen). SVT-AV1, the same
  native build, preset 4 and 240 native frames as the existing anchors.
  Resolutions: full (1920×1080) and half (960×540). ROI QP offset inside the
  player region R relative to the rest: 0, −20, −40 (the SVT ROI map; R from
  the same frozen source masks, encoder-only, which is legitimate because the
  offsets are signalled in the paid bitstream). For each (resolution,
  offset), three CRFs chosen on rate alone in a CPU preflight so that two
  bracket 22.74 kbps. That is 18 encodes. The existing plain-SVT points stay.
  Reuse `experiments/tennis/parallel_baselines.py`'s ROI path where it fits.
- Scoring: the same 80 ticks (every third frame), same fixed foreground R,
  same LPIPS/DISTS code and weights as the real-base judge
  (`experiments/tennis/k6b_real_base_judge.py`). Controls: the anchor points
  already scored must reproduce within 1e-4.
- Decision rule. For each policy, interpolate whole and foreground quality
  at 22.737962 kbps linearly in log-rate between its two bracketing points;
  a policy without a bracket is unresolved, never extrapolated. A policy
  *dominates* the hybrid on a metric if its whole score ≤ hybrid whole +
  0.001 AND its foreground score ≤ hybrid foreground + 0.001. Outcome A: no
  policy dominates on LPIPS and none on DISTS — PointStream is
  Pareto-better on Al; next, score the Med window's actual-base hybrid
  (never scored) and run the same grid on Med. Outcome B: some policy
  dominates on either metric — the player route does not beat a fair
  allocation even by its own thesis at this rate; report and stop.
  Also report, as the candidate headline number, the best ROI foreground
  score reachable at the hybrid's whole-frame score (interpolated across
  offsets at the same resolution), for each metric.
- Hypothesis: players move fast and are small; SVT cannot bring their LPIPS
  near 0.35 at ~23 kbps without the background collapsing.
- Competing explanation: R covers few blocks, so a −40 offset at half
  resolution buys player quality near 0.35–0.40 for a few kbps and keeps
  the whole frame near the envelope, dominating the hybrid.
- Budget: CPU encodes through the fleet (each stage ≤45 min, split by
  window/resolution); scoring ≤1 GPU-hour including smokes and failures.
- Jobs: see the entries below (masks `20261029T092537Z-88270f98`; encodes
  half `20261029T092919Z-a9e696a8`, full `20261029T093403Z-0a9c7eba`,
  quarter `20261029T094410Z-40e65a2f`). A first masks submission,
  `20261029T091730Z-c2a7bbd0`, was never published (its snapshot transfer to
  gpu2 timed out after the files had landed; it has no spec and cannot run);
  it was inspected, not replayed, and the masks job was submitted again with
  a reduced snapshot.

### 2026-10-29 — W1 implementation, rate-only preflight and substitution rule (before any run)
- Written before any W1 job ran and before any W1 quality number exists.
  The decision rule, grid and budget above are unchanged. Code:
  `experiments/tennis/w1_roi.py` (ROI map, CRF search, rule),
  `w1_masks.py` (fixed foreground S), `w1_encode.py` (CPU encodes),
  `w1_score.py` (GPU scoring), `w1_specs.py` (job specifications).
- ROI sign. The map is 0 outside and the offset (0, −20, −40) inside every
  64×64 block touched by the frozen source R; offset 0 is the plain encode
  with no map. Same blocks as the older comparator, which instead raised the
  background by +32; ours only adds bits inside R, so a CRF at the ceiling
  can still be lowered by the offset.
- CRF range. SvtAv1EncApp 4.2.0 accepts `--crf` 1–70, not 63. The plain
  half-resolution CRF 63 base already costs 33.685 kbps (40,422 bytes), above
  the 22.74 kbps target, so reaching it needs CRF above 63.
- Rate-only preflight. For each (resolution, offset) a search over integer
  CRFs finds adjacent CRFs whose full 240-frame rates straddle 22.737962 kbps,
  plus the neighbour CRF+2 (else CRF−1). Rate is the complete IVF file bytes
  over 9.6 s, as for the existing anchors. The search is the preflight and its
  encodes are the final ones; no quality is read by the encode job. Up to 8
  encodes per policy. A policy that has no point at or below the target at CRF
  70 (or none above it at CRF 1) is unresolved by the rule: never
  extrapolated, never a dominator.
- Substitution (root decision 2026-10-29). If full resolution is unresolved at
  all three offsets, it is replaced by quarter resolution (480×272 coded,
  scale 4; same offsets, same CRF search), where the existing SVT envelope
  sits at this rate. The preflight rates and the substitution are recorded in
  an entry below before anything is scored.
- Controls. The plain CRF 63 bases of scale 8/4/2 are re-encoded (their bytes
  must equal the existing 12,237 / 20,557 / 40,422) and rescored; their
  per-tick whole and foreground LPIPS/DISTS must equal the hybrid job's
  recorded scores within 1e-4 (the hybrid job's published scores, sha256
  `e346844d…`).
- Foreground S is the real-base judge's: R plus the warp-drawn player pixels,
  rebuilt from the saved K6 window, keypoints and source mattes, checked
  against the K6 window's recorded hashes of R and S on every tick.
- Scoring runs on an RTX A6000 (the controls were scored on one); the job is a
  shared-GPU job, so its timings are cost only.

### 2026-10-29 — W1 deviation: full resolution unresolved, quarter substituted (before any scoring)
- Written after the rate-only encode jobs and before any quality number. CPU
  jobs, no GPU time. Rates are complete IVF file kbps over 9.6 s, all 240
  frames, SvtAv1EncApp 4.2.0 preset 4, keyint −1; every stream passed an
  independent dav1d decode. Controls: the plain CRF 63 bases of scale 8/4/2
  re-encoded to exactly 12,237 / 20,557 / 40,422 bytes, as the existing anchors.
- Full resolution (1920×1080), job `20261029T093403Z-0a9c7eba`, gpu3: even at
  CRF 70 the rate stays above the target: offset 0: 34.15, −20: 37.94, −40:
  38.01 kbps (target 22.737962). All three offsets are unresolved by the rule.
  By the root decision (2026-10-29) full is replaced by quarter resolution
  (480×272 coded, scale 4); offsets and search unchanged.
- Half resolution (960×544), job `20261029T092919Z-a9e696a8`, gpu2, all three
  offsets bracket. Offset 0: CRF 64 28.01, 65 24.64, 66 22.28, 67 20.56.
  Offset −20: 66 25.79, 67 24.19, 68 22.47, 69 21.22. Offset −40: 66 25.85,
  67 24.10, 68 22.48, 69 21.09. Selected: 65–67 (offset 0), 67–69 (−20, −40).
- Quarter resolution (480×272), job `20261029T094410Z-40e65a2f`, gpu3, all three
  bracket. Offset 0: CRF 60 26.76, 61 24.11, 62 20.61, 63 17.13. Offset −20:
  60 30.60, 61 27.69, 62 24.26, 63 20.55, 64 18.70, 65 17.31. Offset −40:
  60 30.57, 61 27.80, 62 24.32, 63 20.59, 64 18.66, 65 17.31. Selected: 61–63
  (offset 0), 62–64 (−20, −40).
- Observation recorded before scoring: offsets −20 and −40 cost the same rate
  within 1% at every CRF and both resolutions (e.g. half CRF 67: 24.19 vs
  24.10). The grid is unchanged; either the encoder saturates the ROI offset
  or the two policies are near-duplicates. The scores will show which; if −40
  equals −20 in quality too, the two are one policy and the ROI curve has
  effectively two points.
- Scored set: 3 controls + 3 CRFs × 3 offsets × {half, quarter} = 21 points on
  the same 80 ticks. Masks job `20261029T092537Z-88270f98` (80 ticks rebuilt,
  every R and S hash equals the K6 window's record).

### 2026-10-29 — W1 addition: secondary re-billed rate (written before any score was read)
- Disclosure. The scoring job `20261029T095649Z-43181442` (gpu3, A6000) had
  already finished its smoke and full stages when this entry was written; its
  outputs were not opened. The addition below was fixed from the root
  message and W2's file, not from a W1 quality number.
- Basis (W2's measurement of the packet content). W2 (`claude/w2-pose-rate`, `docs/experiments/w2-pose-rate-2026-10-29.json`,
  code `experiments/tennis/w2_pose_rate.py`) found an exact (round-trip
  identical, zero keypoint error) re-coding of Al's keypoint packet: key 4,
  4 px, linear interpolation, previous-frame prediction, 2,174 bytes =
  1.811667 kbps against the original 4,236 bytes = 3.530000 kbps. The
  generated players are unchanged, so the hybrid's quality is identical at a
  total rate lower by 1.718333 kbps.
- Primary (as frozen): the rule applies at 22.737962 kbps.
- Secondary (definition corrected by root before any score was read): the
  same rule (linear in log-rate between the two bracketing points, dominance
  within 0.001 on both axes, outcome A or B, best foreground at the hybrid's
  whole-frame score) applies at 22.737962 − (4236 − B)·8/9.6/1000 kbps, with
  the hybrid's four scores unchanged, where B is the byte size of the real
  new keypoint packet (format KPS2) wired into
  `experiments/tennis/k6b_rate_streams.py`. W2's 2,174 bytes (1.811667 kbps)
  is not used for this: it excludes the roughly 416-byte metadata block every
  packet carries. B is pending from root; the secondary decision is computed
  only once B is known (`python -m experiments.tennis.w1_report
  SUMMARY.json B`). A policy whose scored CRF points do not bracket the
  secondary rate is unresolved there; no new encodes are made for it. The
  selected CRFs were chosen for the primary rate, so some policies will be
  unresolved at the secondary rate by construction.
- Full resolution: no retry is needed; its submission already ran (job
  `20261029T093403Z-0a9c7eba`) and is unresolved at all offsets (above).

### 2026-10-29 — W1 outcome at the primary rate: A (no SVT bit allocation reaches the hybrid's pair)
- Jobs and provenance. Scoring job `20261029T095649Z-43181442` (RTX A6000,
  gpu3, shared GPU, torch 2.10.0+cu128, packed environment `32e68872…`),
  code revision `62afc6604223c7ea885a8a59e2cba83156ee0c21` (ready.json
  `git_head`; branch `claude/w1-roi-pareto`). Inputs: encodes half
  `20261029T092919Z-a9e696a8` (published.tar sha256 `daf90628…`), quarter
  `20261029T094410Z-40e65a2f` (`851cf909…`), masks `20261029T092537Z-88270f98`
  (`fbb39eec…`), hybrid scores `e346844d…`; all in the spec. Full-resolution
  preflight `20261029T093403Z-0a9c7eba` (no bracket, above). Encode jobs ran
  at revisions `79663a43` (half `a9e696a8`, full, quarter; code identical to
  `62afc660`, only the specification generator differs) and `0f5ea24b`
  (masks). Scores: 80 ticks, whole and S-foreground LPIPS/DISTS, 21 streams
  (3 controls + 18). Summary sha256 `44d4bbc3…`, published.tar `ff0d2474…`.
- GPU cost. Scoring execution 125.6 s (smoke) + 612.2 s (full) = 737.8 s;
  the request was admitted for 1,169 s from creation to completion, including
  staging; no other GPU job. Charged 1,170 GPU-s of the 3,600 s cap. The
  validator passed (controls, memory, projection 1,126 s). Timings are
  contaminated by the shared GPU and are cost only.
- Controls reproduce. The plain CRF 63 bases of scale 8/4/2 were re-encoded
  (bytes equal the existing anchors), decoded to identical hashes, and their
  per-tick whole and foreground LPIPS/DISTS differ from the hybrid job's
  recorded scores by at most 3.7e-5 (limit 1e-4).
- The 18 points (means over 80 ticks; rate = complete IVF bytes):

| policy | CRF | kbps | whole LPIPS | fg LPIPS | whole DISTS | fg DISTS |
|---|---|---|---|---|---|---|
| half, offset 0 | 65 | 24.642 | 0.1654 | 0.5452 | 0.1240 | 0.4122 |
| half, offset 0 | 66 | 22.284 | 0.1821 | 0.5737 | 0.1290 | 0.4299 |
| half, offset 0 | 67 | 20.557 | 0.1937 | 0.5927 | 0.1411 | 0.4428 |
| half, −20 | 67 | 24.190 | 0.1932 | 0.5893 | 0.1418 | 0.4470 |
| half, −20 | 68 | 22.469 | 0.2125 | 0.6140 | 0.1496 | 0.4661 |
| half, −20 | 69 | 21.218 | 0.2227 | 0.6255 | 0.1563 | 0.4745 |
| half, −40 | 67 | 24.100 | 0.1931 | 0.6009 | 0.1405 | 0.4579 |
| half, −40 | 68 | 22.478 | 0.2104 | 0.6195 | 0.1475 | 0.4737 |
| half, −40 | 69 | 21.088 | 0.2234 | 0.6439 | 0.1581 | 0.4891 |
| quarter, offset 0 | 61 | 24.110 | 0.2627 | 0.6250 | 0.1788 | 0.4332 |
| quarter, offset 0 | 62 | 20.608 | 0.2699 | 0.6489 | 0.1841 | 0.4597 |
| quarter, offset 0 | 63 | 17.131 | 0.2826 | 0.6830 | 0.1927 | 0.4975 |
| quarter, −20 | 62 | 24.262 | 0.2688 | 0.6504 | 0.1844 | 0.4600 |
| quarter, −20 | 63 | 20.549 | 0.2835 | 0.6912 | 0.1923 | 0.5000 |
| quarter, −20 | 64 | 18.695 | 0.2919 | 0.7161 | 0.1987 | 0.5357 |
| quarter, −40 | 62 | 24.316 | 0.2701 | 0.6539 | 0.1851 | 0.4608 |
| quarter, −40 | 63 | 20.591 | 0.2821 | 0.6952 | 0.1915 | 0.5045 |
| quarter, −40 | 64 | 18.661 | 0.2945 | 0.7215 | 0.2018 | 0.5317 |

- Interpolated at 22.737962 kbps (log-rate, bracketing pair), against the
  hybrid (whole LPIPS 0.2758, DISTS 0.1696; foreground LPIPS 0.3459, DISTS
  0.2526). Full resolution: unresolved at all three offsets.

| policy | whole LPIPS | fg LPIPS | whole DISTS | fg DISTS |
|---|---|---|---|---|
| half, 0 | 0.1787 | 0.5680 | 0.1280 | 0.4263 |
| half, −20 | 0.2094 | 0.6100 | 0.1484 | 0.4630 |
| half, −40 | 0.2075 | 0.6164 | 0.1463 | 0.4711 |
| quarter, 0 | 0.2654 | 0.6339 | 0.1808 | 0.4431 |
| quarter, −20 | 0.2745 | 0.6663 | 0.1875 | 0.4756 |
| quarter, −40 | 0.2749 | 0.6706 | 0.1877 | 0.4784 |

- Decision (primary rate, as frozen): **Outcome A.** No resolved policy has
  foreground within 0.001 of the hybrid's on either metric: the best
  foreground of any SVT allocation is LPIPS 0.568 (half, offset 0) and DISTS
  0.426, against the hybrid's 0.346 and 0.253. The half-resolution plain
  encode has a far better whole-frame score (LPIPS 0.179 against 0.276), so
  the two sides trade: SVT wins the whole frame, the hybrid the player
  region, and no SVT policy reaches both.
- Best ROI foreground at the hybrid's whole-frame score (primary rate),
  interpolated across offsets at one resolution without extrapolating: not
  computable. The quarter-resolution offsets span whole LPIPS 0.2654–0.2749
  and DISTS 0.1808–0.1877, all below the hybrid's whole LPIPS 0.2758 and above
  its DISTS 0.1696 only in the quarter case; the half-resolution curves lie
  entirely below the hybrid's whole LPIPS. Nearest points: quarter, −40,
  whole LPIPS 0.2749 with foreground 0.6706; quarter, −40, whole DISTS 0.1877
  with foreground 0.4784. Both are far above the hybrid's 0.346 and 0.253.
- Caveat that limits the strength of A (flagged, not resolved here). The ROI
  offsets bought almost nothing for the players. At matched CRF the offset
  raises the rate about 17% (half CRF 67: 20.56 to 24.19 kbps) but the
  foreground scores hardly move (half CRF 67 foreground LPIPS 0.5927, 0.5893,
  0.6009 for offsets 0, −20, −40); at matched total rate the foreground gets
  worse with the offset, and −20 and −40 are indistinguishable in rate and
  close in quality. SVT-AV1 4.2.0's ROI map in CRF mode therefore seems
  weak or saturating at these CRFs (47–70). A CPU check of the decoded
  quality inside R at a fixed CRF would confirm that the map works as intended
  (not run, not part of the frozen rule). Until then A shows that the tested
  SVT allocations (resolution and these ROI offsets) do not reach the pair,
  not that no SVT allocation could.
- Next step by the rule: outcome A, so score the Med window's actual-base
  hybrid (never scored) and run the same grid on Med. Not started; Med is out
  of W1's scope. The secondary rate is decided in the next entry.

### 2026-10-29 — W1 outcome at the secondary (re-billed) rate: A
- B = 2,701 bytes: the real KPS2 packet for Al, metadata included, decoding
  bit-identically to the KPS1 packet (SHA256 prefix `c1dd0ec56a5b00d1`, branch
  `claude/kps-exact-coder`, from root). Secondary rate =
  22.737962 − (4236 − 2701)·8/9.6/1000 = 21.458796 kbps. The hybrid's four
  scores are unchanged. No new encodes or GPU time; computed from the scored
  points with `python -m experiments.tennis.w1_report score-summary.json 2701`
  (summary sha256 `44d4bbc3…`, job `20261029T095649Z-43181442`).
- All six policies bracket 21.458796 kbps (full resolution is still
  unresolved). Interpolated whole LPIPS / fg LPIPS / whole DISTS / fg DISTS:
  half 0: 0.1875 / 0.5826 / 0.1347 / 0.4359; half −20: 0.2207 / 0.6232 /
  0.1550 / 0.4728; half −40: 0.2198 / 0.6372 / 0.1552 / 0.4849; quarter 0:
  0.2680 / 0.6427 / 0.1828 / 0.4529; quarter −20: 0.2797 / 0.6806 / 0.1902 /
  0.4896; quarter −40: 0.2791 / 0.6849 / 0.1899 / 0.4936. Hybrid: 0.2758 /
  0.3459 / 0.1696 / 0.2526.
- Decision: **Outcome A** again; no policy dominates on LPIPS or DISTS.
- Best ROI foreground at the hybrid's whole-frame score: LPIPS, quarter
  resolution, 0.672 (the hybrid's whole LPIPS 0.2758 lies between quarter −20
  and −40 whole scores 0.2797/0.2791 and offset 0 at 0.2680); half resolution
  and DISTS are outside the policies' whole-score range (not extrapolated).
  Against the hybrid's foreground LPIPS 0.346.

### 2026-10-29 — W1 validity check of the ROI map (CPU, before any re-scoring)
- Why. At matched CRF the offsets raised the rate about 17% and left the
  foreground unchanged, so the ROI baseline might be broken rather than weak.
  CPU job `20261029T124159Z-64b465a7` (gpu2, revision `0e7ba923`, SVT-AV1 4.2.0,
  all 240 frames, no GPU, 35 s): one CRF per resolution (half 67, quarter 62),
  luma PSNR of the decoded stream against the coded source, inside and outside
  R (R resized to the coded size) and inside and outside the 64×64 blocks
  that R touches. R covers 0.56% of the half-resolution pixels, its blocks 5%.
- Map geometry is right. The map read back from the file we wrote equals the
  touched-block grid on every frame (half: 9 rows × 15 columns; quarter: 5 × 8,
  raster order, row 0 on top, frame index 0 = window frame 0, the same
  converted source frames as the judge, whose sha256 is checked). The encoder
  applies it in that geometry: the opposite convention (below) lowers PSNR
  in the background blocks and not in the player blocks, which would not
  happen with transposed or shifted blocks. (Overlay PNGs for ticks 0, 120,
  237 at half and quarter and on the full frame are in the job's
  published.tar; the reading hook here could not open images, so the check
  above is numerical.)
- Result, half resolution, CRF 67 (luma PSNR dB; bytes):

| map | bytes | in R | out of R | in R blocks | out of R blocks |
|---|---|---|---|---|---|
| none | 24,668 | 18.03 | 34.65 | 26.60 | 34.79 |
| −10 inside | 28,933 | 18.00 | 34.64 | 26.61 | 34.77 |
| −20 inside | 29,028 | 17.94 | 34.64 | 26.57 | 34.75 |
| −40 inside | 28,920 | 17.81 | 34.63 | 26.45 | 34.76 |
| −63 inside | 28,687 | 17.62 | 34.63 | 26.28 | 34.76 |
| +20 outside | 26,685 | 17.95 | 33.81 | 26.47 | 33.90 |
| +40 outside | 25,076 | 17.97 | 32.87 | 26.42 | 32.91 |
| −40 inside, aq-mode 0 | 19,100 | 16.86 | 29.38 | 24.93 | 29.38 |

  Quarter resolution, CRF 62, shows the same pattern (none 24,730 bytes,
  in-R 18.76 dB; −40 inside 29,179 bytes, 18.67 dB; +40 outside 26,740
  bytes, background 35.56 dB against 36.71).
- Finding 1, the bug: **negative (inside) offsets are inert in SVT-AV1 4.2.0
  CRF mode with the default aq-mode 2.** They add 4–5 kB per stream
  (map signalling) and change no PSNR anywhere, from −10 to −63. Positive
  offsets outside R do act: the background loses 0.8 dB (+20) and 1.9 dB
  (+40) and the bytes fall with the offset. Our frozen policy is a
  *relative* QP offset between R and the rest ("0, −20, −40 inside R relative
  to the rest"), and +20/+40 outside R with 0 inside is the same relative
  offset in the convention the encoder honours (and the one the older
  comparator used). The −10…−63 inside realisation was an implementation
  error of the sign convention, not a different policy.
- Finding 2, a limit to keep in view: at CRF 62–67 the background at
  CRF + 20 or + 40 meets the encoder's QP ceiling (70), so the relative offset
  is partly clamped. The corrected search therefore places the base CRF lower
  (the CRF in the table is the player-block CRF); the rate-only search finds
  where 22.74 kbps lies.
- Finding 3, separate: the plain SVT-AV1 half-resolution envelope at CRF 64–70
  is much stronger than the CRF 63 anchors used in earlier studies. At
  22.737962 kbps plain half resolution (offset 0, CRF 65–67) scores whole
  LPIPS 0.179 and DISTS 0.128, against the earlier anchor envelope's 0.234 and
  0.165 (and the hybrid's 0.276 and 0.170). The earlier anchors fixed CRF 63
  and varied the resolution, which cannot reach 22.7 kbps at half resolution;
  letting the CRF go to 70 beats the hybrid's whole-frame scores by a wide
  margin. This does not depend on the ROI question.
- Consequence. The outcome-A entries above used inert inside offsets; they
  hold for offset 0 (plain) and say nothing about SVT with working ROI. They
  stand as recorded and are superseded for offsets −20/−40 by the corrected
  run below, under the same frozen rule: relative offsets −20/−40 realised
  as +20/+40 outside R, the same CRF search on rate alone, the same 80
  ticks, controls and scoring. Offset 0 is the same encode and keeps its
  scores; full resolution stays replaced by quarter. The corrected run needs
  about 600 GPU-s (smoke plus 15 streams), within the remaining cap (about
  2,430 GPU-s).

### 2026-10-29 — W1 corrected ROI encodes: rates (before scoring)
- Relative offsets −20/−40 realised as 0 inside the blocks R touches and
  +20/+40 outside (`--convention outside`), rate-only search as before; offset
  0 is unchanged and not re-run. CPU jobs: half `20261029T125703Z-cb255d46`
  (gpu2, 67 s, includes the three controls, bytes equal the anchors again),
  quarter `20261029T131628Z-fb8af87e` (gpu3, 55 s), revision `0d8bd6f5`. Two
  submissions of the quarter job (`20261029T130327Z-52535e47`,
  `20261029T130956Z-0f82a736`) never published (snapshot extraction on gpu2
  timed out twice, 300 s); both were inspected (unpublished, inert), not
  replayed; the third submission is the one above. Further unpublished
  attempt IDs, same cause, are not listed. The first search probes at low
  CRF are far above the target (e.g. 413 kbps) and only place the search.
- Half, offset −20 (CRF: kbps): 65: 26.30, 66: 24.05, 67: 22.24, 68: 20.68,
  70: 18.48; selected 66–68. Offset −40: 64: 27.99, 65: 24.73, 66: 22.38,
  67: 20.90, 69: 18.37; selected 65–67.
- Quarter, offset −20: 60: 29.04, 61: 26.36, 62: 23.05, 63: 19.87, 64: 18.04,
  65: 16.79; selected 62–64. Offset −40: 54: 42.53, 60: 27.79, 61: 25.49,
  62: 22.28, 63: 19.33, 67: 14.85; selected 61–63.
- Recorded before scoring: the rates that bracket 22.74 kbps again sit at the
  same player-block CRF as plain (half 66–67), so the background at CRF + 20
  or + 40 is at the encoder's ceiling (70) and the effective relative offset
  is at most about 4 QP. At this rate SVT has almost no room to move bits
  towards R by raising the background QP. The score will show it.
- Scoring job: staged inputs as before with these two archives
  (published.tar sha256 `5a48c109…` and `65b5ae9e…`) and the first scoring
  summary (offset 0 carried over, sha256 `44d4bbc3…`); 3 controls + 12 new
  streams.

### 2026-10-29 — W1 outcome with the corrected ROI: A stands (ROI is genuinely weak at this rate)
- Scoring job `20261029T134158Z-f343a04e` (RTX A6000, gpu3, shared), revision
  `ffdb7fd6c0353fd48cae4b442ffa0ce71434c4c6` (ready.json `git_head`); inputs:
  the two corrected encode archives (`5a48c109…`, `65b5ae9e…`), the same masks
  and hybrid scores, offset 0 carried from `20261029T095649Z-43181442`
  (summary `44d4bbc3…`). Controls again within 3.7e-5. Summary sha256
  `c4ae5c89…`, published.tar `9b674bfd…`. Execution 69.4 s (smoke) + 576.2 s
  (full) = 645.6 s; request created to completed 1,996 s including
  staging. Cumulative GPU use of the W1 cap: 1,170 + 1,996 = 3,166 s upper
  bound (execution only: 738 + 646 = 1,384 s) of 3,600 s. A submission
  `20261029T133234Z-e386a4f5` never published (inspected, inert); this one
  published although its client call timed out, and was not resubmitted.
- The 12 new points (means over 80 ticks):

| policy | CRF | kbps | whole LPIPS | fg LPIPS | whole DISTS | fg DISTS |
|---|---|---|---|---|---|---|
| half, −20 (+20 outside) | 66 | 24.047 | 0.1946 | 0.5682 | 0.1415 | 0.4298 |
| half, −20 | 67 | 22.238 | 0.2064 | 0.5925 | 0.1460 | 0.4426 |
| half, −20 | 68 | 20.679 | 0.2254 | 0.6190 | 0.1560 | 0.4609 |
| half, −40 (+40 outside) | 65 | 24.731 | 0.1901 | 0.5396 | 0.1377 | 0.4092 |
| half, −40 | 66 | 22.382 | 0.2127 | 0.5818 | 0.1510 | 0.4323 |
| half, −40 | 67 | 20.897 | 0.2280 | 0.5980 | 0.1617 | 0.4457 |
| quarter, −20 | 62 | 23.053 | 0.2730 | 0.6465 | 0.1866 | 0.4569 |
| quarter, −20 | 63 | 19.869 | 0.2894 | 0.6866 | 0.1993 | 0.4982 |
| quarter, −20 | 64 | 18.036 | 0.3027 | 0.7230 | 0.2077 | 0.5268 |
| quarter, −40 | 61 | 25.491 | 0.2718 | 0.6242 | 0.1859 | 0.4332 |
| quarter, −40 | 62 | 22.283 | 0.2817 | 0.6489 | 0.1934 | 0.4573 |
| quarter, −40 | 63 | 19.326 | 0.2969 | 0.6858 | 0.2075 | 0.4956 |

- Interpolated at 22.737962 kbps (whole LPIPS / fg LPIPS / whole DISTS / fg
  DISTS): half 0: 0.1787 / 0.5680 / 0.1280 / 0.4263; half −20: 0.2030 /
  0.5856 / 0.1447 / 0.4390; half −40: 0.2091 / 0.5751 / 0.1489 / 0.4286;
  quarter 0: 0.2654 / 0.6339 / 0.1808 / 0.4431; quarter −20: 0.2745 / 0.6502 /
  0.1878 / 0.4608; quarter −40: 0.2802 / 0.6452 / 0.1923 / 0.4537. Hybrid:
  0.2758 / 0.3459 / 0.1696 / 0.2526. At the re-billed rate 21.458796 kbps
  (B = 2,701): half 0: 0.1875 / 0.5826 / 0.1347 / 0.4359; half −20: 0.2157 /
  0.6055 / 0.1509 / 0.4516; half −40: 0.2221 / 0.5917 / 0.1576 / 0.4405;
  quarter 0: 0.2680 / 0.6427 / 0.1828 / 0.4529; quarter −20: 0.2809 / 0.6659 /
  0.1927 / 0.4768; quarter −40: 0.2858 / 0.6587 / 0.1971 / 0.4674.
- Decision: **Outcome A at both rates; no policy dominates on LPIPS or DISTS.**
  Full resolution is still unresolved (rate above the target at CRF 70).
  Best ROI foreground at the hybrid's whole LPIPS 0.2758, quarter resolution:
  0.649 at the primary rate and 0.657 at the re-billed rate, against the
  hybrid's 0.346; half resolution and DISTS fall outside the whole-score range
  of the policies (not extrapolated); the nearest points are far from 0.253.
- What the working ROI does. At matched rate the offsets move the foreground
  by at most 0.01–0.02 (half −40: foreground LPIPS 0.5751 against 0.5680 plain)
  while the whole frame loses 0.02–0.03. The cause is the encoder's QP ceiling
  (above): at this rate the player-block CRF is 61–67, the background cannot
  go much above 70, and the relative offset reachable is a few QP. So SVT-AV1 has
  almost no room to move bits to the players at 22.7 kbps; the hybrid's pair
  (foreground 0.346) is far outside what any allocation tried reaches.
- Scope of A. It holds for the tested family (plain, and R-block QP offsets
  that the encoder honours) at half and quarter resolution; full resolution
  cannot reach the rate. It does not say that no allocation could (an encoder
  with unrestricted QP or a different tool, e.g. an explicit player-region
  layer, was not tried). The earlier inert-offset results above are kept as
  recorded.
- Local copies for the visual comparison (server-local, not in Datasets):
  `/local/users/emanuele/pointstream/w1-visual` on gpu3 (5.6 GB). Layout:
  `inside/` (first, inert ROI realisation and the plain offset-0 policy) and
  `outside/` (corrected realisation), each with `streams/*.ivf`,
  `decoded/*.yuv` (dav1d output, 4:2:0, coded size) and the encodes tables;
  `overlays/` with the ROI-map overlay PNGs (ticks 0/120/237) and the
  validity-check JSON; `README.txt`. The best plain streams are
  `inside/streams/s2_o0_crf66.ivf` (22.28 kbps, whole LPIPS 0.182) and
  `…/s4_o0_crf61.ivf`; the working-ROI ones are
  `outside/streams/s2_o40_crf66.ivf` (22.38 kbps) and
  `outside/streams/s4_o40_crf62.ivf`. The permanent copies are the encode
  jobs' published.tar archives named above.

### 2026-10-29 — W1 robustness addition: background at the maximum quantizer (written before any such run)
- Why. A reviewer will ask for the strongest SVT allocation: the background
  at the encoder's maximum quantizer and the remaining rate spent on R. The
  corrected ROI above could not do this because in CRF mode with aq-mode 2
  the background already sits near the ceiling at 22.7 kbps and negative
  offsets inside R are inert. Written before any quality number of this family
  exists; the decision rule, dominance margin, 80 ticks, judge and controls
  are unchanged. W1's GPU cap is raised by root to 4,500 s in total.
- Step 1, CPU, mode probe (half resolution, all 240 frames). Modes, in this
  order: `aq0` (`--crf 70 --aq-mode 0`) and `cqp` (`--cqp 70 --aq-mode 0`, fixed
  quantizer). For each, offsets 0, −20 and −40 inside R (the same map as
  before). Pre-registered criterion, computed from luma PSNR against the coded
  source: offset −40 raises the PSNR inside the 64×64 blocks R touches by at
  least 1.0 dB over offset 0, and moves the PSNR outside those blocks by at
  most 0.3 dB. The first mode meeting it is used. If neither does, no mode
  lets SVT raise R's quality at a fixed background; that is recorded and W1
  stops there (outcome A stands as it is).
- Step 2, only if a mode passes. Policy "background at maximum" (`s2_mb`,
  `s4_mb`) at half and quarter resolution: quantizer 70 everywhere, offset
  −(70 − k) inside the touched blocks (k from 7 to 70), k found by the same
  rate-only bracketing search as the CRF: two adjacent k whose full 240-frame
  rates bracket 22.737962 kbps plus one neighbour (the lower-rate side first),
  at most 14 encodes. A policy without a bracket is unresolved, never
  extrapolated. The same three controls are re-encoded and re-scored. Scoring
  on the same 80 ticks with the same judge, the plain, −20 and −40 policies
  carried over from the earlier runs; the rule is applied at 22.737962 kbps
  (primary) and at 21.458796 kbps (re-billed, B = 2,701; a policy whose
  points do not bracket it is unresolved there, no new encodes).
- Dominance as before: whole and foreground within 0.001 of the hybrid's or
  better on LPIPS, or on DISTS. Streams and decodes join those in
  `/local/users/emanuele/pointstream/w1-visual` (a `maxbg/` directory) and
  its README is updated.
- Code: `experiments/tennis/w1_maxbg.py`; scoring reuses `w1_score.py`
  (now accepting a single archive and carrying every earlier policy).

### 2026-10-29 — W1 background-at-maximum: no SVT mode raises R's quality; stopped (outcome A unchanged)
- Job `20261029T181500Z-e57e65a6` (CPU, gpu2, 23 s, all 240 frames, revision
  `836eb4597af15d22bf7da4db477a2cce86a07817`, published.tar sha256
  `31ca7374…`; the validator passed). Half resolution, quantizer 70, offsets
  0 / −20 / −40 inside R in both pre-registered modes. No GPU time used.
- Result (luma PSNR dB inside R / outside R / inside R's blocks / outside
  them; bytes), identical in the two modes (the fixed-quantizer mode and
  CRF 70 with aq-mode 0 give the same streams):

| offset inside R | bytes | in R | out of R | in R blocks | out of R blocks |
|---|---|---|---|---|---|
| 0 | 12,156 | 16.31 | 26.76 | 23.60 | 26.76 |
| −20 | 16,154 | 16.28 | 26.69 | 23.50 | 26.69 |
| −40 | 16,234 | 16.32 | 26.75 | 23.60 | 26.74 |

- The criterion (≥ 1.0 dB gain in R's blocks with the background within
  0.3 dB) fails in both modes: the gain is 0.01 dB. The negative offset
  costs about 4 kB per stream and changes nothing in the decoded picture, as in
  the default mode. Under rule (3) of the root request, no mode of
  SVT-AV1 4.2.0 tried (CRF with aq-mode 2, CRF with aq-mode 0, fixed
  quantizer) lets the encoder raise R's quality relative to a fixed
  background; the only direction the ROI map honours is raising the
  background's quantizer, and at 22.7 kbps that is already at the ceiling
  (above). Step 2 was therefore not run and nothing was scored.
- Consequence for the claim. Outcome A (no SVT allocation reaches the hybrid's
  pair on Al at 22.74 or 21.46 kbps) covers: resolution (half, quarter; full
  unreachable), plain CRF to 70, and background-QP offsets of 20 and 40.
  The remaining untried allocation is outside this encoder's ROI tool, for
  example two separately coded layers (a player-region stream and a
  background stream), which is a different codec design and not an SVT bit
  allocation.
- Visual-comparison files: the three half-resolution probe streams and
  their decodes are in
  `/local/users/emanuele/pointstream/w1-visual/maxbg-probe` beside the other
  W1 files; the README there is updated.
