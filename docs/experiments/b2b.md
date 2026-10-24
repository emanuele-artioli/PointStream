# Experiments: B2b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-07 — B2b: fair baselines (timing, equal time, ROI, range, complexity)
- Question: are B2's baselines as strong and as fairly configured as they
  should be before PointStream is compared with them? Code: `cdab5fc` and
  later (`experiments/visor/b2.py` variants, `src/codecs/svtav1.py` CPU
  confinement, DCVC-UF codec-call timing). Every SVT-AV1 encode now runs on
  its worker's own cores (CPU affinity), with wall and CPU seconds; every
  codec is scored on the GPU (B2's CPU scores stay valid). The low-rate
  resolution ladder (PLAN B2b item 3) waits for PointStream's operating
  rates; its points will be chosen by PointStream's rate range alone, never by
  how PointStream scores against them.
- J1, DCVC-UF codec timing and range check. HT-L on the 4 pilot items, 240
  frames, QP 27, 36, 45, 54, 63, each with full-range input (B2's
  configuration, also a reproduction check against B2's streams) and with
  limited-range input (source mapped to 16–235/16–240, output mapped back,
  scored against the same full-range reference); RTX A6000 on gpu3, 8 threads.
  Timing: wall seconds inside the codec calls (network and entropy coding,
  GPU-synchronized), apart from reading, 4:2:0 conversion, hashing and the
  second, verifying decode. Decision rule (range): if limited-range input
  changes DCVC-UF's BD-rate against its full-range self (weighted PSNR, mean
  over the 4 items) by more than 3%, B2's DCVC-UF curves are re-run with the
  better input and the paper reports both; otherwise full range stands.
  Hypothesis: limited range costs DCVC-UF little (within 3%). Competing
  explanation: its training on limited-range YUV makes full range out of
  distribution, so limited range is clearly better.
- J2, SVT-AV1 time per preset. Presets 1, 2, 3, 4, 5, 6, 8, 10, 12 on two
  pilot items (P01_107 EK-100, P02_02 EK-55), CRF 41, 48, 55, 62, one window
  at a time on 32 cores of gpu3 (half its 64; DCVC-UF had one GPU and 8
  cores there), after J1 finishes so the host is not shared with it; scored on
  its A6000. Decision rule (equal time): SVT-AV1's B2 preset becomes the
  slowest preset whose median encode wall time per window is at most
  DCVC-UF HT-L's median codec encode time per window from J1 (same host);
  if that is not preset 4, the B2 SVT-AV1 curves and the ROI run use it, and
  preset 4 stays as a reference. Hypothesis: DCVC-UF's codec time is far
  below B2's 12.5 s, so the fair SVT-AV1 preset is faster than 4 (and worse).
  Competing explanation: DCVC-UF's entropy coding on the CPU dominates and its
  time is near B2's, so preset 4 or slower fits.
- J3, ROI SVT-AV1 pilot. `--roi-map-file` with quantizer offsets 0 (control),
  −16, −32, −64 inside every 64×64 block that touches the VISOR foreground,
  per frame; preset 4, CRF 41, 48, 55, 62, the 4 pilot items, scored on the
  GPU. Decision rule: the offset with the most negative mean BD-rate on
  weighted PSNR against offset 0 is adopted for the full ROI run (all 34
  items, at the equal-time preset); if none is below −3%, ROI is reported as
  not helping. Whole-frame PSNR and VMAF BD-rates are reported beside it,
  since ROI trades background for foreground. Hypothesis: a moderate offset
  (−16 or −32) gains more than 5% on weighted PSNR. Competing explanation:
  64×64 blocks cover so much more than the hands (median foreground 20% of
  the frame) that the shift barely changes the weighting.
- J3 outcome (`20261007T174905Z-c6e9381d`, gpu6, RTX 6000 Ada for scoring,
  `aac0cb4`, 373 s full stage; validator passed): lowering the quantizer in
  foreground blocks does not work. BD-rate against no ROI, mean over the 4
  items, weighted PSNR: −16 → +1.7%, −32 → +5.4%, −64 → +15.1% (whole-frame
  PSNR +3.2, +6.0, +11.7%); at CRF 48, −64 adds 6% rate while foreground PSNR
  falls (37.59 → 37.25 dB). A CPU probe on 16 frames of P01_107 (gpu6, 4
  cores, not a fleet job, not evidence) shows why: the map is honoured
  spatially, since a +64 offset on the left half costs it 1.15 dB and saves 7%,
  but a −64 offset adds 8% bytes and leaves the region's PSNR unchanged
  (41.29 → 41.30 dB), with or without adaptive quantization (`--aq-mode 0`).
  In SVT-AV1 4.2.0's CRF mode, ROI can make a region cheaper, not better. The
  validator's ROI check (any positive mean margin) was too weak to catch this.
- J3b, ROI by raising the background: offsets +16, +32, +64 on every 64×64
  block with no foreground pixel, against 0; CRF 34, 41, 48, 55, 62 (lower
  CRFs keep the curves overlapping as the background loses quality); preset 4,
  the 4 pilot items, GPU scoring. Decision rule as for J3: the offset with the
  most negative mean BD-rate on weighted PSNR against 0, adopted if below −3%,
  for the full ROI run; otherwise ROI is reported as not helping SVT-AV1 here.
  Hypothesis: +16 or +32 gains on weighted PSNR (the foreground keeps its
  quantizer while the background pays). Competing explanation: blocks are so
  coarse that "background" blocks are a minority of what the weights favour,
  so the shift barely helps.
- J3b outcome (`20261007T185948Z-f7d9ed2b`, gpu6, RTX 6000 Ada for scoring,
  `6698a9b`, 466 s full stage; validator passed; a first attempt,
  `20261007T180417Z-38209348` on gpu1, ended `contended` near its end, since
  another user's process kept taking the RTX 8000, and published only
  `partial.tar`): raising the background works as designed: foreground PSNR
  holds (e.g. 37.58–37.60 dB at CRF 48 for every offset) while the background
  and the rate fall. BD-rate against no ROI, mean / median over the 4 items,
  weighted PSNR: +16 → −1.6% / −0.5%, +32 → −3.04% / −1.3%, +64 → −0.8% /
  +2.2%; whole-frame PSNR +3.0, +8.6, +31.2%; VMAF +2.6, +6.0, +19.4%. By the
  rule, +32 is adopted for the full ROI run, but only just: one item
  (P09_106, −10.4%) carries the mean, the others are −1.5 to +0.8%. The full
  run on 34 items decides whether the gain is real. `--enable-tf 0`, preset
  4, CRF 41–62, the 4 pilot items, against B2's pilot streams. Reported, no
  decision attached.
- J5, complexity (secondary, CPU): per frame, VCA v2 (`/opt/local/bin/vca`,
  E and h), ffmpeg `siti`, and Sobel gradient and frame-difference means
  inside and outside the foreground, for all 34 items; correlated (Spearman,
  over items) with B2's per-item BD-rates in `b2 report --complexity`. Run as
  two jobs on gpu6 and gpu1, where B2's SVT-AV1 jobs left those items' videos
  cached. Reported, no decision attached.
- J1 outcome (`20261007T174610Z-05be667d`, gpu3, RTX A6000, `aac0cb4`,
  2,402 s full stage; validator passed). DCVC-UF HT-L codec time per
  240-frame window, median over 20 codings: encode 1.90 s (126 fps), decode
  2.03 s (118 fps), the same at every QP; the worker's whole encode took
  13.1 s, so B2's DCVC-UF timings were about 85% pipeline (conversion,
  reading, hashing). All 20 full-range streams equal B2's pilot streams bit
  for bit. Limited-range input is worse: BD-rate against full range, mean over
  the 4 items, weighted PSNR +5.9%, whole-frame PSNR +5.9%, VMAF +1.5%, worse
  on every item; part of it is the range mapping's own rounding. Full range,
  B2's configuration, stands; no re-run (the rule's change exceeds 3% but in
  favour of what B2 used). The hypothesis is refuted, and not in the
  competing explanation's direction.
- J2 outcome (`20261007T174747Z-8a4e6d3f`, gpu3, 32 cores, one window at a
  time after J1 had finished, RTX A6000 for scoring, `aac0cb4`, 2,477 s full
  stage; validator passed; host load average 7–9 before each encode). Median
  SVT-AV1 encode time per 240-frame window (wall, CPU) and BD-rate against
  preset 4 on weighted PSNR, mean over the 2 items: preset 1 80.5 s (750 s
  CPU), −15.6%; 2 46.2 s, −12.0%; 3 30.3 s, −6.2%; 4 18.7 s (149 s CPU);
  5 12.8 s, +3.2%; 6 9.1 s, +12.3%; 8 4.6 s, +49.7%; 10 2.7 s, +110%;
  12 1.94 s (14.6 s CPU), +133%. DCVC-UF HT-L's codec encode on the same host
  takes 1.90 s, so no tested preset fits the equal-time rule; preset 12 is
  2% over. The rule did not say what happens when none fits, and the choice
  changes the baseline's strength by more than a factor of two in rate, so it
  is put to the user before the full runs (decision below).
- Decision (user, 2026-10-07): equal time stays the rule, but the time to
  match is PointStream's, not DCVC-UF's: PointStream will likely be slower
  than DCVC-UF alone. So the equal-time SVT-AV1 preset is chosen once
  PointStream's encode time per window is measured on the same host class,
  by J2's table (slowest preset at or under that time), and SVT-AV1 is
  re-encoded then. The full ROI run (+32, J3b) waits for the same preset, and
  the low-rate ladder for PointStream's rates, so all three run together.
  Until then B2's preset-4 curves are the reference. gpu5, RTX 6000 Ada for scoring,
  `aac0cb4`, 149 s full stage; validator passed): without temporal filtering
  SVT-AV1 is clearly worse. BD-rate against B2's pilot (same items and CRFs),
  mean over the 4 items: weighted PSNR +21.5%, whole-frame PSNR +27.6%, VMAF
  +27.1%, worse on every item. It costs the background slightly more than the
  foreground (−0.28 against −0.13 dB at CRF 48), so filtering explains at most
  a small part of DCVC-UF's foreground advantage.
- J5 outcome (`20261007T175147Z-16c6f429` gpu6, 104 s;
  `20261007T175326Z-67c86dcf` gpu1, 150 s; CPU only, `aac0cb4`; validators
  passed). Spearman correlation over the 34 items between B2's BD-rate of
  DCVC-UF against SVT-AV1 and each complexity measure (positive: DCVC-UF does
  relatively worse on more complex content): weighted PSNR, background Sobel
  gradient 0.56, VCA E 0.47, foreground gradient 0.42, background frame
  difference 0.32, foreground frame difference 0.28, VCA h 0.23; whole-frame
  PSNR, background gradient 0.63, VCA E 0.52, background difference 0.46.
  With 34 items, |ρ| above about 0.34 is significant at 5%. The DCVC-UF
  advantage is not explained by how much the foreground moves; it is largest
  on scenes with little spatial detail, especially in the background.
- Budget: J1 ≤ 1.5 h on one A6000; J2 ≤ 2.5 h (32 threads, GPU for scoring
  only); J3 ≤ 1 h and J4 ≤ 0.5 h on any Ada, A6000 or RTX 8000 (scoring
  only); J5 ≤ 1 h CPU. Ceiling 6 GPU-hours and 8 h wall; smokes ≤ 600 s.
