# Experiments: K5d — expanded ball benchmark

### 2026-11-04 — registered preparation and engineering checks

- Goal: complete a real packet/receiver, matched-rate regional-fidelity and
  independently judged task study, including absent frames, beyond the
  48-positive-frame prototype. Main-paper contribution, with supporting controls
  and negative results in the appendix/supplement, within 23+5 pages.
- Hypothesis: spatially coded appearance/motion preserves ball-region fidelity
  and independent task recall more efficiently than spending all bytes on the
  native video. Competing explanations: bright-mark recognition, duplicate or
  misplaced balls, omitted framing, judge/tracker leakage and known-benchmark
  selection. Neither task recall nor crop metrics alone proves realism.
- Evidence population: all 30 full TrackNet games 8–10 clips, 5,675 frames:
  5,472 visible, 197 explicit out-of-frame, 6 occluded. All30 were scored in
  tracker selection; report retrospective benchmark exposure. Deterministic
  calibration uses game9/Clip5 and game10/Clip3. Each original frame appears
  once in the primary benchmark, including old nested prefixes. Source identities,
  label hashes and exact selection are in [k5d-inputs/study-selection-20261104.json](k5d-inputs/study-selection-20261104.json).
  RacketVision tennis test also has selection exposure and sparse labels;
  it does not supply an untouched dense confirmation set. OpenTTGames test
  videos are a possible cross-sport transfer set, subject to exact 240-frame
  context joins and source qualification; prior camera-motion use is disclosed.
- Frozen components: existing BallTrack encoder positions, existing per-frame
  fit/disk/streak primitives and codebooks; no training or post-test tuning.
  WASB receives output images alone, ground truth only enters the separate
  scorer. Its authors report tennis training on games 1–7 and test 8–10;
  optimizer logs were not independently audited. No documented table-tennis
  WASB checkpoint or training split is invented.
- Primary fidelity report: crop DISTS and LPIPS PCHIP BD-rate against the pooled
  achievable native envelope, per clip and each native family, with actual
  transported byte counts. Full-frame PSNR and perceptual metrics remain beside
  the task result. Minimum overlapping spans fixed before inference: 0.005 for
  DISTS/LPIPS and 0.1 dB for PSNR; two distinct valid points are necessary, three
  initial settings per codec. Add a point only where support is missing. No
  extrapolation; disjoint curves yield supported-rate ordering or unresolved
  coverage, never a fabricated BD number. The old +1% full-frame PSNR BD guard
  remains failed on the original four windows and is evaluated/reported under
  that same contract; pointwise PSNR nondecrease is distinct.
- Primary task report: per-clip 10-pixel visible recall at supported equal rates;
  5-pixel recall, missed/wrong positions, trajectory discontinuity and explicitly
  absent-frame false detections beside it. Descriptive success for this new
  task endpoint requires at least 75% of all 30 registered clips to have qualified
  source inference and supported mean recall gain at least 0.10 over the native
  envelope across the fixed common low-rate range; lack of support is not a
  pass. Pooled excess false detections must be no more than 0.05 per explicit
  absent frame, with all 197 negatives retained and per-clip counts shown. This
  is a prespecified benchmark criterion, not population significance or a
  retrospective pass of the old 6/8 whole-contract rule. Unknown/occluded labels
  never become negatives; report source qualification failure and exclusion
  coverage. Frame counts are not independent sample counts. Show three source
  game outcomes and per-clip results; no precise independent-match CI claim.
- Source feasibility gate: WASB recall at least 80% of visible source labels
  within 10 pixels on every qualified clip, all registered frame IDs preserved;
  failures stay in the coverage denominator, no replacement. Cross-sport source
  qualification uses the existing tennis WASB only; failing it stops that
  transfer claim without switching judges after seeing results.
- Controls: on calibration only, at one fixed rate compare native, billed disk
  and streak, and preregistered wrong-position/brightness stress marks with
  explicit diagnostic status and bytes where a coding arm is claimed. Privileged
  source-pixel/label-placement oracles are diagnostics, never deployable coding
  results. PSNR guards on coding arms remain unchanged; diagnostic stress marks
  cannot be relabeled as admissible codecs. A source-centered fixed 64-pixel
  crop is common to all arms on visible labels; independent full-frame judge
  never sees crop coordinates or the transport. Freeze exact control geometry,
  random seeds and reference crop implementation before the pilot outcome.
- Wire/receiver: PSBL v1 retains exact native IVF/PSDC and canonical BALL symbols,
  dimensions, constant rational fps/count, codec/resource ID, lengths and checksum.
  Every physical byte counts. Generic installed decoder/models are identified
  and reported separately, not amortized as a match-specific adaptation.
  The packet parser/render API is CPU-tested; actual native decoder parity,
  packet-only process input and source/label/saved-output file denials must pass
  on real components before source-hidden/full-receiver claims. New framing is
  a different convention from historical payload-only/SVT framing omissions.
- Budget: GPU 0 authorized for this new study. Proposal is part of
  [the staged research envelope](../cards/submission-research-2026-11-04.md),
  20 GPU-h /24 workload CPU-h total, including a 2 GPU-h /3 CPU-h initial feasibility release;
  no allocation is granted by this entry. Old K5c 463 GPU-s /1,413 CPU-s is closed.
  Cheap local engineering checks may use at most 120 CPU-s, zero GPU, under the
  current reversible code/preparation scope; they are no new quality result.
- Local engineering decision: preserve the six actual 48-frame native+BALL
  records read-only, repack/unpack PSBL and verify exact native hash, canonical
  symbols, frame geometry and total byte sum. No source-frame or model inference,
  re-encoding, new quality metric or statistical promotion. Local Homebrew FFmpeg
  is currently broken by a missing libx265.215 dependency; do not modify the
  personal installation or claim fresh native decoding from this check.
- Stops: bad identity/split/frame join, incomplete or nonfinite metrics, failed
  native or packet parity, source/judge exposure violation, denied-path leak,
  unsupported overlap, unfit all-in forecast or exhausted budget. Retain failed
  controls/cost gates and narrow the claim. New methodological choices after
  test results need a new prospective rule; no automatic training or brute force.
- Jobs: none; no new GPU request submitted. Register exact frozen source/spec/
  input/model/tool/device identities before admission and persist the ID before
  one bounded canonical fleet wait. Every waited stage within 45 minutes; split
  the 30-clip benchmark into independently published clip shards and stop when
  the decision is made.
