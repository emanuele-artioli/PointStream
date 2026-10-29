# Experiments: W3

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md).

### 2026-10-29 — W3: a deterministic neural baseline on the tennis windows (rule)
- Question. The paper needs a state-of-the-art neural video codec beside
  SVT-AV1. DCVC streams decoded differently across GPUs and with the default
  CUDA stream; only a 24-frame reproduction on gpu3 passed
  ([single-stream report](../single-stream-report-2026-10-28.md)). Can the
  installed DCVC-UF (pinned in
  [resources](../resources.md#neural-video-codec-dcvc-uf), worker
  `src/codecs/dcvc_uf_worker.py`) produce real bitstreams that decode
  exactly in the same job on the same GPU, for Al and Med?
- Design: one job per window on one GPU (gpu3, gpu5 or gpu6; not gpu2,
  whose smokes time out). Encode 240 native frames to bitstream files with
  the native entropy coder, decode those files in the same job, compare the
  decoded frames with the encoder's reconstruction, take the rate from the
  file sizes. Deterministic settings (cuDNN deterministic, benchmark off,
  deterministic algorithms where supported); record which were needed.
  Three operating points (QP, scale) chosen in the smoke on rate alone to
  bracket 15–60 kbps. Score whole and fixed-foreground LPIPS/DISTS on the
  same 80 ticks as W1 with the real-base judge's code.
- Decision rule. The baseline is accepted on a GPU class if, for every
  point, the decoded frames equal the encoder reconstruction (max absolute
  difference ≤1 8-bit level on every frame); record it in the model–GPU
  table of [fleet](../fleet.md). Then report the DCVC envelope against the
  SVT envelope per metric (BD-rate, PCHIP, overlapping quality only;
  otherwise better/worse at every rate), whole frame and foreground. If
  decoding is not exact even within one job, stop and report the
  first diverging frame and kernel settings; no retry with other settings
  without a new entry.
- Hypothesis: within one process on one GPU, with deterministic kernels,
  decoding is exact; earlier failures came from cross-GPU decoding and the
  default CUDA stream.
- Competing explanation: nondeterministic kernels inside the entropy model
  make even same-job decoding diverge.
- Budget: ≤2 GPU-hours including smokes and failures; every stage ≤45 min.
- Jobs: (none yet)

### 2026-10-29 — W3 deviation: what "decoded equals the encoder reconstruction" can mean (root's decision)
- Gap. The rule compares every decoded frame with the encoder's
  reconstruction. DCVC-UF's native P-frame `compress` returns only the bit
  stream and the entropy-coder setting (`dmc_htl_proxy.cpp`: `return
  py::make_tuple(get_encoded_stream(), parallel)`), and `test_video.py`
  measures distortion only from the decode. The encoder-side reconstruction
  exists for the I frame alone (the image model returns `x_hat`). The
  reference picture buffer (DPB) of the video model lives inside the native
  proxy; `clear_dpb` only resets Python attributes that stay `None` in the
  proxy path. So the rule cannot be applied to P frames as written.
- Root's decision (coordinator, 2026-10-29). Replace the acceptance check with:
  (a) the decoded I frame equals the encoder's I reconstruction within 1
  8-bit level; (b) the same bitstream files decoded in two SEPARATE processes
  (each process also decodes twice) inside the same job on the same GPU give
  bit-identical frames on every frame of every point; (c) if the encoder's DPB
  after each P frame is reachable from Python without changing native code,
  also compare it with the decoded frame and report it, otherwise say it was
  not reachable. Accept the baseline only if (a) and (b) hold for every point.
- (c) result. Not reachable: the worker records `p_net.ref_feature is None`
  after the encode of every point (`dpb_reachable: false` in the encode
  report) and no accessor to the proxy's buffer exists in the pinned revision
  `cbdae87`. Encoder/decoder drift on P frames is therefore not measured
  directly; as a diagnostic only, per-frame luma PSNR against the source is
  reported (first and last 8-frame groups).
- Run protocol (frozen here, before any submission). Deterministic settings
  are all switched on from the first run (cuDNN deterministic, benchmark off,
  `torch.use_deterministic_algorithms(True, warn_only=True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, legacy CUDA stream); the report records the
  state DCVC's `set_torch_env` already left, so "which were needed" is stated
  as which were added over upstream, not as an ablation. Operating points per
  window: candidates scale {8, 4, 2} x QP {0, 12, 24, 36, 48, 63}; the first
  25 frames at every candidate give a rate estimate (one I frame plus the mean
  P-frame cost times 239); the points are the distinct candidates whose
  estimated rate is closest in log-rate to 30, 15 and 60 kbps (the smoke codes
  the 30 kbps point only). Rate alone decides. The smoke fails if the chosen
  points do not bracket (lowest estimate <= 20, highest >= 45 kbps). The SVT
  comparator is re-encoded in the job exactly as the original bases: scales
  8, 4, 2, CRF 63, preset 4, 240 frames; Al's rates and scores are checked
  against the audited real-base job `20261027T211451Z-f6a754eb`, and Med's SVT
  bases are scored for the first time. GPU: one exclusive RTX 6000 Ada (the
  only idle GPU; a shared GPU would add contention as a confounder).
- Jobs (2026-10-29): `20261029T091713Z-46254121`, spec `docs/experiments/w3-specs/w3.json`
  (gpu6 only, exclusive RTX 6000 Ada), revision `5d851c5c`. NOT PUBLISHED: the snapshot
  extraction on gpu2 (`tar -xf source.tar`, 569 entries, 7.6 MB) timed out after 300 s
  with 523 entries written; the inbox holds `source/`, `source.tar` and `events/` but no
  request, `ps-fleet status` answers "job is unpublished or unknown". No GPU time spent.
  Not resubmitted; a new submission needs the coordinator's go-ahead (suggested: a
  smaller `--snapshot-path` set or a different extraction host).
- Jobs (second submission, root-approved): `20261029T091713Z-46254121` is unpublished
  and stays so (snapshot extraction timeout on NFS; verified: no request, no spec,
  no live process). New submission with `--source-worktree`, `--snapshot-path src
  --snapshot-path experiments --snapshot-path tools --snapshot-path pyproject.toml`
  (a `git archive` of these paths imports the fleet manager and the workload under
  `python -I`) and `--snapshot-transfer-seconds 240`. ID: (below, once known).
- Jobs (second submission result): `20261029T092633Z-578b35ac`, revision `eba13641`,
  same spec and snapshot set. The tool caps `--snapshot-transfer-seconds` at 90 (240 was
  refused before any transfer). The extraction on gpu2 timed out at 90 s; the inbox
  holds a complete `source/` (250 entries) and `source.tar` but no request, and
  `ps-fleet status` answers "unpublished or unknown". Host load on gpu2/gpu3 was about 34.
  No GPU time spent. No further resubmission without root.
- Jobs (third submission, root-approved option c): `20261029T093021Z-81571456`, published,
  same spec, 250-entry `--snapshot-path` set, default transfer timeout, revision `0f2238f6`
  (HEAD at submission; the code is that of `eba13641`, later commits change docs only).
  The two earlier IDs stay unpublished and untouched.
- Outcome of `20261029T093021Z-81571456` (smoke, gpu6 GPU-ef96fc4f-134e-4c01-290e-9a6c8bbef553,
  RTX 6000 Ada, exclusive; revision `0f2238f6`): FAILED after 52 s of workload (about 75 s of GPU
  claim) with a bookkeeping bug of mine, not a codec result: `code_point` read
  `record['within_process_deterministic']` inside the same `record.update(...)` call that
  creates it (`KeyError`, `w3_neural_baseline.py:205`). Before it, on the real GPU, the source
  conversion and identity check, the 18-candidate rate calibration, the encode of the 30 kbps
  point and both decoder processes had run without error. Nothing was scored; no decode
  verdict exists. Fix: compute the flag first; new test `test_code_point_...` runs `code_point`
  against a stand-in worker (identical decoders accepted; a one-level difference rejected).
  Resubmission needs root's approval (the approved single submission is used).
- Status (2026-10-29): job `20261029T093021Z-81571456` = failed smoke (about 75 GPU-s of 7200,
  bookkeeping `KeyError`, fixed at `367716fd`). Next step: one resubmission, same spec and flags
  (`docs/experiments/w3-specs/w3.json`, `--source-worktree`, `--snapshot-path src experiments
  tools pyproject.toml`, default transfer timeout), approved in principle by root, awaiting a
  session with usage. Nothing submitted since.
- Jobs (fourth submission = the approved resubmission): `20261029T101344Z-9e2e2d37`, gpu6
  GPU-ef96fc4f-134e-4c01-290e-9a6c8bbef553 (RTX 6000 Ada, exclusive), ready.json `git_head`
  `48e4d1f4` (the `code_point` fix of `367716fd` included). The publish call timed out on the
  client side after 120 s but the job had published; its smoke FAILED after 51 s of workload
  (about 70 GPU-s; running total about 145 of 7200).
  - Failure: `foreground_masks` -> `k6b.k6_oracle` draws from reference frames anywhere in the
    window (index 233) but the smoke built the truth RGB for 25 frames only (`IndexError`,
    `w3_neural_baseline.py:265`). Fixed: the truth covers all 240 frames in both stages.
  - What ran before it, on the real GPU (read from the retained scratch on gpu6): the
    18-candidate rate calibration (estimated kbps at 240 frames, Al: scale 8 QP 0..63 = 2.8, 6.2,
    12.0, 19.9, 28.7, 39.0; scale 4 = 9.0, 17.1, 33.4, 55.2, 79.2, 117.6; scale 2 = 20.9, 43.2,
    81.8, 138.7, 214.2, 348.0), so the 15-60 kbps bracket is reachable at scales 8 and 4; the
    30 kbps point was scale 8, QP 48 (5,203 bytes for 25 frames); the point passed the job's
    acceptance code (the run proceeded to the foreground step): decoders A and B, two passes
    each, gave identical frame hashes; the encoder report records `dpb_reachable: false`;
    determinism state: DCVC's `set_torch_env` already sets deterministic algorithms (strict)
    and the cuBLAS workspace config, leaves `cudnn.deterministic` off and benchmark off; this
    job added `cudnn.deterministic = True` (and, wrongly, `warn_only=True`; fixed to keep
    upstream's strict mode); no nondeterminism warning was raised. This is a 25-frame smoke,
    not evidence.
  - SVT scale 8 re-encode of Al: 12,237 bytes = 10.1975 kbps, equal to the audited base.
  - Not yet exercised on a GPU: foreground masks, scoring, the audited score control, the
    validator. Per root's rule no further submission without root.
- Jobs (fifth submission, root-approved): `20261029T123801Z-4da73ccd`, ready.json `git_head`
  `f86d1067` (foreground truth over 240 frames, upstream determinism strictness kept, stand-in tests
  of the foreground, scoring, control and validator tail). The publish call timed out client-side
  but the job published late (pending). Full pytest on the Mac (miniconda env): 12 failures in
  court/segmentation/hook tests that do not touch W3 (cv2 and environment differences, plus the
  known `test_patches_into_models_or_datasets_are_blocked`); all W3 tests pass.
- State (2026-10-29, after a 44 min `ps-fleet wait`): `20261029T123801Z-4da73ccd` is still `pending`,
  never started. gpu6 (the only host in the spec) has both RTX 6000 Ada GPUs busy with other users
  (36-42 GiB used, 99-100 % utilization) and `workers status` shows no fresh heartbeat for gpu6
  (nor gpu5, where both Ada GPUs are idle). The exclusive job waits for an idle GPU; deadline
  2026-10-30T12:00Z. No GPU time spent. Not cancelled, not resubmitted; the options (restart
  workers, widen `hosts` in a new submission, wait) are root's.
- Cancelled (root's option b): `20261029T123801Z-4da73ccd` is `cancelled` per `ps-fleet status`, never
  started (no start time, no GPU time). New submission with `hosts: ["gpu5", "gpu6"]` (still RTX 6000
  Ada, exclusive), otherwise the same spec, snapshot paths and revision of this commit. ID: below.
- Jobs (sixth submission, root option b): `20261029T133843Z-d07fc947`, hosts gpu5 and gpu6, revision
  `4caba61b` (code of `f86d1067`). As before the publish call timed out client-side (outcome
  "unknown"); status is checked, not resubmitted.

### 2026-10-29 — W3 protocol change before any quality is known: finer QP grid for the rate calibration
- Reason. Smoke `20261029T133843Z-d07fc947` (gpu5 GPU-07aa7586-7116-8c7b-1e9c-e5dbdc74b162,
  RTX 6000 Ada, exclusive, revision `4caba61b`; 201.7 s of workload, 196.5 s inside the job) ran
  the whole path: both windows, calibration, the 30 kbps point (Al scale 8 QP 48; Med scale 4 QP 0)
  decoded by two separate processes with identical frames (max decode diff 0, I frame exact), SVT
  scale 8 re-encode of Al = 10.1975 kbps as audited, foreground masks matching the original scoring
  hashes, scoring, the audited score control (Al scale 8 max error 9.3e-5 over the four metrics) and
  the projection (2,236 s against a full allowance now raised to 2,700 s). Its validator failed on
  two flaws of mine, both outside the codec: (1) the bracket check judged the smoke's single
  30 kbps point instead of the three-point plan (now `planned_bracket` on the candidate table);
  (2) with the 6-QP grid Med's plan is 20.4, 25.4, 62.2 kbps, which does not reach 15 kbps. The QP
  grid becomes {0, 6, 12, 18, 24, 30, 36, 42, 48, 56, 63} (33 candidates per window); selection
  rule, targets (15, 30, 60), bracket test (lowest estimate <= 20, highest >= 45) are unchanged.
  Only rates were looked at to decide this (the smoke's one-point quality is a 25-frame smoke).
  Budget spent so far: about 145 + 250 = 400 of 7200 GPU-s. One more submission is needed;
  it is requested from root.
- Jobs (seventh submission, root-approved): `20261029T181050Z-64feb092`, hosts gpu5 and gpu6,
  ready.json `git_head` `32606af7280d60051e4b77d5d36b7f6720389fc3`; published at once.

### 2026-10-29 — W3 outcome: the DCVC-UF baseline is accepted on the RTX 6000 Ada; it beats the SVT bases at every measured quality, but its points sit at 2-16 kbps, below the requested 15-60
- Job `20261029T181050Z-64feb092` (seventh submission; smoke 137 s and full 489 s of workload, both
  complete; validator passed), hosts gpu5 and gpu6, ran the full stage on gpu6, GPU
  `GPU-ef96fc4f-134e-4c01-290e-9a6c8bbef553` (RTX 6000 Ada, capability 8.9, extension `sm89`, torch
  2.10.0+cu128, exclusive, no contention), ready.json `git_head` `32606af7280d60051e4b77d5d36b7f6720389fc3`.
  Packed environment `pointstream-20261009T214002Z` (sha256 `32e68872...4661`); SVT-AV1 4.2.0 and dav1d
  pinned by sha256 in `w3_neural_baseline.py`. GPU-seconds of this job: 626 (smoke 137, full 489);
  all W3 attempts together about 1,000 of the 7,200 allowed. Development windows Al (Alcaraz-Hurkacz
  031@129, 25 fps) and Med (Medvedev-Struff 014@194, 50 fps), 240 native frames, 80 ticks 0:240:3. Not
  untouched footage; DCVC-UF saw none of it in training as far as is known (public checkpoints).
  Raw: `Datasets/pointstream-data/jobs/fleet/inbox/20261029T181050Z-64feb092/full/` (`published.tar`
  holds the six bitstreams, the reports with every frame hash, the foreground masks and `summary.json`;
  `w3-summary.json` is the same summary).
- Decision rule (as amended by the deviation entry above). Accepted on the RTX 6000 Ada: for all six
  points (a) the decoded I frame equals the encoder's I reconstruction (max difference 0), and (b) the
  same file decoded in two separate processes, two passes each, gives bit-identical frames on every
  one of the 240 frames (0 frames differ in any byte; all four frame-hash lists equal). (c) The
  encoder's reference buffer is not reachable from Python (`dpb_reachable: false`), so encoder/decoder
  drift on P frames is not measured; decoded luma PSNR against the source does not decay across the
  240 frames (Al first 8 frames vs last 8: 34.3/31.9, 31.8/30.7, 34.8/32.8 dB; Med 26.5/27.8,
  29.9/30.4, 28.7/29.0; minima 24.7-32.7 dB).
- Settings. Upstream `set_torch_env` already sets deterministic algorithms (strict) and leaves cuDNN
  deterministic off, benchmark off; this job added `cudnn.deterministic = True`, set
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and used the legacy CUDA stream. No nondeterminism warning was
  raised. Which setting was needed is NOT established: no run without them was made, and the earlier
  failures (shared GPUs, other users' processes) are not reproduced here because the GPU was exclusive.
  Whether the 2026-10-16 failures came from contention or from missing settings stays open.
- Points (kbps = container file bytes x 8 x fps / 240; QP higher = more bits; scale 8/4/2 =
  240x136/480x272/960x544):

| Window | Point | kbps | max decode diff | whole LPIPS | whole DISTS | fg LPIPS | fg DISTS |
|---|---|---:|---:|---:|---:|---:|---:|
| Al | DCVC s8 QP30 | 3.600 | 0 | 0.4379 | 0.2903 | 0.7937 | 0.5283 |
| Al | DCVC s2 QP6 | 8.396 | 0 | 0.2086 | 0.1811 | 0.5692 | 0.4311 |
| Al | DCVC s2 QP18 | 16.078 | 0 | 0.1589 | 0.1340 | 0.4491 | 0.3414 |
| Al | SVT s8 CRF63 | 10.197 | - | 0.4509 | 0.2994 | 0.8405 | 0.6096 |
| Al | SVT s4 CRF63 | 17.131 | - | 0.2825 | 0.1927 | 0.6830 | 0.4975 |
| Al | SVT s2 CRF63 | 33.685 | - | 0.1393 | 0.1097 | 0.4753 | 0.3666 |
| Med | DCVC s8 QP6 | 2.330 | 0 | 0.5831 | 0.3794 | 0.8608 | 0.6374 |
| Med | DCVC s8 QP18 | 4.685 | 0 | 0.5683 | 0.3257 | 0.8465 | 0.5699 |
| Med | DCVC s4 QP12 | 10.593 | 0 | 0.4399 | 0.2291 | 0.7004 | 0.4645 |
| Med | SVT s8 CRF63 | 19.557 | - | 0.5594 | 0.3030 | 0.8564 | 0.5538 |
| Med | SVT s4 CRF63 | 34.140 | - | 0.4066 | 0.1992 | 0.6783 | 0.4426 |
| Med | SVT s2 CRF63 | 80.253 | - | 0.2008 | 0.1043 | 0.4227 | 0.3223 |

  SVT rows are re-encodes of the audited bases (scale 8/4/2, CRF 63, preset 4): Al's three rates and
  bytes equal the audited 10.1975/17.130833/33.685 kbps, and the scores equal the audited job
  `20261027T211451Z-f6a754eb` to at most 1.0e-4 (control passed); Med's SVT bases are scored here
  for the first time. Foreground = the fixed scoring region S of the real-base judge (hash-checked
  against the original scoring masks on every tick), whole-frame LPIPS/DISTS with the same code.
- Envelope against the SVT envelope (BD-rate, PCHIP in log rate over the overlapping quality range only,
  anchor = upper concave hull of the three SVT bases; negative = DCVC needs fewer bits at equal quality):

| Window | whole LPIPS | whole DISTS | fg LPIPS | fg DISTS |
|---|---:|---:|---:|---:|
| Al | -66.2 % | -58.3 % | -67.5 % | -67.3 % |
| Med | -64.9 % | -64.8 % | -68.4 % | -67.9 % |

  Al: DCVC is also better at every common rate in 10.2-16.1 kbps on all four metrics. Med: the rate
  ranges do not overlap (DCVC 2.3-10.6, SVT 19.6-80.3 kbps), so only the quality-domain BD-rate
  exists; read it as "at the quality the SVT bases reach, DCVC reaches it at roughly a third of the rate".
- Deviations and cautions, to be read before using the numbers.
  1. The points do not bracket 15-60 kbps (actual 2.3-16.1 kbps). The rate-only selection used a
     calibration estimate from the first 25 frames (one I frame plus the mean P-frame cost), which
     overestimated the 240-frame rate by 3.5-6x (e.g. Al s2 QP18 estimated 58.3, measured 16.1 kbps).
     The selection rule was followed as written; its estimator was biased. Covering 15-60 kbps needs
     higher QPs or finer scales and a second job; not run (not in the rule).
  2. The SVT envelope is thin: three CRF-63 bases at three scales, not a CRF sweep, so DCVC's
     advantage over "the best SVT-AV1" at these rates may be smaller; the same limit holds for every
     earlier W-item comparison. A BD-rate of -65 % is large; the overlap interval is narrow (Al
     LPIPS 0.159-0.438 of DCVC against SVT 0.139-0.451).
  3. Per-window results on two development windows, 80 ticks each; no confidence interval; the
     foreground is the judge's S region, which is smaller than the whole frame.
  4. Encoder/decoder P-frame agreement is untested (see (c)); the claim is "this file decodes
     reproducibly in this job", not "the decoder reproduces the encoder's internal state".
  5. Stored streams do not travel: decoding in a later job or on another GPU class remains untested
     and stays a deployment limit as in docs/fleet.md.
- Consequence for the paper: a neural baseline exists with real bitstreams and reproducible
  decoding on the RTX 6000 Ada, and at these ultralow rates it is far ahead of SVT-AV1 on both whole
  frame and foreground. Any PointStream claim against the two baselines must now beat DCVC, not only
  SVT; W1's comparison and the 15-60 kbps coverage are the next consumers.

### 2026-10-29 — W3b: does DCVC-UF at the hybrid's rate dominate the PointStream hybrid on Al? (rule, written before any new quality is seen)
- Question. W3's accepted DCVC-UF points reached 2.3–16.1 kbps. The PointStream
  hybrid on Al (job `20261027T211451Z-f6a754eb`) is at 22.737962 kbps (whole
  LPIPS 0.275787 / DISTS 0.169612; foreground LPIPS 0.345876 / DISTS 0.252551),
  and 21.458796 kbps when re-billed with the KPS2 keypoint packet. Does any DCVC
  point at those rates dominate it? (W3's Al points: scale 2 QP6 8.40 kbps whole
  LPIPS .209 / fg .569; scale 2 QP18 16.08 kbps whole .159 / fg .449.)
- Rule (W1's, [w1-roi-pareto](w1-roi-pareto.md)). DCVC *dominates* the hybrid on a
  metric (LPIPS or DISTS) at a rate R if its whole score <= hybrid whole + 0.001 AND
  its foreground score <= hybrid foreground + 0.001. Its score at R (22.737962 and
  21.458796 kbps) is interpolated linearly in log-rate between the two bracketing
  DCVC points of one scale; a rate without a bracketing pair is *unresolved*, never
  extrapolated. Verdict per (rate, metric): dominates / does not dominate / unresolved.
- Hypothesis. DCVC's player region at about 22 kbps stays worse than the hybrid's 0.346
  foreground LPIPS because its players are blurred (W3: fg LPIPS .449 at 16 kbps).
- Competing explanation. DCVC at about 22 kbps reaches player LPIPS <= 0.35 with whole
  LPIPS <= 0.18 and dominates; the hybrid is then not Pareto-better against a neural
  codec on Al.
- Budget. <= 1 GPU-hour (3,600 s) including smoke and failures; W3's whole job needed 626 s.
  RTX 6000 Ada, hosts gpu5 and gpu6, exclusive. Same job, validator and acceptance checks as
  W3 (I frame within one level; two separate decoder processes, two passes each, bit-identical
  on every frame; reference buffer status recorded); only the point list changes, and the
  25-frame rate calibration is skipped because the points are named. SVT scale 8 stays as the
  scoring control (Al must reproduce the audited job within 1e-3). Frame count 240, ticks
  0:240:3, development windows, the real-base judge's code, as W3.
- Point selection (rate alone, from W3's MEASURED 240-frame rates; the 25-frame estimator
  overshot 3.6–6x and is used only as a within-scale shape). In each window and scale the
  measured/estimated ratio is stable across QP: Al scale 2 QP6 8.396/30.1 = 0.279 and QP18
  16.078/58.3 = 0.276 (mean 0.277); Med scale 4 QP12 10.593/62.2 = 0.170 (Med scale 8
  drifts 0.178 -> 0.166 from QP6 to QP18). Predicted rate = ratio x the calibration estimate
  (log-linear in QP between the 6-QP grid): 
  - Al, scale 2 (scale 1 is not needed: scale 2 reaches the target and the worker has no
    scale-1 path): QP 21, 23, 24, 25, 27, 30 -> predicted 19.2, 21.4, 22.7, 23.8, 26.3, 30.4 kbps.
    At least one point below 21 (QP21) and one above 23 (QP25, 27, 30); W3's QP18 (16.08,
    measured) is a third lower anchor. Choosing this many costs about 25 s each.
  - Med, scale 4 (scale 8 tops out near 15 kbps at QP63): QP 15, 36, 63 -> predicted 12.9,
    27.4, 52.1 kbps, a triple bracketing 15–40 (the high end carries the larger ratio-drift
    risk; Med is secondary to the question).
  No quality was looked at to choose these; if a bracket fails in the measured rates the Al
  verdict at that rate is "unresolved" and the outcome entry says so.
- Second analysis (same entry, rule fixed here). DCVC's whole and foreground BD-rate against
  W1's stronger SVT envelope (the upper concave hull of the plain half- and quarter-resolution
  CRF points scored by W1's scoring jobs) instead of the three CRF-63 bases, reported only for
  metrics where the quality ranges overlap, otherwise "better/worse at every rate" as AGENTS.md
  requires. Inputs are the W3 and W3b DCVC points (all measured, decoded exactly).
- Jobs: `20261029T190006Z-d48a344b` (hosts gpu5 and gpu6, exclusive RTX 6000 Ada; spec `docs/experiments/w3-specs/w3b.json`; submitted 2026-10-29T19:00Z; ready.json `git_head` `1e10451a` per the submit output, to be confirmed from ready.json; the publish call returned normally; started on gpu5 GPU-07aa7586-7116-8c7b-1e9c-e5dbdc74b162).

### 2026-10-29 — W3b outcome of `20261029T190006Z-d48a344b`: terminated by GPU contention before any quality was scored; the rule is not yet applied (stopped for root)
- Job `20261029T190006Z-d48a344b`, ready.json `git_head` `1e10451ad961655711f1ef1ee3535705f9489ef4`
  (the revision of the spec and the named point list), gpu5 GPU-07aa7586-7116-8c7b-1e9c-e5dbdc74b162
  (RTX 6000 Ada, exclusive claim). The smoke passed (144 s; validator passed, projected full 1,930 s of the
  2,400 s allowance). In the full stage another user's process appeared on the claimed GPU twice (pids
  577752, then 586791); the first pause cleared after 156 s, the second lasted the full 900 s pause allowance
  and the supervisor ended the job as `contended` (terminal; `resume_declined: full published no checkpoint`,
  because the job checkpoints per window and Al's scoring had not finished). Nothing was scored; no `result.json`,
  no `summary.json`, no validator verdict for the full stage. Wall time of the claim about 1,626 s plus
  144 s smoke (about 1,056 s of it paused), productive GPU time about 710 s; about 1.8k of the 3.6k GPU-s
  budget is charged at most. Not resubmitted (root's rule).
- What exists (`full/partial.tar` in the job's inbox, copy extracted on gpu5 at
  `/local/users/emanuele/w3b-partial`): Al's six named points were encoded to real bitstreams and each file was
  decoded by two separate processes, two passes each; for all six the four lists of 240 frame hashes are equal
  and the decoders reported deterministic (this is the rule's check (b); check (a), the I frame, is in the
  unwritten `result.json`, and the reference buffer stays unreachable). Rates from the container sizes (no quality
  looked at):

| Al point | bytes | kbps |
|---|---:|---:|
| scale 2 QP21 | 23,104 | 19.253 |
| scale 2 QP23 | 25,995 | 21.663 |
| scale 2 QP24 | 27,806 | 23.172 |
| scale 2 QP25 | 29,587 | 24.656 |
| scale 2 QP27 | 33,194 | 27.662 |
| scale 2 QP30 | 40,066 | 33.388 |

  The prediction (19.2 / 21.4 / 22.7 / 23.8 / 26.3 / 30.4) was within 1–10 %; both hybrid rates are bracketed:
  21.4588 by QP21 and QP23, and 22.7380 by QP23 and QP24 (with W3's QP18 at 16.08 kbps below). Med's points,
  the Al scores, the foreground masks' use and the SVT control were not completed.
- Consequence. The dominance verdict at 22.737962 and 21.458796 kbps and the BD-rates against the stronger
  SVT envelope are NOT available; `experiments/tennis/w3b_report.py` (tested on stand-in data) produces them
  from the saved summaries once a scored summary exists. Options for root, both about 10 GPU-minutes: (a) the
  same job again (the encoder is deterministic, so the streams should reproduce byte for byte and can be
  checked against the sha256 of the saved files); (b) a scoring job that decodes the six saved Al streams
  (and a Med run) and compares the decoded frame hashes with the saved ones, which also tests decoding in a
  later job on the same GPU class, an open W3 limit. Suggest widening the contention allowance or
  checkpointing after the encode stage so a pause cannot discard an encode.
