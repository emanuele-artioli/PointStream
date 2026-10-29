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
