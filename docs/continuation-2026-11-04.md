# Continuation — 2026-11-04: expanded submission research

## Live job checkpoint — 22:28 UTC

- Ball calibration20261104T222053Z-0061cb27: running smoke on gpu5Ada at last canonical status; immutable sourcebf6208c5/sourcec34ddd02;3300GPU/3600CPU phase.
- W6control20261104T222153Z-1145ce7a: pending Ada admission,1200GPU/1500CPU.
- W8diagnostic20261104T222305Z-5d374710: pending gpu6 exactprofile,180GPU/240CPU, deadline23:00UTC. gpu6 CPUheadroom0 atlastprobe; do not force otherhosts with incompatible receipt/inodes.
- One canonical wholebatch wait is running: localexec session15697; result/private/tmp/pointstream-feasibility-batch-wait-20261104.json, max2700s, deadlineapprox23:13UTC. Preservehandle; no recurringtimer/nativecompletionresumption has been verified. Wait exits on meaningful attention or allterminal. Inspect before any replay; no uncertain request is duplicated.
- W7remux20261104T221616Z-49f19251 terminalfailure metadata (no color-space/transfer/primaries),330CPU charged/0GPU, nofull/noquality; terminal+decision events acknowledged.
- CPU fixedmaskoracle150CPU/0GPU reserved; agentexecuting committed237ae214 on authenticated report, no model/native decoding.
- CodePR332 has auto-merge enabled, LinuxCI passedbf6208; local237ae214 push inprogress handle72811. Full localpytest11 unrelated existing cv2/PyYAML court/segmentation failures; mypy7existing unrelated errors.
- Manuscriptworktreehas uncommitted source-screen/remuxqualification,28pages8main2appendixfigures, checker+5registrytestspass. Rootreviewedpage12. No manuscriptpushyet. PRESLEYuntouched.

## Current approved phase (supersedes historical checkpoint below)

User explicitly approved the staged 20 GPU-hour /24 workload CPU-hour program,
with initial feasibility release 2 GPU-hours /3 CPU-hours. No training or legacy
full sweep is authorized. Old allocations remain closed. Current branch is
`codex/submission-feasibility-2026-11-04`, based on main `404c280d` (PR331 merged).
Authority/charges/reservations/jobs live in
`docs/experiments/submission-research-budget-20261104.json`.
W7 CPU remux job20261104T221616Z-49f19251 is terminal failed (metadata gate: unavailable color metadata),330CPU-s/0GPU phase reservation retained pending final accounting. Ball calibration3300GPU/3600CPU, W8diagnostic180GPU/240CPU and W6control1200GPU/1500CPU are reserved before dispatch; no IDs yet at this checkpoint.
Three Terra6.1 medium delegates implement the ball admission wrapper, W8 CUDA
supervisor and remux adapter. Actual model/effort telemetry remains unverified.
No submission is authorized; PRESLEY remains read-only. Coauthors have not yet
reviewed. Footage permissions remain unresolved. Keep23main/reference pages
(at most2reference pages), plus5appendix pages.

The remainder records the previous preparation checkpoint; its pending-budget
wording and draft PR status are historical, superseded by this approval.

## Repositories and safe state

- Code branch `codex/submission-research-2026-11-04`, based on fetched main
  `98d12601` (MoQSplat PR330, unrelated to paper evidence). CPU preparation
  commit `85588f048b20427f32c2b67ca88e4335c343ab87`; source-evidence/receiver
  preparation commit `4f4ea2dc` and transport/continuation preparation `edbefa5b` pushed. [PR #331](https://github.com/emanuele-artioli/PointStream/pull/331)
  draft at this historical checkpoint; full Linux CI succeeds at edbefa5b; later review/commits must inspect exact
  head/checks. Some final documentation/remux-preparation files may remain
  uncommitted; inspect status before acting. Scoped code changes only.
- Manuscript `/private/tmp/pointstream-w7-manuscript`: prior pushed main a297279;
  factual exposure correction `5109c83d6fc5156e390fd79fb54d72f89bee777f` is
  fast-forward pushed to Overleaf main and independently verified. Both variants
  build 28 pages, appendix starts 24; checker and five registry tests pass. Claim binds
  inventory SHA 4ed6ea5857d0fc54a4f9a158e10da31cb3836d82b1acb43b950dfd34d4b762f0.
  That exact inventory is now committed in code 85588f04; do not edit it silently.
  Only push to remote `overleaf` main after fetch/ancestor check and root review;
  never push manuscript to `origin`. Preserve old packages.

## Evidence and scientific decisions

- All 30 TrackNet games 8–10 clips were used in prior tracker selection
  `20261020T214833Z-bf224666` and CRF 62 probe `20261020T220308Z-436c3db3`.
  All 43 RacketVision tennis test rallies were also scored in selection. No
  selection-naive 240-frame dense confirmation set was established. Larger
  TrackNet evaluation is a retrospective benchmark, not untouched confirmation.
- Registered TrackNet corpus 5,675 frames: 5,472 visible, 197 explicitly out of frame,
  6 occluded. Deterministic calibration game9/Clip5 and game10/Clip3 included
  once in all 30. Main frozen metadata in `docs/experiments/k5d-inputs/`; no new
  JPEG/image/model/encoder inference occurred.
- OpenTTGames test 1–7 videos have sparse event-centered labels and explicit
  unknown frames; exact 240-frame contexts remain UNRESOLVED. Prior G1e camera-
  motion use is disclosed. No documented WASB table-tennis checkpoint exists;
  an existing tennis-trained judge must qualify before any transfer task claim.
- Old all 30 source archive 6d2aed8ebe3b6c55e0e168fecd7237a653dc67f580bcb535d6310c7affc32d38
  was recovered read-only. It retains original per-frame predictions and labels.
  WASB-step1 source 5,051/5,472hits within 10 px; three clips fail 80% gate (game10Clip11
  25/39, game10Clip8 40/64, game9Clip7 69/87). Source false detections 10/197
  =0.0507614 fail the proposed 5% source-negative screen set in the root's brief
  before counts were opened. Retain those failures; no threshold/alternate
  judge is tuned after this result. This is instrument calibration, not codec
  failure. Relative detector counts and raw-image fidelity remain separate.
- Old source uses raw OpenCV JPEG RGB, unlike K5c's FFmpeg YUV420→RGB. New source
  oracle explicitly uses raw JPEG. Loaders/labels/counts AST-identical to 7590a774;
  Zoo inference/transforms unchanged, constructor adds named-candidate subset.
  Check full identity/config/runtime before reusing records; no unverified
  pixel equivalence to K5c or current model inference is claimed. CRF 62 archive
  has JSON only; inspected code deleted stream work. No global absence claim.
- Old 48-frame K5c remains valid narrowly; source 45/48, rendered 42–44/48/native 0–8/48,
  all 48 positive; original 6/8 rule/whole-frame PSNR guard unchanged. No new
  compression-quality result exists. New protocol at [experiment](experiments/k5d-ball-benchmark.md)
  reports matched regional fidelity separately from task validity; task target
  mean Δrecall>=.10 at DCVC QP 0+streak and SVT CRF 62+streak, both native-bracketed,
  >=75%all 30clips; negative/calibration limits preserved. No extrapolation.

## Implemented CPU preparation and limits

- `src/pointstream/ball_stream.py`: bounded PSBL v1 exact-native+BALL framing,
  metadata/resource ID, checksum and parser/render parity tests. All physical
  bytes counted; static installed resources separate. Constant fps only.
- `k5d_analysis.py`: pooled native achievable envelope, PCHIP continuous BD,
  orientation, discrete task counts and no extrapolation. Constant-quality
  plateaus use RAW measured-rate support; BD keeps unique-quality Pareto points.
- `k5d_source_oracle.py`: real CLI, source-image model calls remain unexecuted.
  Streams full-archive SHA; binds selection/labels, actual model code-tree/config/
  weights, one CPU/CUDA, scratch outputs, records full frame predictions/indices
  and source qualification; no model imports on help. Model CUDA device binding
  must still be tied to fleet runtime/tool metadata before admission.
- `k5d_worker.py`: **receiver-only** decode CLI. Verifies PSBL/native header,
  exact installed-resource hashes, decodes persisted native bytes and renders
  BALL; actual decoder path has not been run. Not full six-point encoder/fit/metric
  orchestration, no OS/source-isolation proof, no study/fleet admission yet.
- W6 bug fix: failed literal 1e-6 control no longer exits successfully just because
  float32 diagnostic passes. Historical artifacts not modified; paper already
  discloses failed control. No smoothing/model run. New design documents same-
  path control recovery before quality/temporal comparisons.
- W8 CUDA diagnostic is per-condition helpers+strict receipt tests only, no CLI/
  admitted supervisor. Future launcher must authenticate exact JCV/runtime/
  original policy, new namespaces/affinity/exclusivity,90 GPU-s /120 CPU-s proposal,
  two fresh processes and complete receipts. No CUDA initialization was run.
- `w7_remux_oracle.py` is optional separate-transport CPU preparation, not run.
  Original half-out SVT gaps hit prescribed CRF 70 boundary by 69/39 bytes, not
  known lower unscored points. New IVF→Matroska copy must preserve exact YUV,
  cadence and color metadata and count full container; both decodes force CPU
  libdav1d. Strict missing-color metadata fails; no waiver. Needs bounded
  supervisor<=120 CPU-s /0 GPU, authenticated source/tool/pixel controls before
  run. It cannot masquerade as old IVF gap repair; if valid, re-rate all SVT points
  under new contract and independently score only missing quality. Original
  med000 DCVC remains user-dropped; unknown old native-archive listing never replayed.
- Real local wire engineering audit on six saved 48-frame streams PASS under
  code 85588f04, native/BALL bytes exactly retained, 0.020504 CPU-s interval
  excluding startup; conservative 1 CPU-s within 120 CPU-s local engineering cap.
  No decoding, source pixels or quality inference. Local Homebrew FFmpeg fails
  dyld missing libx265.215; do not modify user's installation or claim decoder
  validation. Report committed `wire-framing-audit-20261104.json`.
- Full Ruff passes; focused tests 41 passed before remux three tests (44 expected now).
  Local full mypy retains seven baseline/environment errors in five unchanged files.
  Linux CI passes 4f4ea2dc; no personal environment changed. Final current checks,
  head and raw receipts must be verified before merge/publication.

## Next actions

1. Inspect working tree; review/commit final remux/doc changes and update PR #331;
   full CI thenfollowprojectauto-merge policy. Source/metadata prep is useful
   independently of pendingGPU approval. Don'tclaimfullplancompleted.
2. Review/push the factual manuscript 5109 correction after source pin confirmed;
   fresh frozen selected-source builds/package if delivering a new PDF. Prior a297
   package remains preserved, no new scientific quality results.
3. If user approves resource cap, finish/admit ONLY real source qualification and
   feasibility paths with exact hashes/staging, full job/host preflight and
   substantive validator; initial 2 h GPU /3 h CPU total. Source-hidden receiver/native
   orchestration and matched quality/scaled study remain gated; no placeholder
   as evidence. Persist new ID/budget before one bounded fleet wait. Never replay.
4. If approval is pending, continue bounded CPU preparation/evidence work; no new GPU.
   No automatic timer/wakeup is active. Original broad goal persists.

## Final CPU-phase publication checkpoint

- Manuscript 5109 is now pushed to Overleaf. New review package
  `/private/tmp/pointstream-research-review-20261104-5109c83/` has selected-source
  ZIP SHA 50ebaba74ddafdc27aa88c1584a3fb9201a729e893397b1a789bb9fd2973ecc7,
  anonymous PDF SHA 6a2c998fe6e394884eb926fae435e86c6436b8b49353bcddf8f5d065a0539905,
  author PDF SHA 8199c627128bed24d047496b09d30578e9a6cf1781605fcdbfb59eab85034fe3,
  manifest SHA ef3e1de19d283ceb3ee6bcbf177fbb9f719ca702b5b5367189e3a33ccfb57f1d.
  Root verified all listed hashes, all 23 active-source files against the exact
  checkout, 28 pages/reference 22–23/appendix 24–28 and changed page 12 layout.
  Existing 48-frame anonymous review data is unchanged; separate allowlisted
  exposure-audit JSON hash 90dd72b1389b7ed1d4b501c382c651fdf477675c4fcd208dd4729c10b191fe11
  has no direct author/homepath/identifying-projectURL strings in the scoped scan.
- The manuscript's symbolic `.git/HEAD` was unexpectedly missing; cause unknown.
  Source files, main ref and reflogs agreed on 5109. Root preserved complete Git
  metadata under `/private/tmp/pointstream-manuscript-git-recovery-20261104/`,
  restored only `ref: refs/heads/main` atomically (no active locks), and verified
  clean tracked source plus successful fast-forward push. No object, index,
  branch history or manuscript content was discarded; old packages remain.
- Code full CI passed edbefa5b at run 41912628839. Final doc-only head and PR #331
  readiness/merge state should be read live. Focused 44 tests and full Ruff pass;
  local seven mypy baseline/environment errors persist; no personal installation altered.
- Resource question remains PENDING. No new GPU allocation/request, no native
  remux/decode or model/image inference, no fleet wait/timer. Author-review is
  still not submission approval and original broad research goal persists.
