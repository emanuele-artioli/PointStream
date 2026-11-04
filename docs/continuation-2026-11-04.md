# Continuation — 2026-11-04: expanded submission research

## User goal and pending decision

User explicitly requests expansion of the48-frame ball proof of concept into a
full contribution and completion of remaining PLAN work toward the best paper.
The primary owns scientific choices, with requested Terra 6.1 medium delegates
(actual tier telemetry unverified). PRESLEY must never be modified. Existing
23 main/reference pages (at most two reference pages) plus five appendix limit persists.
Coauthors have not reviewed the paper and footage permissions remain unresolved;
no submission is authorized.

A single asynchronous budget question is **PENDING, NOT APPROVED**: fresh 20 GPU-h /
24 workload CPU-h aggregate program, first 2 GPU-h /3 CPU-h for feasibility, then scale
only under real correctness and complete-cost gates. [Card](cards/submission-research-2026-11-04.md).
Options presented: approve full staged 20/24 (recommended), feasibility 2/3 only,
or code/existing-evidence only. Do not infer approval from the user's broad
research goal or elapsed time. Old K5c/W8/W9 balances are closed; no new GPU
request has been submitted. No timers, LLM keepalives or active fleet waits.

## Repositories and safe state

- Code branch `codex/submission-research-2026-11-04`, based on fetched main
  `98d12601` (MoQSplat PR330, unrelated to paper evidence). CPU preparation
  commit `85588f048b20427f32c2b67ca88e4335c343ab87`; source-evidence/receiver
  preparation commit `4f4ea2dc` pushed. [PR #331](https://github.com/emanuele-artioli/PointStream/pull/331)
  draft, full Linux CI succeeds at 4f4ea2dc; later review/commits must inspect exact
  head/checks. Some final documentation/remux-preparation files may remain
  uncommitted; inspect status before acting. Scoped code changes only.
- Manuscript `/private/tmp/pointstream-w7-manuscript`: prior pushed main a297279;
  factual exposure correction `5109c83d6fc5156e390fd79fb54d72f89bee777f` is
  committed locally, **not yet pushed** at this checkpoint. Both variants
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
