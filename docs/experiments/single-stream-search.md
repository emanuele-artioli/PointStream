# Experiments: single-stream search

### 2026-10-28 — Accepted parallel batch, frozen initial tests

- User instructed execution of the recommended tasks. New independent
  ceiling3600GPU-s/3600CPU-s: restoration1800, diagnosis600, refinement1200
  conditional on both native correctness and differentiation/entropy audit.
- [Card](../cards/single-stream-search.md) records hypotheses, competing
  explanations, scope, configuration, stage gates and stop rules before outcomes.
- Prior paid memory request20261028T181634Z-05c49090 COMPLETE; never replay.
  Its native SVT anchors are immutable inputs to restoration. Existing stored
  DCVC streams are not portable; diagnosis must re-encode then decode within
  its own request. No new model download or full training authorized.
- All six hosts inspected; doctor for compatiblegpu2/gpu3 ongoing. No new job
  submitted yet. Charged0GPU-s; CPU source audit/preflight ongoing.
- Subagents only gpt-6-luna/max with exact frozen launch/report instructions.

### 2026-10-28 — First two independent requests submitted

- CPU source audit20261028T190840Z-eb011d4b, pending at receipt,
  cap600CPU-s/0GPU. Receipt/private/tmp/single-code-audit-submission.json;
  specSHAd667a3cb48e6f0c368bf1b26e7c8eec82804d3b543acf04754337ee2fc132cfe.
- Native diagnosis20261028T190838Z-3dea4d74, pending at receipt,
  cap600GPU-s. Receipt/private/tmp/single-diagnosis-submission.json;
  specSHA923bc96d1f161dc71f6291edca625faf45717fdfad05a3f33131194e4abc22a6.
- Both use frozen sourceafe86f57620d85161aba5a8dc60e17c8e820fb4e,
  snapshotSHAd30a05a849236fae7c78c3a20cacbed98819035c2b206b66608899e7fae0753b,
  no selected patches/untracked files. All6hosts inspected, doctorPASS.
  CPU request uses all6compatible hosts; diagnosis A6000poolgpu2/gpu3.
- Diagnosis initial sandbox health query hit DNS denial BEFORE job_id creation
  (verified submit implementation). Corrected network permissions, then one
  request accepted. No uncertain replay, GPU execution or scientific retry at
  that preflight failure. Parent owns status and one bounded batch wait.
- Pinned upstream source input33fec4182805ebc5c5ab3cd90715abf70d855d71916837e65e55beca992c0f76,
 133120bytes; DCVC-DTe3deeda1cd99566498504231f6853fd554ec00e9,
 DCMVC4aa95ef4b0176b6076b1ba0396d2cd178258d506. Selected original files only,
 archived with per-file SHAs; no model download or source execution.

### 2026-10-28 — Restoration submitted; batch monitoring begins

- Restoration20261028T191318Z-b4727331 pending at receipt, cap1800GPU-s.
  Source49ee1b931c7771876d5dd5aa55f7f2a96f8d1bcb,
  snapshot4608ded8bfb28efd429d9f2164a1750a0b5afcf48beaf4814682d070d6963130,
  spec66b1d70e8bfe19a3a208bdb97c2a59b02f19a04193325a17530a323c35d1d3b5.
  Receipt/private/tmp/single-restoration-submission.json. No selected changes.
- Eight focused alignment/transport/decoder-report/document tests pass;
  configured ruff and mypy pass. Root owns independent audit code; delegates
  only submitted frozen specs. Full local suite running separately.
- One bounded native batch wait covers all three recorded IDs; no verified
  automatic model wakeup, so attended execution with no recurring timer.
  Native wait output/private/tmp/single-batch-wait.json, max2700seconds.
  Initial diagnosis admittedgpu2/A6000; CPU audit admittedgpu5/noGPU.

### 2026-10-28 — CPU audit complete; unsupported native alternative rejected

- CPU audit20261028T190840Z-eb011d4b COMPLETE on gpu5, source inventory214files,
  original Python/CUDA sources preserved. Differentiable UF forward_one_frame
  exists; native bitstream compress delegates to C++ proxies taking pixels,
  with no external optimized-latent input. DCVC-DT targets IntraNoAR/DCVC-DC,
  not installed DMCI/HT-L. DCMVC is another DCVC-DC-based codec with its own
  checkpoints/dependencies. Do NOT claim gradients are absent or substitute
  estimated bpp for a native bitstream. Refinement gate remains unestablished;
  spend0/1200GPU-s until a native optimized-latent transport path exists.
- Initial diagnosis20261028T190838Z-3dea4d74 FAILED validator because the
  default-stream encode raised 'operation not permitted when stream is capturing'
  at dmc_common.cpp:15. No default-stream encode/metrics completed. Legacy
 4-frame arm passed both within-process passes and fresh decoder, encoderI
  equality, native container identities and memory. This does not reproduce
  or resolve the original24-frame nondeterminism. The API assumption was wrong.
- Charge250GPU-s, including admission through canonical terminal232.8s and
  salvage/publication. Complete ONLY the unaffected24-frame legacy reproduction
  within the remaining350GPU-s of the same600 cap, with a fresh smoke. The
  [card](../cards/single-stream-search.md) explains this implementation correction;
  no changed quality rule, new alternative or extra allocation. No further request.
- Initial events acknowledged; CPU stall event occurred during environment/
  publication overhead, actual smoke command3.56s; CPU full succeeded. No
  recurring model polling. Restoration remains the independent quality test.

### 2026-10-28 — Supported-path completion receipt

- Job20261028T192649Z-867e69f2 pending at receipt, cap350GPU-s; first diagnosis
  charged250, aggregate diagnosis cap stays600. Frozen source
  cc94686fcd8b9f8ae885d0987ebaf0af7999ccee, snapshot
  e57f2c1f5b53581b324ba5b605dd9a592c1d694d8124d73df76b54e51e1cc6f2,
  specd4c2a7d5a3591e9a0cb3c00c28345098d021e075cfb761a6fb978fdb58c137d9.
  Receipt/private/tmp/single-legacy-completion-submission.json; no patches or
  untracked code. Source and GPU pool refreshed before this completion.
- Native bounded wait for restoration + supported completion writes
  /private/tmp/single-batch-final-wait.json, max1800seconds. All previous
  attention events acknowledged; no timer, no automatic model wakeup claimed.

### 2026-10-28 — Batch complete, no whole-frame codec win

- [Final report](../single-stream-report-2026-10-28.md),
  [restoration audit](../single-restoration-audit-2026-10-28.json),
  [source audit findings](../single-code-findings-2026-10-28.json),
  [initial diagnosis audit](../single-diagnosis-audit-2026-10-28.json),
  [supported native audit](../single-legacy-audit-2026-10-28.json).
- Restoration COMPLETE96frames/32ticks, exact source/anchor/metric controls,
  full command503.666s, projected948.070<1200, reserved3740MiB. Whole-frame
  LPIPS BD-rate+0.531/+2.287/+4.892%, DISTS+0.297/+1.555/+4.141% at strengths
 0.15/0.35/0.60. Background loses too; foreground curve gains small. No robust
  two-metric whole-frame point; no extra tuning. RestrictedSVT grid, development
  only; this is not a learned-model or other-domain impossibility bound.
- Supported legacy reproduction COMPLETE24frames, QP21/960x544/nativeYUV,
  gpu3/A6000. Four decoded pass arrays equal saved pixels, encoderI matches,
  original container/weight hashes pass. No stored-stream replay. This fresh
  point passes; original nondeterminism was NOT resolved or made portable.
- CPUaudit COMPLETE214files; gradient forwards exist, native optimized-latent
  transport unqualified. Refinement0GPU-s; no estimated-bpp codec win reported.
- Conservative new ledger: restoration750GPU-s, initialdiagnosis250,
  supportedcompletion300, refinement0 =>1300/3600GPU-s,2300unspent.
  DedicatedCPU audit489.457wall-s x2threads charged1000, localanalysis120 =>
 1120/3600CPU-s. Staging/publication included; GPU-job CPU support inside its
  allocations. Old ledgers remain closed; unused balances imply no further run.
- All requests/native waits terminal; completion events acknowledged. No
  recurring model timer, API bridge or verified automatic wakeup claimed.
  Exact source archives match all three snapshots byte-for-byte.
- [Implementation PR274](https://github.com/emanuele-artioli/PointStream/pull/274)
  MERGED b76c27565cb5ba88b0bdb708c526826ad248cd00, Linux CI passed.
  Eight focused tests and configured Ruff/mypy pass. Full macOS suite has the
  same two known platform failures, with process test passing outside sandbox.
- Evidence archive100208640bytes/79manifested members verified at destination,
  SHA71dcbd1cd30e26e56488aadb397ea75eff8cb47e78b7ed9d8376702bb267a5a3,
  mode0444. [Receipt](../single-evidence-receipt-2026-10-28.json).
  All7original completed/full/smoke publications also protected mode0444;
  task-owned PIDs absent on gpu2/gpu3/gpu5, no PointStream GPU claims.
  Budget/source identities in [ledger](../single-budget-2026-10-28.json).
  Resume after token interruption inspected the recorded terminal requests;
  no job replay, request, timer or extra GPU charge was introduced.
- Results [PR275](https://github.com/emanuele-artioli/PointStream/pull/275),
  merge-commit auto-merge after Linux CI; current report/PLAN/continuation all
  point to the closed batch. Root working copy contains no new research job.
