# K5d: conventional-only confirmation feasibility

Status: source design/build authorized; scientific compute UNRELEASED.
Root decision: the exclusive Ada queue `20261113T030323Z-af6c09da` expired
without admission. Keep the registered dual-codec benchmark open. Test a
separately named, narrower confirmation route against SVT-AV1 that can score
with ordinary models on a shared Ada GPU and decode on CPU. This is within the
existing Ball workstream and unchanged user 24 GPU-hour / 28 CPU-hour ceiling.
It is not an equivalent replacement for the neural-codec comparison.

## Goal and fixed scope

Retain all 30 registered clips, their 5,675 frames, visible/absent annotations,
judge exposure rules, thresholds, serialized PSBL overhead, receiver checks,
full-frame AND ROI metrics, independent judges and guard-off outcomes.
Evaluate the existing Ball method at the three unchanged SVT operating points
(CRF48/55/62, preset4). Do not alter global six-point defaults, silently omit
DCVC from their validators, or reuse archived neural decodes as fresh results.
Report per-clip failures, absent-frame false detections and overlapping-range
BD-rate against the achievable SVT anchor envelope. No speed claim from shared
timing. Paper edits remain paused until the research pieces are available.

## Hypothesis and competing explanation

The development-clip benefit may generalize across the registered population
without invoking a custom neural decoder. It may instead be clip-specific,
depend on seen footage, lose full-frame quality, increase false positives on
absent frames, or cost too much to confirm. Preserve those outcomes beside any
positive scope; do not select clips or operating points after seeing quality.

## Stages and stop conditions

1. Build only the minimum separately named three-point evaluation/cost-oracle
   adapter, from frozen scientific source5eb in an independent checkout. Reuse
   real SVT coding, packet production, restricted CPU decoding and the existing
   judges/metrics. Preserve numerical defaults and the six-point entrypoints.
   Before a remote release, review the exact source delta, environment, resource
   profile, native byte provenance, source/receipt inventory and substantive
   validator; establish fresh source-bound CPU/denial evidence as required.
2. Freeze an explicit shared-Ada resource profile before execution: one model
   process, PyTorch allocator cap before initialization, allowance for context
   and nonallocator memory, actual GPU-memory receipts, and CPU decoder children
   with CUDA hidden. Ordinary Torch scoring avoids DCVC's shared-decoder hazard;
   this does not itself qualify correctness or memory safety. No opportunistic
   mode switch or CPU/device fallback.
3. The first scientific oracle covers all three original points on the original
   16-frame span, actual native bytes, both arms, full metrics/judges and receiver
   evidence, including cold setup and teardown. Qualify same-backend correctness
   under the original rules and measure complete cost before promotion. A
   partial or ROI-only result is insufficient. No point retries or threshold
   changes. Any unsupported component, cap violation, OOM, mismatch, missing
   metric, exhausted budget or incomplete accounting stops for root review.
4. Only after that oracle passes, select the already registered 48-frame/absent
   controls prospectively to distinguish fixed from variable cost. Forecast the
   complete 30-clip workload using actual frame counts, measured costs, full
   metric/receiver/audit overhead and uncertainty before a separately released
   scale run. Failure closes this narrowed route, not the original research
   question. Shared timings describe observed expenditure only.

## Resource authority

Initial internal reservation: **600 GPU-seconds / 1,800 workload CPU-seconds**,
transferred from unspent Ball Ada-stage funds, not a new user allocation.
This is a ceiling for prospective source checks, correctness and cost pilots,
not proof of fit or authorization to launch them. Root must precharge and
release each concrete stage after reviewing its complete forecast. No training,
paid compute, unbounded retries, full30 release, or changed quality rule.

Original decision audits:
`/private/tmp/pointstream-cpu-conventional-fallback-audit-20261113.md` and
`/private/tmp/pointstream-shared-ada-fallback-audit-20261113.md`.


November13 readiness audit found existing582fb64c adapter alreadybuilt (16frameonly; archived82focused/1880fullPASS,37skipped, Ruff/mypy524PASS) but actualCPU-only qualificationpath missing: oracle requirescpu_onlyTrue, witnesscallerdefaultsFalse. Rootreleased200CPU/zeroGPULOCALrepair fromheld1800; remaining1600CPU600GPU scientificallyUNRELEASED. Newindependentcheckout integrates frozen2c6d hardening withoutadvancingoriginals, explicitqualificationflag andnonmockedconstructor/parser/admission checks, preserving6pointdefaults. Completefutureforecastnotyetproven: native250+denial365+prep100+metadata50+admin200+reserve100 leaves535CPU foractual16oracle; measuredfitrequired,48/full30notcoveredbyreadiness. No paper/PRESLEY/training/runtime.
