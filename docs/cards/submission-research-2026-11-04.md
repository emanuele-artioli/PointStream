# Submission research program — 2026-11-04

Status: **APPROVED by the user, 2026-11-04; ceiling increased 2026-11-05**:
24 GPU-hours /28 workload CPU-hours total, including all earlier programme charges. Initial release: at most 2 GPU-hours /
3 workload CPU-hours across feasibility. Larger stages remain conditional on
correctness and complete-cost forecasts fitting the approved aggregate and
workstream ceilings. Ordinary bounded implementation corrections need no new
permission; source/numerical rules and stops below remain binding. Closed
K5c/W8/W9 balances are not transferred. Unknown submissions are never replayed.

## Goal and scope

Develop the ball prototype into a reproducible, fully billed small-object
coding contribution; resolve the most valuable remaining baseline, decoder and
temporal questions; use valid oracles to decide whether broader legacy pipeline
and generative-peer work merits scale. Preserve failures beside gains and
update the empirical paper using measured results. The submission limit stays
23 pages including at most two reference pages, plus five appendix pages.
PRESLEY is read-only; submission needs a separate instruction.

The ball benchmark registers all 30 TrackNet games 8–10 clips: 5,675 frames,
5,472 visible, 197 explicitly absent, 6 occluded. All clips were used in past
tracker selection, so this is a retrospective benchmark, not untouched data.
Deterministic calibration clips are game9/Clip5 and game10/Clip3, included once
in the 30, with renderer parameters frozen. New cross-sport testing may use the
seven OpenTTGames test videos only after exact context/label joins and source
qualification with the existing tennis-trained WASB judge. No table-tennis WASB
checkpoint exists in the pinned documented release. Unknown frames are not
negatives. No claim of independent matches beyond documented source identity.

## Approved inclusive ceilings

A total **24 GPU-hours and 28 workload CPU-hours**, including staging, failed
starts, pilots, native encoding/decoding, inference, metrics, independent audits,
monitoring and cleanup. All jobs inherit one-CPU bounds unless a separately
reviewed measured process-tree bound is demonstrably cheaper. Aggregate
administrative CPU accounting limitations remain explicit. Local documentation
rendering and ordinary software checks are not experiment inference charges.
No training, paid API fallback or remote agent session is included.

| Work | GPU ceiling | Workload CPU ceiling |
|---|---:|---:|
| Ball source oracles, controls, matched native ladders, 30-clip benchmark and qualified cross-sport subset |18h|20h|
| Two original SVT half-out gaps, original scoring controls and updated intervals |0.5h|3,620 s|
| New W8 CUDA diagnosis, then cost pilot and one complete fresh witness only on gates |8,200 s|11,300 s|
| W6 literal control reproduction and bounded smoothing study |0.5h|1.25h|
| Paid-mask / generative-peer feasibility oracles, no full 34-window sweep |1h|1.5h|
| Shared admission, audit and uncertainty reserve |6,200 s|3,980 s|
| **Total** |**24h**|**28h**|

The original initial release was at most **2 GPU-hours /3 workload CPU-hours** across
feasibility stages. The November5 revision adds a separately gated first tranche
of at most **2.5 GPU-hours /3 workload CPU-hours** for ball qualification and scale.
The CPU allocation table includes the recorded transfers from shared reserve to SVT. The rest is usable only after representative smoke and
measured complete-cost forecasts fit both remaining ceilings and the relevant
workstream cap. Root may move the shared reserve to a named workstream with a
prospective written justification, never exceeding aggregate ceilings or
changing scientific criteria. Pilot-based allocation is a ceiling, not a
runtime prediction or a guarantee of a positive result.

Each stage/shard ends and saves every completed item within 45 minutes; smoke
commands at most 600 seconds. All runs use canonical fleet submission, fixed
source/input/model/tool/environment hashes, substantive validators, fresh
host/job checks, immutable prior outputs and new result directories. One
bounded canonical fleet wait per batch; no recurring LLM timer or keepalive.
Unknown submissions are inspected, never repeated. Only task-owned processes
may be cancelled. Device/provider/kernel checks use qualified A6000/Ada paths.

## Gates and stops

1. Authenticate inputs, labels/exposure, source modules, runtime and toolchain;
   pass local CPU/parser/metric checks before any inference. New wire framing
   must retain actual native bytes and BALL symbols and charge every byte.
2. Ball: source-only oracle and the real native/wire/metric path on small
   development inputs; then two calibration clips. The full 30-clip benchmark
   requires complete forecasts. Task claims additionally require qualified
   independent image judging; a failed judge does not qualify as a codec
   failure or automatically stop correctly joined fidelity measurements.
   Source qualification failures remain registered exclusions, not replacement
   clips. Extra anchor points are added only for unsupported overlap. Controls
   assess placement/mark sensitivity, not prove perceptual realism.
3. W8: two fresh minimal CUDA conditions in the same authenticated JCV runtime,
   outside and after the original isolation policy, without generator weights.
   The initial diagnostic uses a newly named historical-policy relocation profile: the authenticated original source is reconstructed read-only, outputs/aliases are fresh, the worker is preflight-only with no weight-body reads. The full configuration diff is retained. It is not a literal rerun of the original configuration. No source,
   label, saved-output or numerical-tolerance boundary may be relaxed, and no
   CUDA initialization occurs before isolation. If a defect concerns a missing
   generic runtime resource, any repair must use a newly versioned, authenticated
   profile with repeated denial/identity checks; it never retroactively passes
   the original policy or broadens protected-data access.
   Only a passing diagnostic admits a new one-/three-step cost pilot; only its
   complete forecast and unchanged pixel/quality gates admit a full witness.
4. W6: reproduce the original numerical path and literal 1e-6 control; float32
   diagnostics cannot override it. Smoothing is tested only after correctness
   and all 239 pairs/80 quality ticks fit. No generation or training.
5. Original med000 DCVC remains user-dropped. The two SVT repairs retain exact
   original inputs, regions, 80 ticks, metric controls and full point identities.
6. Oracle or scientific failure narrows the claim and stops that route; no
   automatic larger model, training or 200 GPU-hour legacy sweep. A documented
   implementation defect may receive at most two bounded repair smokes inside
   the approved ceilings, only after correcting and testing its cause; no
   thresholds, samples or quality rules may be changed after test outcomes.
   Lost connectivity never permits replay. If cost or evidence cannot fit,
   preserve the result and return for a new decision.

The approval permits this staged program, not training or an automatic full
34-window pipeline study. Those still need a dominating oracle, measured
proposal and separate resource decision. All current reservations and charges
live in `docs/experiments/submission-research-budget-20261104.json`.

## Root stage2 release —2026-11-05

Independent16/48 accepted-path audit (`66cbbd2e…`, job `20261105T000550Z-d8b5c76b`) verifies exact16 contract and complete48 retained results/timings; it does not independently recompute all judges or re-decode full48. New bounded Clip3 full188 confirmation is released at2500GPU/2800CPU inclusive, with500s smoke,1750s full,2400s oneCPU campaign and100GPU/400CPU external. Its new purpose-specific fidelity receipt `389f3af9…` predicts1649.78s using1.5× measured affine command cost plus200s overhead and60s reserve, adjusted upward for a slower new16 smoke. Old failed188 projection and48-onlyreceipt remain unchanged. No full30 stage is released: the586-frame clip projects2594s command alone, so checkpoint stages require separate16/48 numerical equivalence and complete-cost confirmation. Existingcontrols are a post188 conditional reservation because they require that full saved archive. Initial GVC and one verified table-tennis source-only pilot remain inside7200GPU/10800CPU feasibility bounds, separately from this larger release. Global and per-workstream ceilings remain unchanged.

## Checkpoint equivalence phase — 2026-11-05

Goal: establish that separating preparation from per-point scoring preserves the complete Clip3 16/48 coding and evaluation contract, then measure its cost. Preparation retains all six full native sequences and the joint fit. Each point is freshly decoded, checked against the prepared pixel hashes, and scored over its complete temporal sequence. Assembly requires all six points for each size and exact agreement with the accepted d8b5 reports.

The competing explanation is that a changed CUDA state, resource path or temporal boundary changes results. Any native, packet, pixel, metric or judge mismatch stops the phase; no tolerance is enlarged after output. Failed identity, cost, deadline or resource checks also stop. This phase does not authorize the 30-clip corpus.

The prospective ceiling is 3,400 GPU-seconds and 4,350 workload CPU-seconds: preparation 700/900, six scoring jobs at 450/550 each, and CPU assembly at 0/150. Preparation uses at most 650 campaign seconds, with smoke/full limits of 200/420 seconds. The SHA-bound timing decomposition predicts 393.13 seconds for full48 with a 1.5 margin and reserves, scaled upward for a slower new16 smoke. Its bounded 45 CPU-second evidence acquisition is included in preparation's external allowance. Every stage uses one frozen source checkout; later documentation commits cannot change its identity.

### Checkpoint preparation repair1

The first preparation command completed native/receiver work but incorrectly invoked scored validation on intentionally empty score arrays. Its original700GPU/900CPU ceiling is charged conservatively. A new700GPU/900CPU preparation attempt is reserved prospectively; the whole equivalence phase's cumulative ceiling is now4,100GPU/5,250CPU, including six original point jobs and assembly. Prepared validation retains native, packet, geometry, receiver and accounting checks and requires scores to be empty. Ordinary validation still rejects prepared reports; assembly marks metrics complete only after six complete point receipts. No numeric tolerance, encoding, fitting, guard, source or judging rule changes. Original snapshot and outputs remain intact. Root accepts focused validator/dispatch tests plus source review; the dispatch fixture stubs preparation and does not itself exercise the full encoder. Actual16 parity and cost gates remain required.

## CUDA system-call trace pilot — 2026-11-05

The four-cell saved observation associates failure with the portable Landlock condition; it does not identify a resource or exclude NoNewPrivs, which Landlock requires. The next three fresh, equally traced processes hold namespaces off and compare baseline, NoNewPrivs alone, and the same read policy. The competing explanation is tracing or NoNewPrivs rather than a denied system resource. Each must record its privilege state and identical CPU warmup; the Landlock arm must retain all ten protected-path denials. Trace candidates exclude intentional probes and never cause automatic permission changes. No model, encoding, body read in a control, or decoder qualification is included.

The ceiling is200GPU/300CPU, including a150-second one-core campaign and50GPU/150CPU external. Arms stop at30seconds, the stage at120seconds. Existing paired controls support this small diagnostic forecast; live tracing/driver compatibility remains unverified until the bounded pilot. Failure, incomplete receipts or exhausted budget narrows the observation and stops. Both eligible hosts expose strace; no installation is authorized or needed. Initial charged-plus-active feasibility remains5,413GPU/10,341CPU, within7,200/10,800.

## Fixed system-file grant pilot — 2026-11-05

The trace controls passed without a file policy, including NoNewPrivs alone. The next pilot holds user/network namespaces on and compares the old portable policy, that policy plus the twelve trace-observed kernel/loader file leaves, and those leaves with full mutation confinement. These are explicitly new policies, not a retroactive pass for any old run. No individual denied path is claimed causal.

The exact leaves are recorded in the new module and W8 entry: loader cache, CPU/device/driver metadata, this process's maps/status, memory configuration and the two NVIDIA MIG metadata files. No parent directory is granted. The child resolves its own proc paths; a supervisor receipt binds the root-owned loader cache. All ten actual-source/alias denials remain required. The strict arm must also deny safe scratch mutations outside the output leaf and prove output writes work. No model, protected worker body read or generation is included.

The ceiling is150GPU/200CPU, including a120-second one-core campaign and30GPU/80CPU external, with25-second arms and15-second reserve. Any missing leaf, identity, bootstrap, budget or mutation failure stops as incomplete. A tiny CUDA pass qualifies only that configuration for a later separately gated model pilot.

## Verified baseline and decoder-input phase — 2026-11-05

The four new SVT scores and 75 complete-byte/native-YUV roles are independently audited. Corrected report `d09b0543` and audit `5a435128` cover six familiar-match windows only, using every valid control (71 distinct streams), complete native IVF as primary and complete MKV separately. Conditional player gaps remain positive, whole-frame anchors are better, med000 DCVC is missing and the Medvedev subgroup is inconclusive. The original-IVF strict half-out brackets remain unresolved; MKV resolves them. Tier 2 remains unchanged. The first report is preserved as invalid after independent endpoint/coverage review; both 120 CPU-second analysis allocations are charged. This is a fairer, more complete conditional comparison, not full-codec certification.

The real public-folder I/O oracle `20261105T071725Z-b5eb8a21` independently passes the unchanged strict policy and all four real file hashes/cleanup; conservative charge 135 CPU-seconds, zero GPU. It qualifies warm-cache mechanics only. An opt-in one-/three-step generation cost candidate remains prospective until its full source/spec/forecast is reviewed. The original 20/24 ceiling is unchanged; proposed 24/28 and larger ball release are pending. No training, paper submission or source-footage redistribution follows.

## Research-first ceiling revision —2026-11-05

User approved the pending24GPUh/28CPUh proposal in `submission-research-budget-revision-2026-11-05.md`, with the first additional9,000GPU/10,800CPU-second tranche conditional on per-job correctness, source-compatibility, real native/namespace/CPU-accounting qualification and292/586 scale confirmations. The aggregate increase is assigned to ball; the full30 study remains gated by a complete forecast and reserves. Pause manuscript edits until worthwhile research endpoints are resolved. Preserve completed negatives, historical ceilings and failed attempts. No training, legacy200GPUh study, or submission.

### Root prospective reserve transfer — November6 UTC

The explicit shared-reserve authority above is used prospectively to move1,000GPU/500CPU seconds toW8; aggregate24GPUh/28CPUh and ball ceilings stay unchanged. W8limit is8,200GPU/11,300CPU; shared6,200GPU/3,980CPU. After paired-cost and CPU-structure acceptance, W8has5,159GPU/4,616CPU remaining. Reserve at most4,550GPU/4,550CPU for the staged fresh-witness attempt described in `w8-full-witness-staged-proposal-2026-11-06.md`, including newsourceproof and alloperational costs, leaving66CPU insideW8. This is a ceiling reservation, never proof of fit or execution authority. Actualsource/securityreview and complete forecast precede each released stage; no full candidate/quality run is released. A failed or unfundable stage stops and retains evidence. The old witness allocation stays closed.
