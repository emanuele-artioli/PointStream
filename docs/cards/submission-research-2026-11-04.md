# Submission research program — 2026-11-04

Status: **APPROVED by the user, 2026-11-04**: a fresh staged ceiling of
20 GPU-hours /24 workload CPU-hours. Initial release: at most 2 GPU-hours /
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

A fresh **20 GPU-hours and 24 workload CPU-hours**, including staging, failed
starts, pilots, native encoding/decoding, inference, metrics, independent audits,
monitoring and cleanup. All jobs inherit one-CPU bounds unless a separately
reviewed measured process-tree bound is demonstrably cheaper. Aggregate
administrative CPU accounting limitations remain explicit. Local documentation
rendering and ordinary software checks are not experiment inference charges.
No training, paid API fallback or remote agent session is included.

| Work | GPU ceiling | Workload CPU ceiling |
|---|---:|---:|
| Ball source oracles, controls, matched native ladders, 30-clip benchmark and qualified cross-sport subset |14h|16h|
| Two original SVT half-out gaps, original scoring controls and updated intervals |0.5h|0.75h|
| New W8 CUDA diagnosis, then cost pilot and one complete fresh witness only on gates |2h|3h|
| W6 literal control reproduction and bounded smoothing study |0.5h|1.25h|
| Paid-mask / generative-peer feasibility oracles, no full 34-window sweep |1h|1.5h|
| Shared admission, audit and uncertainty reserve |2h|1.5h|
| **Total** |**20h**|**24h**|

The initial release is at most **2 GPU-hours /3 workload CPU-hours** across
feasibility stages. The rest is usable only after representative smoke and
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
   The initial diagnostic reproduces the original policy exactly. No source,
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
