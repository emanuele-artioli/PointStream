# Background training readiness

Status: **not ready for a training pilot or final training**. This document
defines the next diagnostic request and the evidence needed to choose a pilot.
The diagnostic continuation caps below were approved on 4 October 2026;
training remains unauthorized. Prior failed/canceled attempts and immutable
receipts remain charged and preserved.

Latest outcome: the approved environment-first inventory failed on the initial
DCVC Git diff timeout, before dependency or checkpoint checks. See
[the continuation report](background-smoke-continuation-v3-20261004.md).
The proposed remainder and model stages below were not run; no repeated
request is authorized by their listing.

## Environment audit correction

The previous gpu5 inventory completed its DCVC source/extension check but
timed out before completing the environment part. The changed audit records
each dependency probe's start, command, allowance, exit/timeout, elapsed time,
and bounded output immediately in `environment-probes/`. Source checks and
each completed probe's environment state are saved in `environment-progress/`.
Heavy HNeRV and LPIPS imports execute in bounded children that inherit the
fleet process group; the supervisor can terminate the task. Probe timeouts
are at most 30 seconds (FFmpeg 10) and also limited by the stage's remaining
clock. Reporting retains the stage's existing 20-second reserve.

Source Git commands now share that clock. Missing cached LPIPS weights block
inventory immediately; no download is attempted. HNeRV must identify installed
modules with no stubs. Missing/unusable FFmpeg remains recorded separately
and does not invalidate neural dependency checks, but AV1 requires a working
encoder. These changes do not establish that cold lab imports fit the limits.

The reviewed selected snapshot paths are now tracked in
`demo/experiments/background_smoke_snapshot_paths.json`; future submissions
must include the new probe module. Regenerate from the pushed clean scoped
revision, with explicit 90-second transfer bound, fresh absolute deadline,
passing doctor, and active five-minute heartbeat. Do not dispatch the dirty
Desktop checkout or reuse an old deadline.

## Concrete next diagnostic and proposed budget

Run **one revised B1 inventory**, with `env` first in a 90-second smoke.
Its 120-second remainder is
`preview,frames-f001c3,ckpt-image,ckpt-htl,ckpt-ld,ckpt-hts,frames-f002,ckpt-f002-htl,ckpt-hnerv,training`.
Reserve 30 seconds for validation and 20 seconds overhead: **260 seconds**
total. Settings, native inputs, source references, and checkpoint paths stay
the same. Putting environment first exposes a failed import without spending
the stage on checkpoint reads. This is a changed diagnostic after a failed
pilot, not a replay of its saved request.

Require a passing environment smoke, every requested remainder part passing,
and verified complete manifests before downstream submission. The fleet smoke
gate alone does not certify the remainder. Preserve failure/blocked receipts
and stop the affected path; do not automatically repeat or lower quality.
Recheck current identities even when a previous partial receipt exists.

Only after B1 passes, submit B2 codec (450 seconds), B3 drift (300, after the
valid B2 LD pair), and B4 frozen latent (420, only with its own prerequisites).
Inspect actual independently decoded streams/packets, metrics and contact
sheets before B5's decision. An unfavorable result can justify parking a path;
it does not justify training it anyway.

The last ledger charged **1220 GPU reservation seconds** and **2205.854
conservative preparation seconds**. Before the latest approval, caps were 2400 each. A new
260-second inventory cannot fit the remaining 194.146 preparation seconds
even before publication. No new remote check or job was run for this change.

Continuation authority **approved on 4 October 2026**:

- **50 cumulative GPU minutes (3000 seconds)**, preserving prior 1220. The
  four proposed jobs reserve another 1430, totaling 2650, leaving 350.
- **60 cumulative CPU preparation minutes (3600 seconds)**, preserving prior
  2205.854. Reserve inventory 260, four publications at 90 each, four recovery
  exports at 45 each, availability/management 90, and model preparation at
  120 per diagnostic (360). Conservative planned total: 3455.854, margin
  144.146. These are prospective allowances, not measured lab timings.

Use actual receipts to reconcile each phase against these allowances. Transfer
and extraction together must fit each publication allowance, and management
and recovery costs stay charged. If a phase cannot fit, stop rather than
spending the margin automatically or resetting failed costs. The per-job
480-second, per-file 90-second, and model-stage caps remain unchanged.
This proposal covers diagnostics only; no gradients, new checkpoints,
dependency installs, full evaluation sweep, or final training.

## Gates before a training pilot

| Gate | Required evidence | Current state |
|---|---|---|
| Runtime and source | Installed dependency/source identities, native extension, cached metric weights, bounded probe receipts | Partial DCVC receipt; revised audit locally tested |
| Inputs and split | Exact frame/checkpoint hashes, strict architecture/configuration match, training manifest membership and source-frame overlap audit, sequences confined to one room/second | Partial primary/HT-L receipts; full audit pending |
| Codec behavior | B2 real streams and independent decode; B3 prediction/reset behavior; per-frame quality, temporal error, actual bytes and runtime | Unrun |
| HNeRV representation | B4 independent packet decode, source independence, frozen quality and latent/setup bytes separately | Unrun |
| Selection | B5 causal decision with negative results and at most two justified follow-ups | Inconclusive |
| Training configuration | Selected model/objective, exact scheduling/lambdas/normalization/precision, fit and validation manifests, stopping rules | Deferred until evidence selects a path |
| Pilot and resume | Bounded production-path training, finite gradients, validation trend, peak memory and throughput, checkpoint-resume equivalence | Unrun and not authorized |
| Final scale | Pilot-supported runtime/resource estimate, scaled confirmation if needed, full budget and explicit launch authorization | Deferred |

`inventory_training` currently labels hold-out overlap **unknown** if the saved
hold-out manifest lacks source frame offsets. That may be an honest smoke
result, but it does not satisfy the training split gate. Resolve source offsets
or establish a separate verified split before any gradient update. Do not use
the final hold-out for pilot selection or early stopping; reserve validation
seconds within the fit domain with exact source identities.

The legacy `factory_bg_rd.train_hnerv` calls `hnerv_import_stub` and
`_ensure_target_package`, creates replacement modules, and can install packages.
It also sets a default numeric CUDA device. It is not a ready pilot entry point.
If B5 selects HNeRV, build a separate reviewed entry point with installed
dependencies, the fleet-claimed UUID, a step/time limit, and explicit resume
evidence. Preserve the legacy seed's recorded identity meanwhile. The DCVC
helper likewise launches unrestricted epoch commands; a pilot needs bounded
execution and verified checkpoint/resume behavior around the same production
training path. Model-specific pilot budgets and numerical acceptance targets
must be fixed before viewing pilot outcomes, after B5 selects the candidate.

No model or epoch count is selected by this document. Healthy diagnostics
enable designing a pilot; they do not establish final-training readiness.

## Local verification

The focused probe/provenance/codec/spec suite passed **79 tests, no skips**.
It includes a real timed-out child, retained earlier receipts, rejected
malformed output/nonzero exits, protected receipt paths, partial environment
recovery, and missing LPIPS weights. Installed lab imports, model execution
and training readiness remain unverified; no remote budget was consumed by
these local tests.
