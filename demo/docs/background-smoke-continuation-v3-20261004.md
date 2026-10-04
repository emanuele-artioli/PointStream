# Approved diagnostic continuation: 4 October 2026

The 50-minute cumulative GPU and 60-minute cumulative preparation amendment
was applied. The environment-first inventory ran and failed its provenance
gate. **The background plan remains experimentally incomplete and is not
ready for a training pilot or final training.** B2 codec, B3 drift and B4
latent diagnostics remain unrun; B5 has no model-quality conclusion.

## Execution and exact failure

The gpu5-only doctor passed; existing gpu3/gpu5 workers were fresh. The
five-minute native heartbeat was active before submission. No worker restart,
lock removal or unrelated-process cancellation occurred. No dependency
installation, extension rebuild, model training or model inference occurred.

Request `20261004T195542Z-f8714ffc` used a clean scoped snapshot of
`29d5f7150a7c6895251f459285b0c060a77cf82c` with the reviewed dependency
selection, explicit 90-second transfer limit and a fresh deadline. It reserved
260 seconds: environment smoke 90, inventory remainder 120, validation 30,
overhead 20. Publication wall-time upper bound was 43 seconds. Execution used
gpu5 index 1, RTX 6000 Ada UUID
`GPU-7754164a-33cd-265c-3696-c20966d228fe`; the unrelated index-0 workload
was preserved. Supervisor 1567994 launched runner 1568006.

The `env` part failed after **30.695 seconds**, with
`TimeoutError: git exceeded 30s`. Its saved result and selected-input manifest
are complete and hash-verified, but the manifest's environment is empty.
Validation published `passed:false`, with `all_parts_passed:false`; the
campaign stopped and terminal exit code 1 was confirmed. There is no passing
gate and the remainder was never launched. A valid manifest container with
no verified environment cannot authorize inference.

Both source-diff artifacts are absent. At the frozen execution revision,
the first environment operation is
`git -C <DCVC_ROOT> diff --no-color HEAD`, and its source-diff file is written
before any later Git operation. Together with the explicit Git timeout, this
locates the failure at the first DCVC diff command. This attribution uses
the verified execution order and missing artifacts; this revision did not
save a separate Git command receipt. It does not establish why Git was slow,
nor implicate the HNeRV, FFmpeg or LPIPS probes, which were not reached.

The saved DCVC check from the earlier attempt remains useful historical
evidence, but is not substituted for this failed current provenance gate.
No checkpoint hash, extension import or metric result is newly established.

Snapshot SHA-256:
`b754f14a6e0492e82a825e3da1bd1dec760042a330288a7a46f059b75a0e2441`.
Frozen source SHA-256:
`63d4c0557d61f7033f75d51d0eed76a6f2c2d71c301549aa6aff841c544271cb`.
Selected-input manifest SHA-256:
`905f4e5803236b7b963426ff36c93ca5363e1805f8b0c7b39ce29b133fd41fe6`.
Raw source/spec/input/environment identities, result, failed validation,
dispatch, terminal status, and export receipts are preserved outside the
repository at `/private/tmp/pointstream-background-continuation-v3-20261004/`.
Original remote artifacts and earlier receipt sets remain unchanged.

## Accounting and disposition

GPU reservations remain charged at **1220 + 260 = 1480 seconds**, with 1520
under the approved 3000-second cap. These are reservations, not measured
active GPU time. Preparation accounting retains the earlier 2205.854 seconds
and adds publication 43, publication-to-terminal metadata interval 205.380,
artifact exports 20.297, and a conservative availability/management allowance
of 90. Total: **2564.531 seconds**, leaving 1035.469 under the 3600-second cap.
The elapsed metadata interval includes queueing/admission and supervision;
this is conservative accounting, not CPU-active or exact claim-release time.
No failed charge was deducted or budget reset.

There is budget remaining, but the failed gate stops promotion. No request
was replayed, and no additional inventory or model diagnostic was submitted.
All watched jobs are terminal; reported events were acknowledged and the
heartbeat paused. Increasing a budget would not establish source provenance.

The post-run local correction routes each repository's diff, HEAD and status
command through the same recorded probe helper. Admission/outcome receipts
now name the repository and operation, such as `dcvc-git-diff`, preserving
the exact command, elapsed time and timeout. The 30-second allowance and
source-selection contract are unchanged. This improves identification of
future failures; **it does not fix or explain the lab Git timeout** and has
not been tested remotely.

The budget amendment passed 15 spec tests. The subsequent focused
probe/provenance/spec suite passed **50 tests, no skips**, including a
repository-specific timeout and preserved failed part. The previously
reported 79-test run remains evidence for the earlier audit correction.

At most two technical follow-ups are justified: establish bounded repository
provenance collection on the lab's shared storage without omitting required
tracked source changes; and resolve exact fit/validation/final-hold-out source
offsets before designing any training pilot. Both require evidence, not an
epoch-count choice. The remaining training gates and legacy training-helper
limitations are recorded in `background-training-readiness.md`.
