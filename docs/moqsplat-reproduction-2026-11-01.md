# MoQSplat reproduction validation — 2026-11-01

This is maintenance of the separate `emanuele-artioli/MoQSplat` project, using
PointStream's existing fleet. It is not PointStream codec evidence.

The user authorized real GPU validation of the external rendering/subscription
fixes, reconstruction of missing transport when necessary, and a complete
reproduction path. Code lives in an isolated MoQSplat checkout; the adapter here
stages a checksummed external source archive and real inputs, uses an explicit
host-local interpreter, and runs the same bounded pipeline at both fleet scales.

The combined gate runs CPU regressions, source-scene CUDA rendering, the legacy
TCP client, and separate native MoQ relay/publisher/subscriber processes. It
requires real nonblack CUDA images and byte-exact source attributes. It retains
partial artifacts and reaps only its own relay/publisher children. CUDA compiler
and header paths are explicit; caches and intermediate files stay host-local.

## Preserved attempts

| Job | Outcome |
|---|---|
| `20261101T131954Z-e5f936c7` | Failed CPU import timeout on slow NFS; no CUDA image |
| `20261101T134114Z-eeb9ec59` | 50/52 tests passed; missing FFmpeg plugin and a new assertion mismatch prevented promotion |
| `20261101T135740Z-6f49d789` | Interrupted source transfer; inspected unpublished, no ready marker/owner/run; retained, never replayed |
| `20261101T144900Z-dc0598d9` | 62 tests passed; CUDA compilation found a missing header path; no numerical evidence |
| `20261101T153034Z-1dbe7112` | 76 CPU tests; CUDA compiled, then gsplat packed-background API failure |
| `20261101T154427Z-9679149a` | Snapshot omitted the adapter module; failed before the experiment |
| `20261101T160216Z-31d696de` | 77 CPU tests; full-scene CUDA oracle and legacy TCP client passed; native group-size limit failed |
| `20261101T161843Z-425ac71f` | **Complete:** 82 CPU tests, one-frame gate and three-frame full run passed, including native three-process QUIC |

The unpublished request is retained under the shared inbox. The task's separate
client state `/tmp/moqsplat-fleet-state-20261101` selects verified gpu6 for
management and source transfer, preserving the shared client configuration.
Single-job status in the installed release ignores the management-host environment
override; this limitation must not be interpreted as fleet-wide unavailability.

The ceiling is 3,600 GPU-seconds; final conservative charges total 1,440 seconds.
No training, performance sweep or paid compute was used. The correctness decision
is complete; no task-owned timers remain.
The MoQSplat reproduction worklog carries later job IDs, source/spec/input
checksums and results. All historical source/data/output artifacts remain intact.

## Final correctness result

The final native client received all 6,131,954 Bicycle Gaussians in five layers,
with exact original attribute hashes, 45 changed priority updates across 72 pose
samples and zero additional payload requests on cache revisit. Progressive CUDA
rendering overlapped active transfers. Three complete received-data renders
agreed with direct source renders within two RGB channel levels on the 0–255
scale at 320×240; the images were visually inspected. The legacy TCP client and
source-scene oracle also passed both scales. These short loopback checks do not
establish paper performance or a complete trace session.

The native library hard-caps groups at 32 MiB. The repair segments complete
application layers into at most 16 MiB transport groups and validates sequence,
completeness and whole-layer hashes. Actual QUIC regressions cover payloads
larger than 32 MiB and late recovery after 33 seconds idle; indefinite relay
retention remains unproven. The semantic publisher/VMAF pipeline was not rerun,
and GPU/RAM watermark swapping and standalone moq-relay interoperability remain
unverified.

The fleet ran its substantive validator at the smoke gate. The identical
adapter validator also passed locally on the exported full summary after job
completion; that was post-run verification, not another fleet promotion gate.

Validated external code is MoQSplat `42b6bbc`, archive SHA256
`d64052cbee3e821d6dc83fb0fd767f883ec119c358165ce9bb011706a108d2c8`;
the PointStream adapter snapshot at `71153358` is
`793af4d68595848369d8b3bc2d99866c03cc953245a41424710e14c47f100ff7`.
The full artifact archive SHA256 is
`1aef56423afffc14e63e7f019fb2c3308e8fc49da32dd9acbd0747aacab6d4c0`.
Evidence lives at the shared fleet inbox under the final job ID and in the
[public validation record](https://github.com/emanuele-artioli/MoQSplat/blob/9a2aaa49a56d9ca44c25bf8f36cf71925303c153/reproduction/VALIDATION-20261101.md).
[MoQSplat PR #3](https://github.com/emanuele-artioli/MoQSplat/pull/3) is ready for
review; the original custom wire protocol and published network results have
not been recovered or reproduced.

## Findings that invalidate historical trace reuse

The original EyeNav CSV converter omitted the scene-world axis conversion and
combined the wrong signed FOV fields. The pinned raw CSV was verified by its
full Git blob and SHA256, and a separate corrected trace preserves four
asymmetric angles. Existing JSON is historical evidence and is not overwritten.

The four public repositories' five branches and releases were inspected. None
contains the paper's customized native transport. The new native MoQ layer-track
implementation is an independent functional reproduction; it is not evidence
that the original custom subgroup wire protocol or published bandwidth results
have been reproduced.

## Local validation

With the declared project dependency pins: Ruff passes, mypy passes for all 318
configured files, and pytest passes 1,215 tests with 24 skips. The initial macOS
test exposed a protected-store alias mismatch through `/System/Volumes/Data/home`;
canonicalizing both candidate and store targets fixes that bypass. The existing
hook regression now passes on macOS. Requested Codex tier overrides and observed
runner discipline are recorded in `docs/sessions.md`; effective model identities
remain unverified because the runtime did not expose them.
