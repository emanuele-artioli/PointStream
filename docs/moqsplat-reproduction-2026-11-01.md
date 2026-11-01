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

The unpublished request is retained under the shared inbox. The task's separate
client state `/tmp/moqsplat-fleet-state-20261101` selects verified gpu6 for
management and source transfer, preserving the shared client configuration.
Single-job status in the installed release ignores the management-host environment
override; this limitation must not be interpreted as fleet-wide unavailability.

The ceiling is 3,600 GPU-seconds; conservative charges before the corrected
trace/configuration test total 660 seconds. No training or paid compute is used.
The MoQSplat reproduction worklog carries later job IDs, source/spec/input
checksums and results. All historical source/data/output artifacts remain intact.

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
