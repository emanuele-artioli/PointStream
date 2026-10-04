# Background execution after budget approval — 4 October 2026

The background plan remains **experimentally incomplete**. The approved caps
were applied: 2400 seconds cumulative GPU allocation, 1500 seconds remote CPU
preparation. Existing attempts remain charged. A new provenance request was
published, but its runner never started. No new codec, drift, HNeRV inference,
training, optimization, dependency installation or extension rebuild ran.

## What ran and why it stopped

The gpu3/gpu5 doctor failed because gpu5 timed out. A subsequent gpu3-only
doctor passed, and both existing workers were fresh. The required native
five-minute heartbeat was activated before submission. Healthy workers and
unrelated jobs were preserved.

Published `20261004T132632Z-ff3a1bac`, a 340-second inventory on gpu3, from
clean scoped HEAD `5ca6e0f575fea6418ef785db2bbde0bda3672d3a`, without local
patches or untracked files. Its snapshot SHA-256 is
`4ccede614ae4f6a1661bb967bfe0c3143f8717a702baaf516ad6b5818a98c4c1`;
its frozen source SHA-256 is
`4ba0113fb7be7f6b18409d15d835fbb0e5720436c6f5933acc8b2c95d1c11bbc`.

Admission failed three times on the same shared CPU-claim lock:
`jobs/claims/cpu/gpu3.itec.aau.at.lock`. Each archived status has
`admission_rejected:true`, no child PID, and the same CPU-claim error.
Supervisors 224662, 224730 and 224807 never launched the inventory. The manager
releases the acquired device claim when CPU admission fails. Their
start-to-final-status-write intervals total 34.218 seconds; these are filesystem
metadata intervals, not exact physical GPU allocation durations.

After the repeated failure, only this request was canceled through `ps-fleet`.
Terminal `cancelled` was confirmed. Shared locks were not removed, and workers
or unrelated processes were not stopped. A fresh gpu5-only doctor then passed;
that makes gpu5 a candidate for later work, not authorization to bypass the
aggregate preparation cap. The heartbeat is now paused with all requests
terminal. No gpu5 inventory was submitted.

## Corrections retained on the scoped branch

- `5ca6e0f`: approved limits and `submit --source-worktree`, restricted to a
  checkout sharing this repository's Git common directory. The canonical
  client can snapshot the scoped branch without dispatching the dirty Desktop
  checkout. A real Git-worktree test verifies the selected commit and preservation
  of the canonical dirty file.
- `d84b684`: explicit `--snapshot-transfer-seconds` (at most 90) bounds archive
  transfer and extraction. Repeated `--snapshot-path` selects reviewed tracked
  dependencies and records them in snapshot provenance. Existing default
  behavior remains available for other tasks.
- `00511d1`: bounded, read-only admission inventory through
  `status --artifact admission-index.json`. It reads only status/supervisor
  JSON for at most twelve task-owned admission directories, records individual
  source hashes and modification times, rejects symlinks/oversized data, and
  explicitly identifies the exported index as synthetic.

The first new submission did not specify a transfer limit. The canonical
Desktop fleet file has a pre-existing uncommitted 900-second override; the
scoped tracked default is 300 seconds. That discrepancy should have been
checked before submitting. Publication had completed by the time the local
process check returned, so no transfer process was killed and the same job
was inspected. There is no per-operation timing receipt proving a transfer
itself exceeded 90 seconds. Do not describe this submission as having verified
90-second transfer enforcement. Future requests must pass the new explicit
flag. The unrelated override is preserved and is not selected for snapshots.

The dependency selection contains 40 committed files in a 727040-byte archive.
Isolated imports of the manager, runner, HNeRV helper and seed experiment
modules passed from that archive at `00511d1`. No native models were loaded;
this verifies snapshot completeness for imports, not installed lab dependencies
or model inference. Required model inputs/settings are unchanged by selection.

## Corrected preparation ledger

The earlier continuation proposal undercounted **publication before admission**.
Recovered immutable `ready.json`/dispatch records now establish those intervals.
Charge them conservatively instead of treating only admitted stages as preparation.

| Charge | Seconds |
|---|---:|
| Original first request ID to publication | 227.776 |
| Original second request ID to publication | 364.689 |
| Original first environment start to terminal | 205.670 |
| Original second environment start to terminal | 295.015 |
| Earlier artifact recovery wall time | 188.640 |
| Approved request ID to publication | 195.974 |
| Three rejected supervisor/status intervals | 34.218 |
| Admission-record recovery wall time | 15.068 |
| First publication-record recovery wall time | 6.051 |
| Conservative subtotal | **1533.100 (25.55 minutes)** |

Publication intervals include local archive work and network latency; exports
also include connection startup. This is a conservative elapsed ledger, not
an exact remote CPU-active measurement. Filesystem modification times and
host clocks are not an exact claim-release ledger. Doctor, worker-health and
management query costs have not been added to this subtotal. Those costs cannot
create more available time. It is therefore not safe to assert that a further
complete inventory fits the approved 1500-second preparation ceiling. Stop;
do not silently reset it, deduct failed attempts, or count only successful reads.

Allocation reservations remain **680 + 340 = 1020 seconds**, with **1380 seconds**
left under the 2400-second GPU cap. These remain conservative reservations,
not measured GPU-active time. The CPU preparation gate prevents using that
remaining GPU allowance now.

The new immutable receipts, synthetic admission index, corrected budget ledger,
source-selection preflight and checksum index are outside the repository at
`/private/tmp/pointstream-background-approved-20261004/`. Earlier receipt sets
and original remote evidence are preserved. The old planning ledger remains
as history; `budget-reconciliation.json` supersedes it.

## Concrete next step, not submitted

A lean 200-second B1 inventory (90-second smoke, 60-second remainder, 30-second
validator and 20-second overhead) preserves the same diagnostic parts and
input identities. It is a bounded pilot; it is not a promise that all parts
will pass on the lab filesystem. All subsequent work remains gated by actual
artifacts and substantive validation.

Existing 1020-second reservations plus B1 200, B2 450, B3 300 and B4 420 total
**2390 seconds**, within the existing 40-minute allocation ceiling. This
combination passed the local cap checker. The downstream schema check used
explicitly labeled local hash fixtures; no downstream job specs with invented
hashes were written. Real passing B1/B2 hashes are required before submission.

Continuing requires a new explicit **40-minute cumulative CPU preparation cap**,
including all previous attempts and recovery. The GPU cap would stay at
40 minutes, and every original per-job, per-file and model-stage limit would
remain. That additional authority has not been applied. Before any submission,
regenerate the absolute deadline, rerun availability checks, reactivate the
heartbeat, use the clean worktree and forty-file selection, and specify the
90-second transfer bound. Record start/end receipts for publication and data
preparation as they occur. Stop on another substantive failure or a budget that
cannot accommodate the next gate; do not replay this canceled request.

## Verification and decision

The final combined focused suite passed **220 tests, one skipped** in 65.10
seconds. The skip is the torch-dependent packet test because the Mac test
interpreter has no PyTorch. Tests cover all required fleet suites, bounded
exports, real scoped-worktree snapshots, selected dependencies, transfer
limits, malformed budget/inputs, source/checkpoint gates, packets and substantive
two-stage promotion. Isolated selected-snapshot imports and `git diff --check`
also passed. These checks do not establish GPU model correctness.

B0's visual review and corrected HNeRV labels stand. B1 still has no passing
current checkpoint/input/environment gate. B2/B3/B4 remain unrun; B5 is
inconclusive. Source mismatch, checkpoint degradation, inference state and
representation limits remain unresolved hypotheses. Do not select a model,
start training or publish a codec-quality claim from this execution.
