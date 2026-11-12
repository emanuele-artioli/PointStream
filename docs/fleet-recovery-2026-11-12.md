# Fleet recovery and storage decision — November 12, 2026

The user restored terminal SSH access and requested a decision between resuming
research and cleaning/moving work to avoid NFS problems. This phase used bounded
metadata checks, not model inference. No folder was removed or relocated; no
worker, server, model, dataset, completed result or manuscript was modified.

## Recovery and storage findings

- Canonical status is available again. The two previously unknown Ball/W8
  requests are **expired without assignment/execution**. Root independently
  checked canonical status, physical source archive and normalized spec hashes,
  preserved directory/claim evidence and expiry-worker identities. Both terminal
  events are acknowledged. See the [closeout](experiments/fleet-recovery-closeout-20261112.json).
- Canonical all-host inspection reached all six hosts and found fresh worker
  heartbeats with source-archive support. One gpu5 Ada was idle at that snapshot;
  all gpu2/gpu3 A6000s required by the frozen Ball pilot were occupied. These are
  observations, not reservations or guarantees of later availability.
- Separate host-local `/local` filesystems on gpu1/2/4/5/6 had **1,353–6,732 GiB
  free**. Shared NFS had approximately **20,164 GiB free**. Root filesystems had
  only 12–56 GiB free and remain forbidden for large scratch/cache files.
- A tiny 4-KiB write+fsync on the verified separate local filesystem took
  0.72–1.48 ms, and repeated reads 0.007–0.049 ms. The first small existing NFS
  state-file reads took 1.88–14.48 ms, with much faster repeated cache hits;
  gpu2's home metadata lookup took 146.53 ms. These narrow, mostly warm-cache
  observations do not measure bulk throughput or prove the cause of the outage.
- A separate gpu3 storage probe timed out locally before its start receipt.
  Other gpu3 metadata attempts also stalled, although canonical inspection and
  status could reach it. Preserve that uncertainty; do not call gpu3 reliably
  healthy or replay/signal the unknown read-only process.

## Decision: resume local staging, preserve the archive

There is no capacity-based reason for broad cleanup or wholesale migration.
Retain NFS as the durable source of truth for datasets, models, archives and
published evidence. Keep active source extraction, packed Python environments,
staged inputs, caches, scratch, Python bytecode and temporaries under
`/local/users/emanuele/pointstream` on each host. Do not use `/tmp`, `/var/tmp`
or the root filesystem for them. The existing fleet already implements this
split; inspect each job's `staging.json` to verify actual paths and modes.

The frozen Ball spec declares 32 GiB local storage and the packed MAIN runtime;
the W8 spec declares 40 GiB and the packed JCV runtime. Both use source archives
and verified staged inputs. Preserve these contracts. Source, environment or
numeric-path changes require new qualification rather than a claim of equivalent
evidence. Worker control/state and cold archive reads still depend on NFS; moving
scratch does not eliminate that dependency or repair a server-side login issue.

Do not run recursive `du`/cleanup over NFS merely to tidy it. Do not remove
failed scratch, existing caches, checkpoints, legacy environments or live editor
servers without exact ownership, dependency and recoverability checks. A future
space-driven cleanup should enumerate candidates and preserve recoverable state;
it is not required to resume at the measured capacities. Five tiny uniquely named
local diagnostic files were retained for audit; their exact paths are in raw
receipts, and no existing file was overwritten.

## Scientific admissions after recovery

The Ball corrected-source 16-frame parity/lifecycle-cost gate remains useful.
Prepare at most one new bounded request with the same scientific source and
caps, after fresh source/input/doctor/readiness review. The old expired request
is closed, not replayed. No 48/292/586/full-30 stage follows automatically.

**Hold further W8 compute for the full-witness objective.** After the final
unassigned expiry, its parent has 929 CPU seconds left (995 in the workstream,
including 66 outside that parent). An illustrative extrapolation of the accepted
1/3-step sampling CPU measurements gives about 1,150 CPU seconds for two full
sampling phases alone, or 1,724 with the required 1.5 margin, before qualification,
loading, composition, scoring and audits. This is not a measured lower bound or
proof of physical impossibility. It is insufficient evidence of a complete fit.
Another 550-CPU pilot would leave 379 in the parent and could only establish a
narrow operational result. Do not spend it merely to approach an unfundable
endpoint. Reconsider a concrete reallocation only after protecting the Ball
forecast, or define a separately justified narrowed endpoint. No Ball-to-W8
transfer is authorized by this decision, and the fresh-witness limitation remains.

Current charges are **19,362 GPU / 48,391 workload CPU seconds**, including the full
400/150 external expiry charges, 20 shared recovery-diagnostic CPU seconds and
50 Ball readiness CPU seconds and400CPU for the subsequent prepublication stop. Aggregate administrative CPU was not measured.
Ball remaining nested reservation: 7,920 GPU / 1,850 CPU; W8: 4,550 / 929.
Global authority remains 24 GPU-hours / 28 CPU-hours. No paid server is needed
for this recovery decision and none has been purchased or authorized.

Raw snapshots: `/Users/manu/.cache/pointstream-evidence/fleet-recovery-20261112-a1/`
(65 files, 110,170 bytes plus manifest). This is an immutable diagnostic snapshot;
subsequent admissions need new records. The tracked closeout lists raw SHA256s.
Older uncertain read-only probes remain preserved without teardown claims.


## Actual bounded resumption

The first launch stopped on gpu1 TCP timeout during the repeated full-pool
safety check, before ID allocation; no unknown publication was replayed. A
subsequent **isolated local** client config with participating hosts gpu5/2/3
passed every participating-host visibility and one-winner atomic gate, using
gpu5 for management/upload and the original gpu2/3 A6000 workload pool. Source,
inputs, validators and local32GiB/runtime contracts are unchanged. Global
client config and shared workers remain unchanged.

Actual request **`20261112T172806Z-8709406d`** is pending, deadline
2026-11-12T17:52:06.008295UTC, original1000GPU/1200CPU maximum nested in its
parent. One canonicalwait2500 is authorized; no automatic further queue/scale.
The [Ball journal](experiments/k5d-ball-benchmark.md) records its handle and outcome.
A later small distinct readiness probe also confirmed gpu3 has a separate local
disk with2,676GiB free; its earlier unknown read-only probe remains unresolved.
