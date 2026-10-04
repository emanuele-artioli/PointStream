# Background smoke: final bounded attempt, 4 October 2026

The plan remains **experimentally incomplete**. The user approved cumulative
40-minute GPU allocation and 40-minute remote CPU preparation caps, retaining
all previous attempts. This attempt reached execution and recovered useful
provenance, but B1 did not pass. B2 codec, B3 drift and B4 frozen latent
diagnostics were not submitted. B5 cannot select a model or distinguish
checkpoint degradation, inference state and representation limits.

## Execution and preserved evidence

A fresh gpu5-only doctor passed, and existing workers were fresh. The native
five-minute heartbeat was activated before submission and paused after the
request was terminal. No worker restart, shared-lock removal, unrelated-process
cancellation, dependency installation, extension rebuild or training occurred.

Request `20261004T142355Z-99540d53` was published from clean scoped revision
`d68ad93c04bb493528e0ee16eb9913fcb6358716`. The reviewed forty-file dependency
selection was used, with explicit `--snapshot-transfer-seconds 90`; no Desktop
patch or untracked file was included. Snapshot SHA-256:
`9722e94878633f0e39fb8fc5da179f225431be3bc9b8cba59e4dd48d97ab47a0`.
Frozen source SHA-256:
`eacc6bc9a772bcd23b30ac60dc68e27f5f438d1b1407f8e3297153c3014ae168`.
The input seed helper identities remain pinned in the saved specification.

The spec reserved 200 seconds: smoke 90, remainder 60, validator 30 and
overhead 20. Admission succeeded on gpu5's RTX 6000 Ada, UUID
`GPU-7754164a-33cd-265c-3696-c20966d228fe`. The unrelated index-0 workload
was preserved. Supervisor 1546456 launched runner 1546463. The job failed
after the smoke command exceeded 89.999983 seconds; the saved terminal status
has exit code 1. No remainder ran.

| Completed part | Seconds | Result |
|---|---:|---|
| Saved preview | 0.708 | Passed metadata audit; earlier visual review stands |
| Primary frames 120..151 and masks | 3.129 | Verified identities and expected geometry |
| Image checkpoint | 0.255 | Verified SHA-256 and metadata |
| HT-L pretrained / factory001 s1 | 16.939 | Verified hashes, matching tensor layouts and separately recorded stage epochs |

Checkpoint SHA-256 identities:

- Image: `b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302`.
- HT-L pretrained: `934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3`.
- HT-L factory001 s1: `e2c6e1021916a5cbe00b6bc96205c2c539df4930e935c8b337d8f97901ed50d1`.

The independently saved `smoke/dcvc-check.json` reports all nine reviewed
DCVC files matching their reference hashes and successful import of the
installed CUDA extension. It records Python 3.12.14, torch 2.5.1+cu121, CUDA
12.1, and the extension's native path and SHA-256. This is an environment
receipt, not encode/decode or quality evidence.

The environment part has no completed record. Its DCVC check finished, but
the surviving receipts do not identify whether the subsequent HNeRV import,
FFmpeg/LPIPS check or reporting consumed the remaining allowance. Do not
attribute the timeout to a specific dependency without evidence. The final
`smoke/result.json`, `smoke/selected-inputs.json`, execution receipt,
`validation.json` and `gate.json` are absent. The partial manifest from the
completed HT-L part is preserved; it cannot replace a passing B1 gate.

All raw exports and their read-only SHA-256 receipts are outside the code tree
at `/private/tmp/pointstream-background-final-smokes-20261004/`. Original remote
artifacts remain under
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261004T142355Z-99540d53`.
Exports made through gpu3 read that shared filesystem; execution host identity
comes from the job's saved environment. Earlier evidence directories remain
unchanged. No failed request was replayed or migrated.

## Budget and disposition

Conservative GPU reservations are **1020 + 200 = 1220 seconds**, leaving
1180 seconds under the 2400-second ceiling. This is reserved allocation,
not measured active inference time. The runner's start-to-terminal status-write
interval is 107.884 seconds; it includes supervision and is not a claim-release
measurement. No model inference, native bitstream or latent extraction result
was produced.

The preparation ledger retains the previous 1533.100-second subtotal. It adds
180 seconds for previous doctor/worker management, 110 seconds for fresh
doctor/worker checks, 77 seconds as the measured publication wall-time upper
bound, 107.884 seconds for this supervised attempt, 117.870 seconds of measured
artifact recovery, and an 80-second conservative allowance for status/event
management. Total: **2205.854 seconds (36.76 minutes)**. Management allowances
are conservative bounds, not CPU-active measurements; the exact historical
CPU-active ledger remains unavailable. No earlier failure is deducted.

At most 194.146 seconds remain under that conservative preparation accounting.
A repeat of the 200-second pilot would already exceed the allowance before
publication, and the approach has just failed its substantive gate. Stop here:
do not replay the inventory, bypass provenance, or promote the remaining GPU
budget into model diagnostics. The next technical prerequisite is an
environment audit that preserves each probe's timing/result and completes
within a demonstrated preparation budget. This report does not authorize
another run or claim that such a configuration has been established.

The cap amendment passed 15 focused spec tests. The prior combined focused
suite remains 220 passed / one torch-dependent skip, and this report changes
no inference implementation. Local passing tests do not establish GPU model
correctness. The scoped implementation and factual execution record are
retained on `codex/demo-background-smokes`; no PR has been opened.
