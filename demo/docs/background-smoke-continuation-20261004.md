# Background smoke continuation — 4 October 2026

The background plan is **not experimentally complete**. The implementation
is committed and tested. B1 has not verified any checkpoint, so B2 codec pairs,
B3 drift and B4 trained HNeRV packaging have not run. B5 remains inconclusive.
This report updates the earlier recovery report without replacing its history.

## Recovered evidence

Added bounded read-only artifact export to the canonical `ps-fleet status`
client. This required no worker restart, new allocation, inference, checkpoint
load or dataset scan. Original results remain unchanged. Local exports and
SHA-256 transport receipts are read-only under:

`/private/tmp/pointstream-background-continuation-20261004/`

| Receipt directory | Job | Export wall seconds | File-reader seconds |
|---|---|---:|---:|
| inventory-1 | 20261004T082204Z-4515f8fa | 43.063 | 3.443 |
| inventory-2 | 20261004T083413Z-e373aa74 | 38.414 | 2.967 |
| preview | 20261003T101656Z-61d46aea | 81.175 | 9.115 |
| inventory-2-management | 20261004T083413Z-e373aa74 | 25.987 | 0.942 |

Total export wall time is 188.640 seconds; in-process remote file-reader time
is 16.466 seconds. Reader time excludes interpreter/connection startup and
cannot substitute for the entire remote preparation cost. A sandbox DNS
failure occurred before the last export; the same standalone entry point
succeeded with network escalation. That failed attempt allocated no GPU.

The first inventory's execution receipt reports 158.297 seconds. Its ledger
shows an environment adapter timeout taking 105.898 seconds, a passing preview
metadata part taking 26.252 seconds, then a failed input-mask hash and blocked
image/HT-L checkpoint parts. Its result and validator explicitly fail.

The second inventory's ledger shows a passing preview metadata part taking
33.212 seconds, then an input-frame hash failure taking 113.936 seconds. The
parent smoke command timed out after 170 seconds. Its execution/result/input
manifest/validation files are absent, not successful empty results. Its
preserved dispatch records the exact selected source hashes and explicitly
excludes the unrelated Desktop `experiments/jobs/fleet.py` patch.

**Neither inventory completed a checkpoint part or a passing selected-input
manifest.** This is now independently established from exported artifacts.
The old timeout requests did not produce reliable wall-clock bounds: reported
elapsed operations exceeded their requested allowances. Removing repeated
NumPy imports from hash/copy children is a plausible corrective change,
covered by CPU tests; its performance on the lab filesystem is still unknown.
Do not claim that increasing a budget alone resolves these failures.

## B0: reviewed previews and corrected labels

Recovered the manifest and all eight JPEG contact sheets; reviewed offsets
0, 2, 4 and 6 for both factories. The manifest SHA-256 is
`9e9e17c9bd37c5b3756fa94e92b7d63a643780362628471c561e261be20f3abe`.
It records eight-frame cuts beginning at hold-out offset 120, 30 fps, QP 21,
with separate factory001 clip-3 and factory002 inputs.

Large human regions remain visible in the displayed filled inputs. These
are codec reconstructions of those filled inputs; the panels do not establish
successful empty-background reconstruction. AV1 240p appears softer than
1080p. DCVC resembles the displayed source more closely than HNeRV, which
visibly smears structure. The JPEGs are scaled contact sheets; native output
sizes and frame identities cannot be independently verified from them.

The following numbers are **historical manifest values**, not newly measured
rate-distortion results. PSNR is the helper's mean RGB PSNR versus its filled
input across eight frames. No LPIPS or pretrained/fine-tuned pair is present.

| Method | Factory001 kbps / dB | Factory002 kbps / dB |
|---|---:|---:|
| AV1 240p | 70.35 / 30.487 | 93.36 / 28.517 |
| AV1 1080p | 275.13 / 34.492 | 415.86 / 34.142 |
| DCVC LD | 401.94 / 36.716 | 407.10 / 36.041 |
| DCVC HT-S | 303.81 / 34.144 | 378.36 / 33.065 |
| DCVC HT-L | 289.80 / 34.018 | 368.67 / 33.258 |
| HNeRV, setup-inclusive estimate | 26716.44 / 23.725 | 24758.28 / 23.130 |

HNeRV's old panel labels combine estimated model and latent bits; they are
not ordinary latent streaming rates or actual encoded-byte measurements.
Original panels were retained without relabeling their contents. Native
frames/bitstreams remain unexported, not proven absent. The historical helper
retains paths under `previews`, `streams`, `inputs` and `hnerv`; its manifest
omits the historical input/checkpoint SHA-256 values. Current checkpoint
hashes cannot retroactively establish which checkpoints that job loaded.
Consequently B0's panel/label review is complete, but native-artifact and
historical checkpoint provenance remain unresolved.

## Availability and budget gate

A fresh standalone doctor restricted to gpu3 and gpu5 **passed**, including
common filesystem visibility and a single winner for the atomic claim check.
The saved report is
`/Users/manu/.pointstream/fleet/jobs/checks/cba2ecde4a9d4ab7bdebf695c18ea88b.json`.
Workers were fresh on both hosts. gpu3's A6000 and gpu5's second RTX 6000 Ada
were eligible in that inspection. gpu2/gpu6 timed out and gpu4 was measurement
locked. A later request must check availability again and let admission select
the device. The native five-minute recovery heartbeat remains paused because
no request is running; register/reactivate it before any new submission.

Keep the two failed 340-second reservations charged: **680 of 1800 seconds**,
leaving **1120 seconds**. These are conservative submitted reservations, not
measured GPU-active seconds. Supervisor start-to-terminal intervals are
197.304 and 283.966 seconds. Environment-record start-to-terminal intervals
are 205.670 and 295.015 seconds. The second job's GPU claim was acquired at
1791103576.8076031 and its CPU claim at 1791103578.4748726. Exact release times
are not in the recovered records. Do not equate stage execution duration,
terminal status time or later worker updates with an exact allocation ledger.

The existing cap checker charges inventory reservations to its 600-second
preparation allowance. Conservatively charging the two existing requests
already gives 680 seconds, so it rejects any additional inventory request.
Actual elapsed receipts do not prove that another 340-second inventory fits
within 600 seconds. Even ignoring cleanup and recovery overhead, the two
supervisor intervals alone leave less than 119 seconds. No new GPU request
was submitted and the previous requests were not replayed.

## Concrete continuation requiring a budget amendment

The reviewed implementation's default, gate-controlled continuation is:

| Request | Total reservation | Prerequisite |
|---|---:|---|
| Corrected B1 inventory | 340 s | Passing doctor, heartbeat, reviewed source snapshot and fixed input identities; stop on another failure |
| B2 codec pairs | 450 s | Passing B1; HT-L pretrained/fine-tuned first, then remaining primary pairs and AV1 |
| B3 LD drift | 300 s | Hash-pinned passing B2 LD pretrained/fine-tuned pair |
| B4 HNeRV packets | 420 s | Passing B1; primary six-bit result before optional factory002 six-bit remainder |
| New reservations | 1510 s | No training, optimization, downloads or rebuilds |
| Existing plus new | 2190 s (36.5 min) | All failures and waits remain charged |

This sequence cannot fit the original 30-minute cap. Request authority to
raise the **cumulative allocation cap to 40 minutes** and the **remote CPU
preparation cap to 25 minutes**, counting the existing attempts and recovery.
Keep all original per-job, per-file and model-stage caps. These proposed caps
are not applied in code or used to submit work until explicitly approved.
The extra margin is a ceiling, not a commitment to spend it. Retain an
aggregate preparation ledger including manager/adapter startup and waits;
stop before a request that cannot fit its remaining allowance. Reassess the
source snapshot and stage overhead before submission. Missing dependencies,
wrong strict-load configuration, further timeout or failed validation stops
the affected path; approval cannot guarantee a passing scientific result.

## Verification and disposition

Artifact export plus the four required fleet suites passed **112 tests**.
The export tests exercise the actual fixed reader against temporary files,
checksums, safe roots, traversal/symlink rejection, missing files, truncation,
read failures and non-overwriting read-only exports. These are infrastructure
checks, not model evidence. The background focused suites were rerun for
the continuation: **97 passed, one skipped** in 47.53 seconds. The skipped
packet test requires PyTorch, which remains unavailable in the Mac interpreter;
no installation or substitute inference was attempted. `git diff --check`
also passed.

The export implementation is pushed as `f2b730b` on
`codex/demo-background-smokes`, following the tested runner fixes `6ab31df`.
The original Desktop fleet edit and unrelated work remain preserved. Only
the reviewed export client files were copied there to enable the canonical
entry point; their implementation is retained on the scoped branch.

Until the budget amendment and a passing B1, checkpoint/source mismatch,
inference state, training degradation and representation limitations remain
open explanations. No model should be selected or retrained from this review.
