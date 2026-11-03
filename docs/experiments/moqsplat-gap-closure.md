# Experiments: MoQSplat client and reproduction gap closure

### 2026-11-01–03 — prospective gap-closure protocol

User authority (1–3 November 2026): implement continuous rendering, reconnection,
client memory management, clean installation and broader network tests; merge
validated changes and draft replies for the user to send. No extended-paper work
and no messages sent to colleagues. External code is in MoQSplat PR #3.

## Goal, scope and stop conditions

Goal: a usable, tested native MoQSplat client and an honest reproduction release.
Server preparation can remain offline and resource intensive. Keeping a link
busy is valid; the delivery objective is useful viewport quality at available
bandwidth, not minimum transferred bytes. Preserve all source data and prior
results. Recovered original artifacts and new reimplementation evidence remain
distinct.

Stop when the implemented paths pass representative checks, external-only
limitations are documented, and the validated release and email drafts are
ready. Stop compute on a failed gate or exhaustion of the remaining budget;
revise before any promotion. Do not silently claim internet/WAN qualification
from a local impairment proxy, or fresh-OS qualification from a new environment.

## Hypothesis, alternative and existing oracle

The existing full-source CUDA renderer provides an upper bound on complete
received-data image correctness. Its valid evidence is job
`20261101T161843Z-425ac71f`, 6,131,954 original Bicycle Gaussians and corrected
EyeNav trace. Source/trace SHA256 identities are in the existing reproduction
record. New client components preserve those bytes, camera convention and
renderer. Budget-limited progressive images are not expected to equal the full
source image; base dependency, nonblack output and memory bounds gate them.

Hypothesis: disk-backed completed layers, bounded GPU residency and continuous
rendering permit view changes without waiting for every enhancement; reconnect
can retain verified completed layers and retry incomplete layers safely.
Alternatives: SDK reconnect retains stale catalogs, interrupted groups become
cached, scene-dependent rasterizer workspace defeats memory estimates, camera
metadata makes semantic publisher outputs invalid, or dependency inheritance
hides missing packages.

## Decision rules and prospective bounds

- CPU/native regression suite passes. Real relay interruption recovers identical
  complete content, rejects changed catalog identity, and never caches partials.
- Independent environment has system-site-packages disabled; package imports
  resolve there and `pip check` passes. Actual GPU execution must use it before
  calling that environment qualified.
- Continuous received-data rendering produces finite nonblack CUDA frames during
  reception and after user/trace pose changes; report actual achieved frame rate,
  responsiveness, stalls and memory, without promising 30 fps.
- Completed disk cache avoids new payload requests on revisits/reconnects.
  GPU memory policy protects base layers, evicts enhancements and restores them
  without network. Measured Torch allocation must meet the selected budget after
  adaptation; record transient overshoot and excluded driver allocations.
- Semantic publisher uses real existing YOLO weights, full real scene and a
  source-identified camera. Require detections, nonempty semantic output, exactly
  five DELTA layers per selected group, valid counts/hashes and native consumption.
- Impairment tests use three initial settings only: unrestricted, constrained
  bandwidth/latency, and bounded drop/outage. Multiple subscribers and longer
  sessions have bounded duration and recorded counters. A user-space UDP proxy
  is a controlled network test, not an independent internet client.

Pixel bounds: uint8 [0,255], finite and at least one nonblack pixel. Exact wire
and source-array hashes must agree. Frame counts and priority/residency events
are nonnegative and trace/command identified. Loss recovery may retry incomplete
layers from the beginning; do not claim byte-range resume.

## Budget and dispatch

Original ceiling 3,600 GPU-seconds; earlier conservative charge 1,440. This phase
retains **2,160 GPU-seconds remaining**, no training or paid compute. CPU setup
and tests are bounded; all remote scratch, caches and the independent environment
are under `/local/users/emanuele/moqsplat-gap-closure-20261101`, never server `/tmp`.
No new GPU job submitted yet. Initial smoke at most 600 seconds; only promote
after output/device/resume checks pass. Later cases are sized from its measured
cost and save finished cases individually. Each awaited batch stays within
45 minutes. Use only `scripts/ps-fleet`, inspect the recorded job ID before
acting, preserve handles, and never replay uncertain submissions.

Prior final job: `20261101T161843Z-425ac71f`, COMPLETE. New jobs: none.
New GPU charge: 0 / 2,160 seconds. No recurring wakeups.

## Pre-dispatch findings, 3 November

Independent Python 3.10 environment: system-site-packages=false, pip check
clean; Torch 2.10.0+cu128, gsplat 1.5.3, moq-rs 0.5.2 / moq-ffi 0.4.20.
Resolved conflicting OpenCV providers by selecting opencv-python 4.10.0.84.
CPU3 frozen source a3b9b47e1fc6af13cb4ba4defdf6cca9ae6ef8d380fb5ee1e8a0715ae7362f59:
**111 tests passed in 11.33 seconds** in that independent environment. These are
regression results, not GPU evidence. A subsequent reviewed qualification-only
path fix places semantic output under bicycle_opacity/seg as the VMAF driver
requires. Final frozen source is rechecked by the actual fleet smoke.

All six hosts were inspected. The newly installed independent environment is
host-local on gpu6; its Ada devices have other workloads. Use explicit shared
GPU correctness mode with 16,000 MiB request and 8,192 MiB safety margin.
Timing/FPS and responsiveness under this contention are diagnostic observations,
not an uncontended performance claim. No outside processes or caches are touched.
Protocol input manifest SHA256:
638183a70501192cf61de132e7b40fcc9bd7b6ddac62ad16270ac56374a1ef4f.

PointStream adapter checks: Ruff passes; mypy passes (381 source files);
full pytest 1,450 passed, 30 skipped. Global legacy MoQSplat Ruff is not clean
(564 existing/legacy diagnostics); focused new/native files pass Ruff.

### Publication attempt 20261103T150639Z-1ceedad4

Submission returned UNKNOWN: inbox publication rejected staged weights outside
the Datasets fleet root (weights correctly live in Models). Do not replay.
Inspect this ID before preparing a corrected request; it is not GPU evidence.
Source revision 2f371f631c8dbe83944e03c58d2f5940db8931c3; source archive
502b09b34cb41448571cd8e01e290e0810ef57e8f22cec1745d5589dd789ed06.

The status command rejected that ID as unpublished. Read-only inbox inspection
found only source.tar and events, no ready/spec/owner/run: no runnable request,
no GPU charge. Preserve it. Corrected adapter hashes weights in Models in place;
only scene, trace, camera and source are staged under Datasets. Prepare a new
request; this is not replay of an uncertain runnable request.

### Published GPU job 20261103T150751Z-2f6bb31b

Pending at publication, gpu6 Ada shared correctness class. Original protocol
ceiling 3,600 GPU-s; prior conservative 1,440; request reserves at most 2,000
of 2,160 remaining (600 smoke, 1,000 full, validators/staging within ceiling).
Deadline 2026-11-03T15:49:26.945258Z. Adapter f1b5d6cbccc95b72b8f17537e4ec9176f695b627;
MoQSplat 2f371f631c8dbe83944e03c58d2f5940db8931c3, archive
502b09b34cb41448571cd8e01e290e0810ef57e8f22cec1745d5589dd789ed06.
Spec cf90122079ca8062962d2f453562a32077d8af5770fe10687bf24425f3cbbb44.
Results: fleet/inbox/20261103T150751Z-2f6bb31b under the external jobs root.
Bounded native wait handle to be persisted below; no recurring model wakeups.

Bounded wait process handle 94443; output
/tmp/moqsplat-gap-gpu2-wait-20261103.json, maximum 2,400 seconds. The fleet
reports completion-first monitoring; this client has not independently proved
that a command completion resumes sampling. Remain attended and preserve the
handle; do not add timers or repeated polling.

Initial wait ended after 7 seconds on declared shared-GPU contention event
d19565dcf0045eeda7d269a99e803e83 (already expected, acknowledged). No claim
changed. Start a new single bounded wait after the event; retain its handle.

Second bounded wait handle 97158, output
/tmp/moqsplat-gap-gpu2-wait-after-ack-20261103.json, max 2,100 seconds.
Independent read-only review found and fixed the semantic seg-directory mismatch
before freezing GPU1; final read-only transport audit found no further concrete
blockers. No remote agents were used.

### First executed gate failed before rendering

20261103T150751Z-2f6bb31b terminal FAILED; native wait returned terminal attention.
111 CPU tests passed in 14.24s. Oracle CLI rejected unsupported --width/--height
from the new qualification harness (oracle already uses fixed 320x240). No
semantic or live qualification passed. Actual supervisor 30.44s plus staging;
charge conservatively 60 GPU-s, phase remaining 2,100 (aggregate 1,500/3,600).
All outputs retained. Remove those unsupported harness arguments; unchanged
production render/transport paths, then a new bounded gate with new source hash.

### Revised gate 20261103T151437Z-d50b3f16

Published/pending, new source ff4f08d04062429ee8ae6022c122fb0ef8e877f7, archive
2292a2f78eff6254ec28c0328b264799321be0548f3a28ee747e8ee79c194a1b.
Spec 2255957db2c8094b05209e0912ce4327a3b6d67394c70ff713e68888628a16fb; 2,000-second reservation within 2,100 remaining.
Deadline 2026-11-03T15:57:33.696115Z, same 600s smoke / 1,000s full bounds.
Only oracle CLI argument correction; no failed gate promoted.

## Prospective independent-machine network check

CPU-only Mac-to-gpu6 native QUIC, 65,536 real source Gaussians explicitly
selected for a bounded network check (not image quality or full-scene performance).
Temporary relay binds ports 24533/24534 for at most 140 seconds each. No firewall
changes, no public deployment, no GPU claim. External checker must verify source
attributes, QUIC counters and cache revisits; server requires that saved proof.
If UDP is blocked, preserve failure and request a reachable trusted endpoint or
VPN/firewall assistance; do not tunnel TCP and call it native WAN delivery.
CPU budget 480 seconds, 2 threads; final stage preserves each client's proof.
No human needed for this network-only test.

### Cross-host CPU job 20261103T151800Z-053ccd17

Published/pending, CPU only (0 GPU-seconds), 480-second CPU ceiling, ports
24533 smoke / 24534 full. Source archive same GPU2 revision, source subset
65,536 explicit. Adapter16090b7b, proof must be written from this Mac by native
checker. No GPU qualification claim. Request identity e75f918a78a2216d06a0fc402d393441b1927a1250432174bbe485d65fea6d9a.

### Semantic preparation passed; orbit CLI import failed

20261103T151437Z-d50b3f16 terminal FAILED, supervisor 191.44s (conservative
240s charge). Full real-scene source CUDA oracle and original semantic publisher
completed in the independent environment. VMAF stopped at orbit generation:
upstream helper imports missing gsplat_object_segmentation rather than the
actual gsplat_clustering_vmaf module. Fix the import and add actual --help
subprocess tests for all three publisher CLIs; isolated camera math tests did
not exercise imports. Preserve all prior outputs. New CPU4 regression must pass
before another GPU gate. Phase 300 GPU-s charged, remaining 1,860; aggregate
1,740/3,600. Revised reservation 1,800, smoke600/full850; no promotion on failure.

Cross-host 20261103T151800Z-053ccd17 FAILED with no external proof. Initial Mac
venv lacked NumPy (setup issue, fixed using binary numpy2.5.4 for Python3.14).
Retry began15:21:02Z, endpoint stopped15:21:12Z: only10s overlap, insufficient
for a conclusive network-block attribution. Server ready15:19:32Z, publisher
QUIC session confirmed. Preserve failure, charge150 CPU-s of480 ceiling.
A fresh synchronized attempt has330 CPU-s remaining, awaits readiness before
connecting, same source/scene subset and ports; never replay old request.

### Final revised GPU gate 20261103T152241Z-0989d383

Published/pending, source3ff6f9a35a88db681e89c9d5ac95962df5e80805, archive
a4312a7aeb3adb9bef74fa8d852543302df97dbdedd1518afdfe0e2603295257.
Independent CPU4 full suite115passed in16.52s, including three actual publisher
CLIs. 1,800-second reservation of1,860 remaining; smoke600/full850.
Specd45c5d7ad3cffd5b39770ab80b0367abff1d8a1ee78c610dc27b42742ad19d66; canonical external inbox path named by this ID.

Synchronized CPU retry20261103T152350Z-49a47c66 published;330-second CPU ceiling, no GPU.
Root client waits for each endpoint.json before dialing; saves proof then
allows smoke/full progression. Native checker Python3.14.3+numpy2.5.4, MoQ
bindings0.5.2/0.4.20 on this Mac. Spec510c9dc3bd709e58e863ac0966527fcb56b59046dfdb43cd0fc39e5e5c993dc6.

Synchronized external check failed native QUIC connect with timeout despite
waiting for endpoint readiness. Do not infer the exact firewall/VPN cause.
Remote relay accepts its loopback publisher; no separate Mac session accepted.
Save /tmp/moqsplat-crosshost-synchronized-20261103/{result.json,smoke.log,
smoke/summary.json}. Endpoint self-expires; no firewall/route changes attempted.
Need a permitted UDP route/trusted reachable endpoint (possibly university VPN
or IT firewall assistance) before claiming outside-network success. No GPU used.
