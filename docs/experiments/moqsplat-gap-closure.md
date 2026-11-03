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
