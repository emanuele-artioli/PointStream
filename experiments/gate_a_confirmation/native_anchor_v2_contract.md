# Native anchor v2 draft contract

This is a local implementation draft, not a registration or authorization to run.
No native encoder, remote CPU/GPU job, or environment modification was performed.
The v1 worker and all completed results are preserved.

## Supported evidence and next confirmation design

Saved native help at `/private/tmp/gate-a-local-20261002/native-tools-help.txt`
reports SVT-AV1 QP and CRF 1–63; CRF is rc0/aq-mode2, preset0 is supported,
and VVC FFmpeg wrapper QP -1–63 with boolean QPA. The legacy worker restricts
VVC to QP0–63, original3840x2160, preset slower, and AV1 to rates1–63,
preset0, widths3840/2560/1920/1280. These are supported settings, not measured
quality predictions. No speculative encoder knobs are introduced.

A prospective bounded design should retain separate VVC QPA0 and QPA1
families. QPA0 QP50 is already near the desired VMAF50–72 support (68.285094);
QPA1 QP50 is above it (77.436324). Candidate higher-QP samples can bracket
lower quality, with distinct prospective grids per family and a fixed rule
for subsequent dense sampling. Do not pool QPA families or use completed
posthoc selection as prospective evidence. The completed AV1 CRF59/1920
point (88.612482) motivates a supported lower-rate CRF60–63 and resolution
ladder including1280; this is a hypothesis, not a demonstrated bracket.
The current worker intentionally keeps existing supported widths. Nativehelp
permits smaller dimensions, but adding those requires a revised arm contract,
explicit even raster dimensions, command review, and representative smoke.

Before full96 execution, freeze a v2 hash, tools/libraries, exact arms, source
window indices, CPU affinity/budget and stopping rules, then run the same-path
bounded smoke and pilot. Keep every registered result and failure, complete
physical stream+manifest costs, original4K common reference, and fresh decode
from persisted streams. No matched native candidate is established yet.

## v2 artifacts and resource observations

The worker persists `frame-metrics.json` with original window identity and unambiguous within-window frame index,
VMAF, Y-MSE and SSIM per frame, linked by SHA256 in report.json. It preserves
the v1 summary definitions, manifest bytes, native stream bytes, and fresh
payload-stream decode. Metric files/resource logs are audit evidence, not
receiver payload and not included in physical payload bytes.

Every directly instrumented run command writes a resource receipt, including
failed/timed-out calls. POSIX wait4 provides exact child user/system CPU time
and kernel peak RSS; the unit is recorded (Linux KiB, macOS bytes). Parent
constraints record affinity, nice, and RLIMIT_AS; observed child constraints
are read immediately after launch where supported. Linux prlimit observes
the child's address-space limit. Short-lived children may exit before that
observation; the receipt says so. These limits are per process, not aggregate
job memory limits. Detached/reparented descendants are excluded. Nice and
limits are not a file-access sandbox or a no-interference guarantee.

Legacy RGB-to-YUV conversion helper and native ffprobe subprocesses are not
instrumented by this version. They are explicitly excluded in the report;
this worker must not be described as a complete whole-job resource audit.
Scientific execution explicitly requires Linux affinity, wait4, and child prlimit. RSS units for scientific receipts are therefore Linux KiB; Mac unit handling is only for local helper tests. Both original windows are opened and hashed solely to verify the pinned source identities; only selected windows enter encoding and scoring. Frame indices0–95 are within-window coordinates, not absolute scene frame numbers (source000 is original interval38:134). Tests on Mac verify
receipt mechanics only, not scientific/native compatibility. Receiver decode
argv consumes persisted stream, but decoding is not isolated with OS file
permissions; sender arrays remain in the worker process, as disclosed by v1.

## Local checks

`python -m pytest -q tests/experiments/test_native_anchor_v2.py`: 5 passed.
Checks cover full96 source000 within-window identity and exact metric coverage, child RSS/constraints receipts,
nonzero exit receipt retention, and timed-out owned-child reaping.
`python -m py_compile experiments/gate_a_confirmation/native_anchor_v2.py`
passes. Native same-path smoke remains required before scientific execution.

Timeout cancellation kills and reaps only the exact direct child. It does not promise descendant process-group cleanup; native commands must not daemonize. No detached descendant lifetime/resource claim is made.
