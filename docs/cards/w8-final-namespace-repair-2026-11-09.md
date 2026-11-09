# W8 final namespace integration repair — November9

Status: local engineering and source-freeze review only. No combined CPU proof,
GPU qualification or full witness is released by this card.

## Goal and decision

The first CPU proof failed before its work because an imported legacy helper
lowered the inherited CPU hard limit. Its replacement passed that startup but
failed at the packet worker: host-mounted proc paths resolve a host PID whereas
`os.getpid()` returns the nested PID; the failed child's reaper then encountered
ESRCH when no descendants remained. Both failures and raw receipts remain.
These are infrastructure failures, not negative codec-quality measurements.

This is the second and final implementation-repair smoke proposal in this
finite programme. Narrowly bind only the existing maps/status self leaves to
the independently captured worker host PID, start ticks and campaign PGID.
Preserve the exact12system leaves and2clock helpers, loader identity, protected
inode exclusion and all source/denial/quality boundaries. Catch only ESRCH in
namespace-wide child termination; preserve a failed child's nonzero status and
publish the final complete-descendant-wait receipt. No arbitrary PID grant,
whole proc tree grant or UID relaxation is admitted.

The public metadata oracles passed on gpu5 MAIN with the actual host proc
mount and nested user/PID namespace. All14 paths were regular and distinct from
five protected inodes; no maps/protected/model bodies or CUDA were read.
A real immediate worker failure produced a failed final receipt and nonzero
launcher return. Root accepted mechanism scope only; direct policy check ran
at namespace PID1, while the real packet worker will run at PID2. It does not
replace the combined proof. See the root oracle review and original scripts.

## Prospective inclusive allocation

Closed proof-family CPU is238seconds:120first failure,108replacement including
8preparation,10both public-metadata probes. Zero GPU. Root proposes a new
**1,100 CPU-second cumulative proof-family cap**, inside the existing W8 parent
and unchanged24GPU-hour/28CPU-hour programme. It is not an extra reservation.
The862future CPU envelope is30source-preparation/publication,600single-core
campaign,150post-campaign/management/teardown and82independent audit/contingency.
This is a conservative ceiling, not measured full-proof cost. Administrative
CPU telemetry limits remain explicit. Any unused part is released inside the
parent at closure. Reconcile actual preparation before admission.

Before execution, root must review the complete new archive's Git blobs,
configured checks, exact physical archive/spec/CPU-gate SHA joins for bootstrap,
input helper and system-leaf policy, actual command/environment/input paths,
protected stat-only prechecks, eligible-host/claim state, deadline and complete
CPU forecast. No protected body may be opened before the complete producer is
sealed; historical SHA references are not fresh byte verification.

## Stops and promotion

One newly identified combined CPU-only proof may be released after review. It
must finish within600campaign seconds and the45-minute recovery bound, with a
new immutable output directory and one canonical fleet wait. Stop on any proof,
identity, denial, native, accounting, receipt, source, budget or deadline failure.
Inspect uncertain submissions; never replay. Retain all outputs and cancel only
task-owned processes. No third implementation-repair smoke is preauthorized.

A passing CPU proof qualifies only packet/native/policy/process mechanisms for
that exact source and host/runtime. No CUDA/kernel, generation, reconstruction
quality, timing or fresh witness result follows. Both-unit24-frame qualification
and the full original unbroken240-frame witness require separate admissions and
complete remaining CPU/GPU forecasts. Scientific recipe, comparison endpoints,
bit-exact masks and registered RGB/LPIPS/DISTS tolerances remain unchanged.

The eventual independently reviewed CPU gate must bind `system_leaves_sha256` to the exact policy member in the validated physical source snapshot and to the proof spec's `witness_system_leaves_sha256`. It remains an audit decision after real proof completion; the raw proof does not emit or approve that gate. The self-leaf PID, start tick and process-group values must join the corresponding namespace worker receipt for packet, both strict runtime leaves and assessment.
