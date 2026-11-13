# Control-only physical GPU routing — November 13

Status: source reviewed,206focused tests/Ruff/mypy/fullCI PASS; PR389 merged
at56c5f2ec219c14831cdbc5e54a3353a6a202b53b. Actual activation backend46477
started after native7a3274d8 independent cleanup acceptance; new worker
PID1470412/birth562919381, first-tick qualification pending.60CPU is charged,
0GPU. Existing activation must be inspected, never replayed.
Output: `/private/tmp/pointstream-uuid-control-activation-readygate-UNRUN-20261113/execution-uuid-control-56c5f2ec-20261113-a1279b5c8048`.
Scientific2c6d is immutable; no new GPU result follows.

Root releases 220 local CPU-seconds for implementation and checks; separately holds
60 CPU-seconds for a reviewed Linux activation qualification. Both come from
unallocated Ball/programme headroom under unchanged 24 GPU-hour / 28 CPU-hour
programme and Ball 64,800 GPU / 72,000 CPU ceilings. No GPU/model/deployment is
released by the local implementation stage.

The paired decoder calibration must use the same physical GPU across its
preparation and scoring jobs. Model-name filtering alone cannot guarantee this.
Add an optional single GPU UUID, alongside existing model/memory/idle filters.
Keep all existing defaults. UUID requests require a unique bare executable
sentinel resolved only through an explicit eligibility-only search in the new
immutable release directory containing only that sentinel. Verify exact owned,
non-symlink executable path and bytes/hash. Other commands use ordinary PATH.
Never add the directory to global, worker or campaign PATH. Existing legacy
workers check required_commands before GPU selection/claims and must reject
these requests safely. Client capability verification is required before
publication. Before publication and rollback, prove the sentinel is absent from ordinary
PATH; worker_start and all downstream environments remain byte-identical. No cross-job continuous
GPU lease is required; jobs may queue between stages.

Scientific source 2c6d/source6f/archive5ca/env32e, configuration e478, all31
source grants, native/runtime paths and accepted CPU workload proofs remain
unchanged. Preserve old proof control provenance. Control edits may touch
only validation, eligibility and capability/client checks;
monitor, claims, staging, downstream launch/campaign and receiver environment
must remain unchanged. Actual source-delta review decides noninterference.

Meaningful local tests cover exact UUID selection, busy/missing target with
no fallback, malformed fields, legacy rejection before claims, capability
failure before publication and unchanged defaults. Run pinned CI tool checks;
full CI may run remotely. Stop at the allocation, unreviewed deltas, safety
ambiguity or any scientific/config change. No automatic deployment or retry.

Only after candidate review may a separately released bounded real Linux
control/source-plane check activate the task-owned quiescent worker, retaining
old release and evidence and recording PID/birth/release/ordinary-PATH/capability.
A first explicitly conditional useful GPU witness must independently join
requested UUID, claim, dispatch and device environment. CPU tests cannot
establish actual GPU routing. Refresh prospective spec/control/observer pins
before any publication; preserve expired UNRUN packages without replay.


External-controller follow-up: root separately releases20localCPU from the existing
queue administration other32 reserve for a NEW V5 copy. Bind expected UUID to
the exact hashed scientific spec and reject a mismatched admitted request before
observer assignment; retain V4. Tests must exercise correct UUID and mismatched/
missing spec or selector. No runtime or observer release. Queueadmin300 now
includes140past(120Linuxpilots+20V5local),148futurechain and12other. Separate
control activation60CPU remains held; it is not hidden in this administration.

Root reviewed all candidate production changes and218line meaningful test file,
including hash-pinned actual legacy eligibility rejection before claims.
New worker activation must preserve the captured old worker interpreter,
CPU limits and ordinary PATH. Never put a low hardCPU bootstrap limit around
worker_start: the dispatcher and later supervisors inherit it. Qualify startup
with attended bounded parent supervision and measured administrative cost;
steady-state machine infrastructure is separate from scientific campaign time.
A fresh capability heartbeat does not prove the first1700-record historical
scan is finished; verify readiness before relying on the60s admission window.
At that historical review, no launch/stop/staging helper had run; current
actual activation is recorded at the top and in the Ball journal.
