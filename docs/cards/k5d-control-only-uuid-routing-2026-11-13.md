# Control-only physical GPU routing — November 13

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
