# Fleet health recovery, 2026-10-27

Purpose: unblock the already approved real-background player study without
changing its scientific gates or its remaining3020 admitted GPU-seconds.
The user instructed the root to continue after the stopped execution preflight.
No previous research request is replayed.

## Repair and checks

Remote probes and small health/atomic-claim RPCs now have an explicit60-second
infrastructure deadline, matching
`inbox.local_probe`, rather than20seconds. Earlier complete diagnostics took
28–36seconds with valid environment/resource records. This is a prospective
management-timeout change, not acceptance of an incomplete probe or an increase
to the scientific GPU reservation. All GPU/process, CPU, shared-input,
executable, environment and atomic-claim checks remain required. Independent
PATH/version queries overlap the resource reads; SSH runs the selected Python
without a redundant nested login shell. Authenticated SSH connections may be
reused and expire after120idle seconds; these are machine connections, not
model wakeups or cache keepalives.

There are1316 historical inbox requests. Workers now remember requests whose
state they have verified terminal, avoiding repeated NFS reads on later scans.
Terminal requests are immutable by contract: recovery creates a new request,
and a declared resume is decided before terminal publication. Pending and
owned nonterminal requests remain inspected each scan. Heartbeats refresh as
the scan advances, not from a timer that could conceal blocked I/O.

Worker identity also used `/proc/<pid>` inode mtime as birth time. gpu2's
worker59678 was verified by UID, exact command, cwd and kernel start ticks:
it was the original worker, while its proc inode mtime had advanced. Birth
identity now uses kernel start ticks plus boot time. Legacy later inode times
conservatively protect a live process; a recycled PID starts after its saved
record and is still rejected. Worker replacement retains exact command and
host checks and never signals detached research supervisors.

A probe-specific shell-startup trial was tested, then the exact original
.bashrc was restored before rollout. Original SHA256
`2e93fca878766c030d986f10a92acd642b2417bed017a62d4eae89002226a9c5`.
The original backup and trial remain under `.pointstream-runtime/`; the final
repair requires no shell configuration change. Live application state and
protected storage remain untouched.

## Verification and rollout

Configured ruff/mypy and focused probe, worker-scan and process-birth tests pass.
The complete fleet regression suite passes with permission to inspect its own
children. Full local pytest has only the known macOS protected Linux symlink
failure; Linux CI remains the merge gate. A successful required compatible-host doctor,
frozen worker bundle and verified worker identities precede deployment. Scoped
refreshes preserve other configured hosts and leave their workers running. No
scientific GPU work is performed by these infrastructure checks.

Current receipts are retained locally under
`/private/tmp/pointstream-*-doctor-20261027.json` and test logs alongside them;
successful doctor reports also name their durable shared checks directory.
Rollout and research submission identities are appended below when confirmed.

### Confirmed compatible-worker rollout

PR #257 merged with Linux CI passing, source `c24dfd19` (merge `7c65b018`).
The gpu2/gpu3 doctor passed all checks at shared receipt
`jobs/fleet/checks/9bab43964d24470e86b63b03b42c72c2`.
The first refresh froze release `20261027T160223Z-3313c3a0` and replaced gpu2's
worker with PID4158772. gpu3's stop acknowledgement timed out; read-only
inspection confirmed old PID714429 was gone before any next action. A gated
`workers start --hosts gpu3` then froze the same source in release
`20261027T160853Z-49771504`, starting PID2735228. Its doctor receipt is
`jobs/fleet/checks/1075618d0cec47db9ba523f204359263`.
Canonical worker status confirmed both new heartbeats fresh. All six hosts
remain configured; only gpu2/gpu3 were refreshed, and all detached supervisors
and other workers were preserved. Storage safeguards are therefore rolled out
on these two hosts, not yet universally across the fleet.

### Stable management routing

Shared-inbox RPCs can select a verified configured `management_host` and an
absolute `management_python`; temporary overrides are
`PS_FLEET_MANAGEMENT_HOST` and `PS_FLEET_MANAGEMENT_PYTHON`. Explicit worker-host
operations retain their recorded worker interpreters. The saved client route
is gpu2 with `/usr/bin/python3`, after a successful read-only health call; the
manager dependency closure is standard-library-only. All six compute hosts and
worker interpreter mappings are retained. Scoped refreshes preserve these
management settings. No write is automatically retried or failed over.
Client-side control-master reuse was removed after delayed captured receipts;
normal SSH calls retain their bounded deadlines. The prior reuse description
above records the earlier repair revision, not the final client configuration.

### Opt-in actual staged-byte verification

Repeated canonical NFS byte reads delayed both publication and pre-claim
preparation despite verified disk caches. `verify_staged_inputs: true` permits
publication/admission to hash the content-addressed local copy for an input
that is explicitly passed as a staged placeholder. Canonical paths must still
exist inside the external data root; a directly passed input is never
substituted. Missing cache entries are copied and verified before admission.
Campaign prechecks defer only those payloads to native stage_input verification,
which rehashes every actual payload before invoking smoke or full. Corruption
and direct-input mutation remain hard failures. All other jobs keep the old
canonical-byte policy. This changes physical verification location, not the
bytes, scientific method or decision. The reviewed worker refresh is recorded
with its source and request below; no unknown submission is replayed.

The staged-verification worker was deployed on gpu2 from04ca05a7 at release
`jobs/fleet/releases/20261027T175327Z-aef44f0a`, PID154609, doctor receipt
`jobs/fleet/checks/d1f1140d403a418ca24d3b271e36316b`. gpu3 retains c24dfd19
from the prior scoped release; other workers are unchanged. This is a partial
rollout. Exact request outcomes and all study costs are in the experiment
ledger. The subsequent source snapshot explicitly includes tools after an
import-only failure; its unchanged judge and all direct/transitive imports
passed CPU preflight with CUDA hidden before another GPU request.
