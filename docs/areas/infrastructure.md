# Infrastructure Area

## Local coordinator / remote GPU pilot — 26 September 2026

Codex now coordinates from the Mac; jobs execute from immutable, uniquely named
snapshots on whichever compatible GPU host is clear at launch. The bounded pilot
passed on gpu5 and gpu6. `experiments/jobs/fleet.py` inspects the fleet, launches
snapshots, and retrieves status or cancellation state; `experiments/jobs/monitor.py`
retains supervision, GPU claims, device isolation, and conservative child cleanup.
There is no persistent queue, automatic migration, or Codex event-delivery dependency.

Admission fails closed on incomplete probes and requires no GPU processes, device
memory within the configured idle baseline, enough free memory for the declared peak
plus reserve, CPU headroom, required software, and available inputs. Every candidate
host is probed before selection; the selected GPU is claimed by canonical host and
UUID and rechecked just before launch. Independent jobs can run concurrently on
separate claimed devices under the aggregate host CPU limit. The default is one GPU
per job. Claims coordinate PointStream jobs but cannot reserve a device against other
users or guarantee an arbitrary job will avoid OOM. Contention stops only the affected
PointStream child, preserves outputs, and marks timing contaminated; jobs are not
replayed or migrated automatically.

No comparable cross-model workload timings are recorded yet. The default selector
uses the fallback order RTX 6000 Ada, A6000, RTX 8000, then GV100; future comparable
timings can be supplied with ordered `--prefer-gpu-name` values and are recorded in
the launch manifest. `gpu4` was observed with about 46 GiB in use per GPU despite 0% utilization;
Slurm reported it idle but uses `task/none`, so scheduler state is not treated as
exclusive access.

### Pilot evidence

- Concurrent acquisition on the shared NFS claim path serialized gpu5/gpu6 correctly;
  ownership, release and interruption paths were exercised.
- Two simultaneous synthetic CUDA probes ran on separate RTX 6000 Ada hosts with
  `CUDA_VISIBLE_DEVICES` set to each claimed UUID. A deliberate foreign test process
  on gpu5 caused only that probe to stop and become `contended`; gpu6 completed.
- A follow-up run verified exact filtered snapshot SHA-256 identity, separate output
  directories, status retrieval after launch, and cancellation of only the requested
  child. Both hosts returned to baseline after the probes.
- Detached job `20260926T221103Z-a19f2af8` verified SSH-loss recovery: the local
  launcher returned, a fresh status request found the 64 MiB CUDA allocation still
  running, and a later reconnect reported exit 0. The retrieved log confirmed the
  CUDA allocation; gpu5 returned to 1 MiB baseline use with no compute process or
  PointStream claim. Required executable checks now accept both PATH names and
  absolute paths and fail closed on incomplete responses.
- Bounded smoke job `20260926T212943Z-970d4e98` completed on gpu6 using cached
  `alcaraz_highlights/scene_000` inputs. Pasteback MAE was 0.0, all-off was bit-identical,
  residual-absent took 47.19 s, and the fast tier took 28.71 s. This validates the
  runner against real inputs; that tier path did not use CUDA and is not paper evidence.
- Focused tests: 54 pass across resource claims, fleet dispatch and job monitoring.
  Ruff 0.11.2 and Python compilation passed in the pinned remote environment.

### Installed agent configuration cleanup

The six host-local Codex homes no longer contain custom instruction files, hook
links, generated model roles or rule ladders. Shared Claude/Cursor/Gemini imports,
hooks and generated role/skill links were disconnected after backup to
`~/.pointstream-agent-backup/2026-09-26`. Official plugins, built-in skills,
credentials, preferences, session databases and host-local application state
remain. The NFS/editor/cache bootstrap now lives at
`~/.pointstream-runtime/bootstrap-hostlocal.sh`; fresh login-shell checks passed
on all six hosts, and `codex --version` passed in a new gpu5 login shell.

No running app process was stopped. The original `.agent-rules` checkout remains
at its original path without source edits so already-running hooks can finish;
global instructions and hook configurations no longer point to it for new
sessions. The GitHub configuration repository was not committed to or pushed.

**Remaining limit:** launch-time checks and cooperative claims cannot prevent a
non-participating user from allocating a GPU later. The monitor detects changed
occupancy, terminates only its own child, and preserves a contaminated run record.
The two-host claim test and bounded smoke are complete; this does not certify every
workload's memory requirement or make scheduler reports authoritative.

## Coordinator follow-up — E01/E02

Atomic resource claims are implemented in `experiments/jobs/claims.py` and reused by
the dispatcher and monitor: UUID and canonical hostname keying, pre-launch recheck,
child device isolation, conservative stale-owner verification, aggregate per-host CPU
limits, and process-group supervision. The shared NFS concurrent-acquisition and
interruption tests passed on gpu5/gpu6 as recorded above.

## Current execution policy

Inspect all reachable candidates for each request. Unreachable or malformed host
probes are unavailable. Use existing pinned environments and external datasets; never
modify or overwrite a working remote checkout. Snapshot the selected local revision
and only explicitly selected changes to a unique remote run directory, recording code
hash, environment, command, GPU UUID and native codec versions. Keep detached logs and
supervision remote so SSH loss or laptop sleep does not stop a job. Retrieve status and
results locally. Preserve scientific protocols and evidence classifications; the
infrastructure smoke above is not a result for the manuscript.

The unsafe `scripts/cleanup_merged_worktrees.sh` remains prohibited (`INFRA-ACT-01`);
never bypass Git's refusal or remove a potentially paused worktree. Historical
campaign plans below describe their original setup and are not the current dispatch
interface.

**Evidence Revision**: Local dispatcher pilot, 26 September 2026; resource claims from R0/R0R.
**Owned Scope**: Environments, CI/GitHub Actions, worktree lifecycle, runner integration, local caches, hardware profiling.

---

## 1. Current State

### Atomic cross-host resource claims (R0 / R0R)

`experiments/jobs/claims.py` implements filesystem resource claims for PointStream
workers without cluster schedulers or colleague preemption. The two-host NFS
contention and interruption check is now recorded above. Claims are cooperative and
never preempt another user's work:
- **Shared jobs location**: Keyed under `PS_DATA_ROOT/jobs/claims` or explicit `PS_CLAIMS_DIR`.
- **Atomic cross-host primitive**: POSIX atomic directory creation (`mkdir`) on the shared filesystem for device claims and host CPU allocation locks.
- **Device keying**: Canonical hostname and GPU UUID, not ordinal alone.
- **Pre-selection and pre-launch recheck**: Inspects memory and processes before selection and again under claim immediately before child launch.
- **Device isolation**: Hides unallocated GPUs from the child using `CUDA_VISIBLE_DEVICES` and `PS_CLAIMED_GPU_UUID`.
- **Conservative stale-owner handling**: Releases only by owning token; never steals a remote or unverifiable stale claim.
- **Child lifecycle**: Terminates and reaps the process group before releasing claims; retains claims if child death cannot be verified.
- **CPU allowance**: Refuses declarations above 90% of available cores and applies cooperative thread environment limits.
- **Validation**: The two-host acquisition, ownership, release/interruption, isolation, cancellation and contention paths passed in the pilot above. Focused tests pass; see `tests/experiments/`.

### Quiet long-job monitoring (historical transport)

`experiments/jobs/monitor.py` can supervise detached jobs, retain durable logs and
publish event records. The previous Codex CLI wakeup adapter is historical and is not
used by local dispatch: local tasks retrieve remote status explicitly. See the current
[long-job workflow](../workflow/long-jobs.md) for launch and retrieval.

### Environment & Startup Performance
PointStream runs on a shared remote Linux GPU server with an NFS-backed home directory. On **gpu6** (commit `bc09184`, September 2026), process startup and import latency were measured under clean conditions (`PYTHONNOUSERSITE=1`, explicit `PYTHONPATH`):

| Operation | n | Mean ± SE (s) | Range (s) | Notes |
|---|---:|---:|---:|---|
| `git status --short` | 5 | 0.057 ± 0.034 | 0.020–0.195 | Fast metadata walk |
| `git ls-files` | 5 | 0.0051 ± 0.0001 | 0.0046–0.0053 | Memory cached |
| `rg --files` (source dirs) | 5 | 0.0147 ± 0.0006 | 0.0134–0.0168 | Fast local traversal |
| System Python no-op | 5 | 0.0227 ± 0.0006 | 0.0215–0.0243 | Base interpreter overhead |
| PointStream Python no-op | 5 | 0.0455 ± 0.0113 | 0.0330–0.0905 | Conda env startup |
| `import sqlite3` | 3 | 0.0440 ± 0.0074 | 0.0353–0.0587 | Enforces ABI compliance |
| `import sqlite3` then `torch` | 3 | 2.101 ± 0.620 | 1.350–3.331 | PyTorch import tax |
| `import sqlite3` then `src.runner` | 3 | 0.529 ± 0.117 | 0.406–0.764 | Runner does **not** load Torch |

These timings are warm-cache metadata walks and imports. File creation and data
movement on the same home are far slower.

### NFS home file I/O (2026-10-05)

Every host mounts the home as `data3:/x/home/itec/emanuele` (NFSv4.2, `soft`, 10 Gb
link, 0.2–0.3 ms ping). Single runs of `n = 200` small-file operations per host
(fsync `n = 20`), on gpu3, gpu5 and gpu6:

| Operation | NFS home | Host-local disk |
|---|---:|---:|
| Create and write 4 KB | 174–221 ms | 0.03–0.07 ms |
| Open and read (warm) | 11–17 ms | 0.01–0.02 ms |
| Write and fsync | 138–231 ms | 0.3–1.8 ms |
| Unlink | 73–95 ms | ≈0 ms |
| Sequential write | 26–49 MB/s | 258–1,644 MB/s |
| Sequential read, direct I/O (one file each) | gpu6 93, gpu5 42, gpu3 16 MB/s | — |

Network round trip is not the limit; the delay is server-side.
`/proc/self/mountstats` showed per-operation server time of 5–25 ms for
`GETATTR`/`OPEN` and queueing of 240–410 ms on `WRITE`. Ten-second samples
showed 20–570 ms per operation, and one 18 s `DELEGRETURN`, while our clients
issued only tens to a few hundred operations per second. data3 is a shared
institute server (12–23 mounts per host). gpu3 and gpu6 had issued 1.95 and
1.76 billion `TEST_STATEID` calls since boot, which suggests repeated revocation of
NFSv4 state. That count is a lead for the server's administrators, not a diagnosis.

Workaround: [host-local staging](../workflow/long-jobs.md#host-local-staging). A 6 GB
archive took 313 s to read cold on gpu6, and 4.2 s to verify from the local cache on
reuse. `/local/users/emanuele` (the local root) exists only on gpu6; other hosts
fall back to shared paths.

### Worktree Cleanup Audit
PR #68 introduced `scripts/cleanup_merged_worktrees.sh`. The documentation audit in PR #73 flagged critical safety hazards in this helper:
- Suppresses error codes from `git fetch`, `status`, and `log`.
- Falls back from `git worktree remove` to `rm -rf` when git refuses to delete a dirty worktree.
- Performs irreversible remote-ref pruning (`git remote prune origin`).

**Operational Directive**: Do **not** execute `scripts/cleanup_merged_worktrees.sh` in automated workflows until `INFRA-ACT-01` is completed.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| External data separation | #32 (`d436b02`), #35 (`420c3bec4a`) | `paths.py` resolver routes data to `.ps-data-root`; zero symlinks in repo. |
| Worktree recovery | #53 (`ec581e957d`) | Strict branch-and-worktree isolation protocol established. |
| CI automation | #23, #44 | Integrated `ruff`, `mypy`, layer boundaries, and synthetic tier tests. |
| Host profiling | #73 | Measured process startup and import times; confirmed runner does not load Torch. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `INFRA-ACT-05` | Complete for cooperative PointStream jobs | Pilot evidence above | R0/R0R and 2026-09-26 pilot | **Atomic resource claims and local fleet dispatch**: two-host contention, lifecycle, CUDA isolation, remote supervision, status retrieval and bounded real-input smoke passed. Other-user allocations remain outside cooperative claim control. |
| `INFRA-ACT-04` | Complete | None | PR #82 | Quiet monitor and approved scheduling/stall/restart/budget regression tests implemented. Use the workflow for new jobs; transport acceptance is verified, but automated idle wakeup timing is not a guaranteed service. |
| `INFRA-ACT-01` | Ready | None | #68, #73 | **Repair worktree cleanup helper**: Refactor `scripts/cleanup_merged_worktrees.sh` to halt on any git refusal, verify clean working tree against `origin/main`, remove the `rm -rf` fallback, and drop remote pruning. Acceptance: Script refuses to delete unmerged or dirty worktrees and passes unit test. |
| `INFRA-ACT-02` | Ready | None | Host rules | **Host-local cache enforcement**: Configure local caching (the checkout-specific cache paths in `docs/setup.md`) in CI and runner scripts. Acceptance: Zero mypy cache files written to NFS home. |
| `INFRA-ACT-03` | Closed / Archived (D1/D6) | None | `plans/DEFERRED.md` | **Static typing and test pollution**: Mypy passes cleanly across all 350 source files; tests isolated from global environment. |

### PR #73 review follow-up (2026-09-08)

Moved closeout procedures into the session skill and verification rationale into setup; corrected pytest cache configuration and the README output path that bypassed the data root. Replaced nonexistent area evidence commit hashes with verified PR merge references. Repaired the audit brief recovery row, which stored a commit abbreviation instead of a blob hash. Reconciled stale assignment state and qualified unverified codec claims. Gate B now distinguishes per-video encoding from shared-model training and source generalization; the existing six-match validator remains authoritative. Validation: relative-link and recovery-blob audit, documented CLI inspection, skill frontmatter validation, and diff whitespace checks; CI results are recorded in the follow-up PR. Next infrastructure action remains `INFRA-ACT-01`; this documentation review does not repair or authorize the unsafe cleanup helper.

### Worktree retirement — 2026-09-11

User approved removal of six clean worktrees; removed without force or the unsafe
helper. The following remote tags preserve their tips under `archive/20260911/`:

| Removed worktree under `/tmp/` | Archive tag suffix | Preservation evidence |
|---|---|---|
| `pointstream-pr88-audit` | `pr-88` | `d1d24b7` ancestor of PR #88 integrated head |
| `pointstream-probe-framework` | `codex/probe-framework` | Tree identical to merged `dc4a0cd` (#94) |
| `pointstream-recovery` | `antigravity/overnight-recovery` | Tree identical to merged `2b7c2b0` (#88) |
| `pointstream-submission-dispatch` | `pr-90` | Tree identical to merged `221aa14` (#90) |
| `pointstream-worker-b` | `antigravity/recovery-worker-b-verdict` | `ea4fee8` ancestor of preserved #88 head |
| `pointstream-worker-c` | `antigravity/recovery-worker-c-generator` | `1992878` ancestor of preserved #88 head |

Tracked and untracked state was clean; ignored output directories contained no
files. Cache directories were disposable. All local and remote branches remain.
The larger local-branch deletion proposal was rejected by automatic approval
review as beyond the six-worktree approval; it was not executed.

Retain `/tmp/pointstream-worker-a`: `artifacts/residual_smoke.json` is modified.
Retain `/tmp/pointstream-wave1-a`, `-b`, `-c` until #93 is repaired/integrated;
their commits are ancestors of #93, but that PR is still open. Retain
`/tmp/pointstream-pr92` as a possibly paused Cursor checkout; after #92 updates,
fetch and reconcile it before resuming. No process observed in this sandbox
proves another host/session idle. `INFRA-ACT-01` remains open.

### Handoff cleanup — 2026-09-12

With explicit user approval, retired all nine remaining stale worktrees and 32
local non-main branches after remotely archiving every tip. Preserved the dirty
worker-a residual-smoke artifact in archive-only commit `26a7555`; it is not new
mainline evidence. Exact refs, preservation proofs and removed paths are in
[cleanup inventory](../history/cleanup-2026-09-12.json). No remote branch, dataset,
experiment artifact or paper file was deleted. Only main and the current handoff
checkout remained immediately after this cleanup. Do not use the unsafe cleanup
helper; `INFRA-ACT-01` is still open.
