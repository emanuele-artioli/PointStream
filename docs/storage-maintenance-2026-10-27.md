# Host-local storage recovery, 2026-10-27

The administrator reported user data filling gpu3's 100 GB root disk. The
previous shared shell bootstrap deliberately used `/var/tmp` for editor
servers, Codex state and Python bytecode to avoid slow NFS small-file access.
That performance requirement remains valid. `/var/tmp` was the wrong disk:
`/local/users/emanuele` now exists on every GPU server, is owned by emanuele,
and is on a separate local filesystem. Neither dataset/model placement nor
published experiment evidence was changed.

## Work performed

Inventoried only emanuele-owned top-level entries and their contents on all
six hosts. Process executable, cwd, open descriptors, mapped files and path
arguments were checked before copying inactive trees. Inactive files were
copied using `rsync -aH`, verified with a checksum dry run, then switched to
compatibility symlinks. No editor, Codex or research process was signalled.
The original inactive duplicate was removed only after verification; source
permissions and content were preserved in the destination. A read-only old
source cleanup on gpu1 left a small additional recovery copy, retained rather
than discarding unverified state.

Copies and per-host journals are under
`/local/users/emanuele/relocated-var-tmp-20261027/`, with `migration.jsonl`
recording paths and outcomes. Do not delete this tree: some entries are live
application state, not disposable cache.

For gpu1–gpu4, the remaining public directory paths were switched atomically
while preserving every remaining live inode in a hidden
`/var/tmp/.emanuele-live-20261027-*` tree. Destination symlinks continue to
reference those exact live files. New directories use the local disk.
Sockets, locks, SQLite files and mapped executables were retained, not copied
or restarted. gpu3's control socket and existing processes were checked after
the switch.

Automatic review rejected that final live path switch on gpu5 and gpu6 because
of possible disruption to active editors, locks, sockets and databases. It
was not retried or bypassed. Their inactive content was already moved and
verified, but their remaining live public trees stay in place. Runtime aliases
on these two hosts deliberately resolve those trees to their existing paths;
this is **not** a claim that all application state is off the root disk.

## Verified state

A fresh SSH shell on every host verified that Python's actual temporary
location and Python cache resolve to `/local/users/emanuele`, whose device is
different from `/`. Measurements are in the accompanying JSON.

| Host | Root free, GiB | Remaining owned `/var/tmp`, MiB |
|---|---:|---:|
| gpu1 | 22.08 | 251.4 |
| gpu2 | 21.48 | 248.5 |
| gpu3 | 20.19 | 600.0 |
| gpu4 | 35.63 | 248.9 |
| gpu5 | 24.88 | 532.5 |
| gpu6 | 31.10 | 532.6 |

Before maintenance gpu3 had about 6.9 GiB free (93% used); afterward it was
79% used. Remaining usage includes live files retained intentionally. Do not
remove them while their processes still use them. After the relevant sessions
are closed, a separately checked relocation can finish those files; no timer
or unattended cleanup was installed.

## Preventing recurrence

The maintained script is `env/bootstrap-hostlocal.sh`, installed in the shared
home as `~/.pointstream-runtime/bootstrap-hostlocal.sh`. Existing `.bashrc`
and `.profile` already source it. The original is preserved as
`bootstrap-hostlocal.sh.before-local-20261027`. New stable aliases live under
`/local/users/emanuele/runtime/`. Cache and installer aliases preserve existing
content; they do not re-download models or move published results.

The bootstrap sets `PS_LOCAL_ROOT`, `TMPDIR`, `TMP`, `TEMP`,
`PYTHONPYCACHEPREFIX` and `AGENTCTL_RUNTIME` to host-local storage. Existing
application state is preserved through the aliases described above.
The script refuses setup when the administrator-provided user directory is
missing or on the root filesystem. For another host, provision/migrate its
runtime aliases before using this shared bootstrap.

Fleet staging now refuses Linux root-disk roots, including symlink aliases.
Workloads and validators get separate per-stage temporary directories outside
published scratch, with a reusable local bytecode cache. Successful stage
sensitive temporaries are removed; failure state stays recoverable, with paths
in `staging.json`. The environment builder also refuses root-disk builds and
places its implicit temporaries and caches inside the chosen build root.
The fleet's existing 50 GiB local-disk reserve remains in force.

Verification: `ruff check`, configured mypy, 28 focused storage tests, and both
shell syntax checks pass. The full local suite produced 895 passed and 25
skipped, with the two previously known macOS/sandbox failures: protected
`/home` symlink resolution and child-process inspection. The process test
passes with permitted child inspection. Linux CI remains the complete gate.
A Linux refusal test checks the builder before it can create a root-disk
build directory. Worker deployment is recorded below when performed.

### Worker rollout result

All 258 fleet tests pass with permitted process inspection. The Linux build
refusal check returned exit 2 before creating its forbidden build directory.
The shared bootstrap is installed and verified on all six hosts.

A fleet-manager refresh from code revision `ff81c561` was attempted but stopped
at `doctor`, before snapshot deployment or stopping any worker: gpu2 and gpu3
exceeded the existing 20-second SSH/environment probe limit. A bounded read-only
diagnostic subsequently verified gpu2 in 38.84 seconds; gpu3's same probe did
not finish within 60 seconds. This does not imply SSH or storage is unavailable:
the separate fresh-shell/storage checks succeeded on both hosts. No scientific
job was submitted or replayed, no worker was changed, and the health gate was
not bypassed. The new manager-level safeguards need a successful worker refresh
once this management-path latency is resolved. Existing long-running processes
keep their original environments; the new bootstrap applies to fresh shells.

Remaining actions are therefore: finish retained live originals when their
sessions are idle (gpu5/gpu6 public live-tree switches require resolving the
explicit approval rejection), then complete the gated worker rollout. Do not
silently treat either action as complete.


### Root-chat closure audit

The closing audit rechecked all six hosts. Every substantial remaining owned
real `/var/tmp` tree has active executable, cwd, mapped-file or descriptor
references. These files are intentionally preserved; do not stop an editor or
remove a live database to claim zero root usage. The one inactive 1.3 MiB gpu1
recovery duplicate was compared using a checksum dry run against its preserved
`/local/users/emanuele/relocated-var-tmp-20261027/` destination, checked again
for active references, and removed. Its completion is in `migration.jsonl`.
The prior paragraph describing that duplicate as retained is historical.
No further broad live-directory switching was attempted on gpu5/gpu6.

PR #253 passed Linux CI and merged. Shared shell defaults are active; fleet
manager deployment remains pending exactly as documented above. A new research
root session should start with `docs/continuation-2026-10-27.md` and PLAN.md.

### Later compatible-worker refresh

The resumed root resolved the management health issue in PR #257 without
changing scientific resource gates. See [fleet recovery](fleet-health-recovery-2026-10-27.md)
for bounded60s infrastructure deadlines, stable process identity, advancing
worker heartbeats, full doctor receipts and the inspected lost stop
acknowledgement. gpu2/gpu3 now run the frozen repaired worker; the other four
workers and all detached supervisors remain preserved. This is a partial
rollout of the manager storage safeguards. No protected live-directory cleanup
or gpu5/gpu6 live-path switching was attempted. The shell-startup trial was
restored to the exact original configuration before this rollout.
