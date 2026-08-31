# Prompt — why editors cannot open PointStream on this host

Paste below the line. This is a **diagnosis-first** task: the measurements below
were taken 2026-08-31 and point at a likely dominant cause, but the causal link
to the editor symptom is *not* established. Confirm it before fixing it.

---

You are debugging why **Cursor and VS Code cannot work with PointStream on this
GPU server**, while VS Code reportedly works fine on TIGAS on the *same* server.
That asymmetry is the most useful clue available and the investigation should
start from it: whatever is wrong is a property of this project or its layout,
not of the host or the editor alone.

**Read first:** `/home/itec/emanuele/.agent-rules/AGENTS.md` (the NFS section)
and `src/contracts/paths.py` in this repo.

## What is already measured — do not re-derive

**The home directory is NFS at roughly 6 file opens per second.** Bulk
throughput is fine (174 MB/s read); metadata latency is not, and page cache does
not help.

**Inode counts, 2026-08-31:**

| location | inodes | of which `.mypy_cache` |
|---|---:|---:|
| `pointstream` (main checkout) | 17,028 | 9,444 |
| `pointstream-w6-b` | 9,068 | 8,301 |
| `pointstream-w8-a` | 8,894 | 8,311 |
| `pointstream-w8-b` | 8,858 | 8,304 |
| `pointstream-w8-c` | 8,932 | 8,303 |
| `pointstream-w8-d` | 8,929 | 8,307 |
| `pointstream-w8-e` | 9,168 | 8,364 |
| `pointstream-w8-fix` | 8,861 | 8,303 |
| **total across worktrees** | **79,738** | **~67,600 (85%)** |

Actual tracked source is about **750 inodes per checkout**. So **85% of
everything an editor would walk across these worktrees is duplicated mypy
cache**, and at 6 opens/second a full walk of 79,738 inodes is roughly **3.7
hours**.

`assets/` and `outputs/` (~565,000 files) were moved out of the checkout on
2026-08-29 to `/home/itec/emanuele/pointstream-data`, resolved via a gitignored
`.ps-data-root` marker. **Do not reintroduce symlinks to them** — that is how
one dataset became twelve worktrees' worth of walking.

**`/tmp` is local ext4** (`/dev/mapper/ubuntu--vg-ubuntu--lv`); the home
directory is `nfs4`. This is the single most useful fact here.

**A co-tenant's editor was hammering the same mount.** On 2026-08-31 a `find`
plus recursive `grep -RIn` belonging to user `ayman`, spawned by their
VS Code server, had been running **11 h 52 m**. Whatever else is true, this host
has contention that is not PointStream's fault, and a measurement taken while it
runs is not a measurement of PointStream alone.

## Establish the cause before fixing it

The inode count is a *hypothesis*, not a diagnosis. Two things must be checked:

1. **Does the editor actually walk all of it?** Find out what Cursor/VS Code is
   opening as its folder. If it is the *parent* (`/home/itec/emanuele`), it
   walks every worktree **and** `pointstream-data`'s ~565,000 files, which is a
   different and much larger problem than a single worktree. If it is one
   worktree, 9,000 inodes should be survivable and the cause is elsewhere.
2. **Why does TIGAS work?** Compare directly: its inode count, whether it has
   multiple worktrees, whether it has a `.mypy_cache`, what its
   `files.watcherExclude` looks like. If TIGAS is also ~9,000 inodes and works,
   inode count is *not* the cause and this brief's hypothesis is wrong — say so.

Also worth separating, because they may be two problems wearing one coat: a
**connection/remote-server** failure (the VS Code Server dying, timing out, or
never installing) is not the same as an **indexing** failure (connects, then
spins forever). Establish which symptom is actually occurring before treating
either.

## Fixes, in the order their evidence supports

1. **Move the mypy cache off NFS and out of the tree.** `MYPY_CACHE_DIR` (or
   `cache_dir` in `pyproject.toml`) pointing under `/tmp` — local ext4, and
   outside anything an editor walks. This is worth ~8,300 inodes per worktree
   and should also make mypy itself dramatically faster: a full run currently
   takes 15-25 minutes locally against **3m30s on CI**. Watch for one trap:
   several worktrees must not share one cache directory, or they will fight over
   entries keyed by the same module names.
2. **Remove merged worktrees.** As of 2026-08-31 all six wave-8 branches
   (`plate-codec-sweep`, `intra-sidecar`, `low-rate`, `panorama`,
   `weights-path`, `coordination`) are **0 commits ahead of main**, so their
   worktrees are pure walking cost. Follow `AGENTS.md`: read a branch before
   deleting it, tag anything unmerged, and never `--force` away a worktree with
   uncommitted changes. **Check with the user first** — a session may be paused
   in one.
3. **`.ruff_cache` is not gitignored.** Small, but it is in the tree.
4. **Editor excludes**, if the folder-scope check in step 1 says they are
   needed: `files.watcherExclude`, `search.exclude`,
   `git.autoRepositoryDetection: false`, and open **one worktree** as the
   folder, never the parent.

## Done when

The symptom is reproduced and attributed to a named cause with a measurement
behind it; the fix is applied; and an editor opens the project and reaches a
usable state in a stated time. If the inode hypothesis turns out not to explain
it, the report says what does — a wrong hypothesis closed out is a result.
