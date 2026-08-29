# Where the data lives

`assets/` and `outputs/` are **not** part of the code tree, even when they
happen to sit inside it. Together they are roughly 565,000 files against the
~700 this repository tracks, and this host's home directory is an NFS mount
serving on the order of ten milliseconds per file open. A tool asked to index
the project walks half a million files at that rate and never finishes — which
is why VS Code's Source Control view used to sit at "scanning folder for git
repositories" indefinitely, and why anything waiting on it did too.

## Pointing somewhere else

Three mechanisms, in order of precedence:

1. **`PS_DATA_ROOT`** in the environment, for a one-off override.
2. **`.ps-data-root`** — a one-line file in the repository root naming the
   directory. Gitignored, because where the data sits is a property of the
   machine, not of the branch.
3. **The repository root**, which is the historical layout.

The marker file is the one to use. An environment variable has to be exported in
every shell, every editor terminal, every cron entry and every agent session,
and when it is missing the failure is a confusing "file not found" rather than a
clear one. The marker travels with the checkout, so a process that inherits
nothing still finds the data.

`src/contracts/paths.py` is the only place that resolves this; nothing else
should join `"assets"` or `"outputs"` onto a repo root.

**Do not use a symlink.** A symlink inside the project is what tools follow, and
it is how one dataset became twelve: every git worktree carried `assets` and
`outputs` symlinks back to the same directories, so repository auto-detection
found the same half-million files once per worktree. Point the variable at the
data and leave nothing in the tree to follow.

## Moving the data

Done on this host, 2026-08-29, to `/home/itec/emanuele/pointstream-data`:

```bash
mkdir -p /home/itec/emanuele/pointstream-data
mv assets outputs /home/itec/emanuele/pointstream-data/
echo /home/itec/emanuele/pointstream-data > .ps-data-root
```

**Keep the destination on the same filesystem.** Then `mv` is a rename and is
effectively instant; across filesystems it copies, and at ~6 file opens per
second under load, 565,000 files is not a quick copy.

Measured effect on this checkout: **579,918 files before, 14,958 after** — a
factor of 39, and most of what remains is `.git` itself.

The tracked empty `assets/weights/.gitkeep` was removed with the move. It
existed to keep an empty directory in the tree, and an empty `assets/` in a
checkout whose data lives elsewhere is a trap rather than a convenience.

## What this does not change

Result paths quoted in the paper's `CLAIM` lines, and in `plans/`, stay written
as `outputs/...`. They are relative to the **data root**, not to the checkout.
A run that records its own paths should also record what they resolved to —
`paths.describe()` returns exactly that.
