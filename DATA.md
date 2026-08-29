# Where the data lives

`assets/` and `outputs/` are **not** part of the code tree, even when they
happen to sit inside it. Together they are roughly 565,000 files against the
~700 this repository tracks, and this host's home directory is an NFS mount
serving on the order of ten milliseconds per file open. A tool asked to index
the project walks half a million files at that rate and never finishes — which
is why VS Code's Source Control view used to sit at "scanning folder for git
repositories" indefinitely, and why anything waiting on it did too.

## Pointing somewhere else

Set `PS_DATA_ROOT` to the directory that contains `assets/` and `outputs/`:

```bash
export PS_DATA_ROOT=/home/itec/emanuele/pointstream-data
```

Unset, it defaults to the repository root, which is the historical layout — so
a checkout with its data still in place needs no configuration and behaves
exactly as before. `src/contracts/paths.py` is the only place that resolves
this; nothing else should join `"assets"` or `"outputs"` onto a repo root.

**Do not use a symlink.** A symlink inside the project is what tools follow, and
it is how one dataset became twelve: every git worktree carried `assets` and
`outputs` symlinks back to the same directories, so repository auto-detection
found the same half-million files once per worktree. Point the variable at the
data and leave nothing in the tree to follow.

## Moving the data

```bash
mkdir -p /home/itec/emanuele/pointstream-data
mv assets outputs /home/itec/emanuele/pointstream-data/
export PS_DATA_ROOT=/home/itec/emanuele/pointstream-data
```

Expect the move itself to take a while — at ~6 file opens per second under load,
565,000 files is not a quick `mv` unless the destination is on the same
filesystem, in which case it is a rename and effectively instant. Keep it on the
same mount.

## What this does not change

Result paths quoted in the paper's `CLAIM` lines, and in `plans/`, stay written
as `outputs/...`. They are relative to the **data root**, not to the checkout.
A run that records its own paths should also record what they resolved to —
`paths.describe()` returns exactly that.
