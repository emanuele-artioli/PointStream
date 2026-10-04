# Storage layout and attended migration

PointStream uses three roots under `/home/itec/emanuele`: `pointstream` for
source, `Datasets` for source data and experiment records, and `Models` for
model repositories and checkpoints. Generic dependencies such as Detectron2
may be sibling repositories in `/home/itec/emanuele`. There is no new
`PointStream/`, `third_party/`, or `weights/` organizational layer in those roots.

## What belongs where

| Root | Contents |
| --- | --- |
| `pointstream` | Source in `src`, `demo`, `experiments`, and `scripts`; configuration, tests, schemas, documentation, and shipped demo assets. |
| `Datasets` | Named corpora, raw/curated frames, masks, annotations, holdouts, network traces, runtime manifests, audit bundles, logs, scores, and generated experiment outputs. |
| `Models` | Model source repositories, pretrained weights, converted weights, and trained checkpoints, grouped by model family. |

Keep existing corpus identities and internal directory structure. For example,
`Egocentric-10K/raw`, its curated versions, `COCO/train2017`, and
`pointstream-demo/holdouts` remain named datasets. Flattening their individual
clips would lose useful identity and introduce filename collisions. The
`pointstream-demo` directory is a dataset, not an additional storage root.
Move the dataset's Python builder into the source checkout before retiring its
copy; do not move a code file while an active task may be editing it.

Promote the existing `Datasets/pointstream-data/{assets,audits,jobs,manifests,outputs}`
into `Datasets` as direct children. Preserve their internal run IDs and names.
Reusable model code must leave experiment job folders: HOPformer, DINOv3,
DeltaDorsal, DCVC, DiffuEraser, and HNeRV go directly under `Models`. Checkpoints
in `pointstream-data/weights`, `assets/weights`, and experiment checkpoint
folders go into their model-family directories. Detectron2 belongs in the home
as a nonmodel dependency repository. The `gvcrt-qualified` bundle is evidence
and belongs in `Datasets`.

Existing model bundles under `Models` remain in place. A destination that already
exists requires a content/revision comparison; the migration never overwrites
or automatically deduplicates it. Saved results and historical manifests retain
the original paths, commands, revision IDs, and hashes. Compatibility aliases
outside the code checkout keep these records resolvable during the transition.
Do not add data or model symlinks inside the source checkout.

The separate manuscript repository should eventually be a sibling of
`pointstream`, after reading its own policy and verifying it has no active
writer. User-owned/paused worktrees and tracked demo assets remain protected.
Generated files inside the source tree require a separate tracked-file and
consumer audit before relocation; never sweep whole directories by name.

## Path contract

`src/contracts/paths.py` centralizes paths:

- `data_root()` uses `PS_DATA_ROOT`, then `.ps-data-root`, then an existing
  `~/Datasets`, with the historical checkout fallback for development fixtures.
- `models_root()` uses `PS_MODELS_ROOT`, then the existing `POINTSTREAM_MODELS`
  alias, then `.ps-models-root`. Otherwise it uses the canonical sibling/home
  `Models` root, with the historical external weights tree as a transition fallback.
- `model_asset()` accepts model-family names and historical `assets/weights/`
  prefixes. Existing legacy weights remain usable until migration. An explicitly
  configured model root is authoritative: a missing file or dangling alias
  never silently loads another checkpoint or triggers a download.

After the complete migration and checkpoint identity verification, configure:

```bash
export PS_DATA_ROOT=/home/itec/emanuele/Datasets
export PS_MODELS_ROOT=/home/itec/emanuele/Models
```

Set the checkout-local `.ps-data-root` and `.ps-models-root` files to the same
paths for tools that do not inherit the environment. Frozen fleet workers retain
their saved configuration; stop/reconcile them before the storage cutover and
restart them through the fleet administration interface afterward. Do not edit
frozen snapshots or rewrite job specifications to update their paths.

## Migration interface

`scripts/storage_layout.py` is a standard-library, attended administration tool.
It adds no Codex allow rules. Transfer its reviewed revision to a temporary path
on gpu3; do not change an active checkout. Run `plan` there against the shared
home and keep the immutable plan and journal outside every moved subtree:

```bash
python3 /tmp/ps-storage-layout.py plan \
  --home /home/itec/emanuele \
  --out /home/itec/emanuele/Datasets/storage-plan-YYYYMMDD.json
```

On the Mac, `check-hosts` probes through Mac-to-host SSH, verifies one shared
sentinel on all six hosts, and reports only process identities, never command
lines or credentials:

```bash
python scripts/storage_layout.py check-hosts --via gpu3 \
  --home /home/itec/emanuele --out /tmp/storage-host-check.json
```

The report must show all six hosts reachable, shared storage visible, and no
PointStream reader/writer processes. Fleet workers and Cursor sessions using
these paths are blockers, even when their GPU is idle. Use a maintenance window;
do not cancel their work or kill processes to force a migration. Reports expire
after 60 seconds. Copy a fresh successful report to gpu3 and run:

```bash
python3 /tmp/ps-storage-layout.py apply \
  --plan /home/itec/emanuele/Datasets/storage-plan-YYYYMMDD.json \
  --journal /home/itec/emanuele/Datasets/storage-journal-YYYYMMDD.jsonl \
  --host-check /tmp/storage-host-check.json
```

Application refuses changed source identities, destination collisions, paths
outside the selected roots, cross-filesystem moves, existing locks, and missing
host checks. It uses the OS atomic no-replace rename; an unsupported filesystem
requires attention rather than a copy/delete fallback. Every original file keeps
its inode and bytes. The old external path gets an alias to the destination.
Plans with collisions must be resolved and regenerated before application.

After disconnecting, inspect the journal and lock owner before doing anything.
Re-running `apply` with the same plan/journal and fresh idle-host checks only
reconciles recorded identities: an acknowledged move is not replayed; a recorded
move missing its alias can have that alias completed. Unknown state is an error.
A stale lock is never stolen automatically. After confirming its recorded owner
has exited and inspecting storage identities, an administrator may retire that
lock and reconcile. Do not infer permission from a missing heartbeat.

`rollback` accepts the same arguments and fresh host checks. It reverses only
recorded moves, checks the destination inode, and removes only this transaction's
verified aliases. Recreated paths are protected. It may leave newly created empty
model-family directories; it never recursively deletes directories or data.
Newly written files stay with the preserved inode tree. Retire legacy aliases
only in a later audited maintenance window when all consumers have migrated.

## Deployment status on 4 October 2026

The source and tests implement canonical roots and guarded migration. Physical
cutover remains pending: gpu1 refused SSH, gpu4 was locked for measurements,
and fleet workers were present on gpu3, gpu5, and gpu6. Cursor also had active
code sessions. No live datasets, weights, checkpoints, or source checkout were
relocated. The generated host check and immutable plan identify these blockers
and destination collisions. Infrastructure migration smokes are not paper evidence.
