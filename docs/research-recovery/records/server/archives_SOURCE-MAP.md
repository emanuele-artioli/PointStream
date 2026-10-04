# GPU1 PointStream archive and fleet source map

Audit date: 2026-09-30. Shared storage was scanned once through `gpu1`; all paths below retain their original host path. The inventory is read-only and does not read/copy/hash media. Fleet dispatch source checksums are preserved verbatim.

## Fleet snapshots and runs

- `fleet-provenance-gpu1.json` parses snapshot-root `dispatch.json` first and run-root dispatch as fallback. It records 157 snapshot directories, 150 run directories, 153 snapshot dispatch records with Git HEAD/source patch/snapshot SHA metadata, and 144 run dispatch records. Four snapshots have no dispatch record: `20260927T201003Z-26f7ce24`, `20260927T212735Z-3e3f9daa`, `20260929T132042Z-72c58cb5`, `20260929T140215Z-a0bb64bc`. Three dispatch-bearing snapshots have no run directory.
- Run status counts at capture: 68 complete, 60 failed, 12 cancelled, 7 contended, 2 budget exhausted, 1 running; 7 snapshots have no matching run directory. The running run was `20260930T074903Z-a18a6b9a`; no action was taken.
- `fleet-command-source-map-gpu1.json` has the exact command vector and `required_paths` for 143 script/module dispatches. Ten inline Python probes are summarized without their code and classified as infrastructure probes. No credential-like arguments were found/redacted.
- `fleet-structure-gpu1.tsv` is a depth-4 metadata listing of fleet snapshots/runs (90,858 rows); `fleet-files-gpu1.tsv` is the shallow file listing. Snapshot dispatch SHA-256 values are the recorded identity; no media was hashed.

## Git worktrees and archives

- Main repository at `/home/itec/emanuele/pointstream`: HEAD `3725388ebadb14f09c046657082d187b7735786d`, branch `main`. Confirmation worktree at `/home/itec/emanuele/pointstream-confirmation`: HEAD `5009f68c7855981bcb92d8bc1314a631eac47eb9`, branch `feat/gate-b-confirmation-and-second-domain`.
- `git-worktrees-gpu1.txt` records five `/tmp/pointstream-*` worktree paths marked prunable on gpu1. This is host-local state only; do not claim those paths are absent on other GPUs. Parent reports it preserved dirty copies from gpu3.
- Current `refs/tags/archive/20260916/*` inventory contains 16 direct-to-commit refs in `pointstream-archive-tags-gpu1.json`. Eleven correspond to the PointStream `branch-archives-verified.json` entries; five additional `stash-*` refs are present. The PointStream cleanup plan and verification agree on 11 branch/tag pairs.
- `project-cleanup-20260916/pointstream-dispatch-20260916.md` and `pointstream-worktree-audit.json` are preserved in `pointstream-cleanup-records-gpu1.txt`; filtered archive/checkout metadata is in `cleanup-pointstream-records-gpu1.json`. `cleanup-local-file-names-gpu1.tsv` inventories the cleanup preservation root; three PointStream-specific local-file copies are named in `cleanup-local-file-names-gpu1.tsv`; their detailed payloads were not recursively walked.

## Legacy outputs, data, weights, links

- `initial-lists-gpu1.tsv` is the raw depth-2 metadata inventory of the named shared roots (`assets`, `manifests`, `jobs`, `weights`, `third_party`, source/worktree roots, tmp, and cleanup root). `jobs-families-gpu1.json` groups all 38 top-level job family/script entries and direct children. `asset-identities-gpu1.json` records data family and scene names through depth 2.
- `symlink-targets-gpu1.json` resolves 15 links in the bounded asset-weight and dependency scope; all 15 targets existed at capture. No broken links were found in that scope.
- No selected report/manifests SHA-256 inventory was completed. Existing dispatch/source checksums are complete for 153 snapshots; other report/manifest paths and sizes appear in the depth-2 raw listing.

## Limits and evidence status

Two recursive metadata walks did not finish and produced zero-byte captures; the explicit reason and limits are in `recursive-coverage-status-gpu1.json`. The final inventory avoids indefinite frame/media/vendor traversal, so recursive byte/file totals and deeper dataset payload paths are gaps.

Fleet GPU probes and dispatcher metadata are infrastructure provenance, not scientific results. Job report names and saved outputs are inventory leads only; no scientific claim is made from their filenames or presence. Source/run path and checksum claims are limited to fields in the captured dispatch JSON.
