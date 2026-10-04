# PointStream

PointStream is an object-centric semantic video codec. The target is an ACM TOMM submission on **30 September 2026**. Report searches transparently, including negative results, and scope claims to evidence that supports them.

## Work and execution

Run remote GPU work through `scripts/ps-fleet`, which enforces availability checks, resource claims, smoke validation, budgets, and durable monitoring. See [docs/workflow/long-jobs.md](docs/workflow/long-jobs.md) for job specifications and recovery.

Before dispatch, inspect any existing job through `scripts/ps-fleet status JOB_ID`
and its preserved supervisor receipts, then audit all eligible hosts with
`scripts/ps-fleet inspect`. A failed connection makes that host unverified, not
the whole fleet unavailable. Report each unreachable host separately and continue
checking the others. Submit the full compatible host pool; let fleet admission
claim an available node rather than pinning one in advance. Compatibility includes
input/artifact access, pinned binaries, CPU/memory requirements and GPU models.
An uncertain prior job must not be replayed or migrated: first resolve its
fleet-wide status and receipts. Independent work may use other verified hosts
only if it cannot duplicate that uncertain job. CPU-only work must use a supported
CPU admission path with equivalent claims and monitoring; do not reserve an
unneeded GPU or bypass admission to obtain portability.

Codex subagents default to `gpt-6-luna` at `max` reasoning effort unless the task explicitly specifies otherwise.

## Branches and review

Do this work on a scoped branch, not directly on `main`. Commit a coherent change when it is in a state worth keeping, and push that branch. Do not leave finished work only in the local checkout. When focused tests cover the behavior, suggest a pull request and wait for the user to ask before opening it.

## Reproducibility and checks

Keep datasets and outputs outside the code tree; do not add `assets/` or `outputs/` symlinks. Preserve the exact code revision and selected patch checksums, input identity, command, GPU UUID, environment, and native encoder/decoder paths and versions with every run. Infrastructure smoke runs are not paper evidence.

Run focused tests for the changed behavior. For this dispatcher and monitor, use:

```bash
python -m pytest -q tests/experiments/test_resource_claims.py tests/experiments/test_gpu_fleet.py tests/experiments/test_job_monitor.py tests/experiments/test_fleet_inbox.py
```

Do not run `scripts/cleanup_merged_worktrees.sh`; its `rm -rf` fallback can discard uncommitted work. Preserve paused or user-owned worktrees.

## Paper repository

The manuscript is in the sibling repository `[../67a9ea6275d3d9785ce57026/](../67a9ea6275d3d9785ce57026/)`, with its own `AGENTS.md`. Make and commit manuscript edits there; keep this repository's evidence references current.
