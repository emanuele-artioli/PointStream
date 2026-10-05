# PointStream

PointStream is an object-centric semantic video codec. The target is an ACM TOMM submission on **30 September 2026**. Report searches transparently, including negative results, and scope claims to evidence that supports them.

## Work and execution

Run remote GPU work through `scripts/ps-fleet`, checking existing jobs and all eligible hosts before submitting to the compatible host pool. Treat connection failures per host without replaying uncertain work; see [docs/workflow/long-jobs.md](docs/workflow/long-jobs.md) for enforced availability checks, claims, smoke validation, budgets, and durable monitoring.

The shared home is a slow NFS mount (~200 ms per small-file create, 11–17 ms per open, minutes for a cold `import torch`). Do not do per-file work on it. In fleet jobs, write intermediates to `PS_SCRATCH_DIR`, pass large or many-file inputs as SHA256-identified archives in `staged_inputs`, and run from a packed `environment`; the fleet stages them on host-local disk or RAM with a safety margin ([host-local staging](docs/workflow/long-jobs.md#host-local-staging)). Outside the fleet, keep caches and scratch in `/tmp` or `/dev/shm` per [docs/setup.md](docs/setup.md).

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
