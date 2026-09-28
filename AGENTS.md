# PointStream

PointStream is an object-centric semantic video codec. The target is an ACM TOMM submission on **30 September 2026**. Report searches transparently, including negative results, and scope claims to evidence that supports them.

## Work and execution

- Edit, coordinate, and inspect from the Mac checkout. Dispatch CUDA work per request with `python -m experiments.jobs.fleet`; do not start new Codex sessions on GPU hosts.
- Run `python -m experiments.jobs.fleet inspect --hosts gpu1 gpu2 gpu3 gpu4 gpu5 gpu6` before choosing a host. Failed or incomplete probes are unavailable. Launch only onto a GPU with no compute processes, memory use at or below the inspected idle baseline (default 256 MiB), utilization at or below 5%, enough free memory for the estimate plus 4 GiB, and sufficient aggregate CPU headroom.
- `fleet launch` rechecks and claims the chosen GPU by host and UUID immediately before launch. It isolates the child with `CUDA_VISIBLE_DEVICES`, records a remote supervisor and durable logs, and returns a job ID for `fleet status` or `fleet cancel`. Jobs are detached; do not rely on an SSH connection or remote Codex session to keep them alive or deliver events. Do not automatically replay or migrate a running job.
- A clean `HEAD` snapshot is the default. Add only intended tracked edits with `--include-change PATH` and intended new files with `--include-untracked PATH`; never dispatch the whole dirty checkout implicitly. Keep inputs and outputs under the external data root, outside this repository.
- Before any multi-hour compute, preprocessing, training, or evaluation run, pass a bounded smoke through the same entry point and processing path on a representative input. Do not start the full run until the smoke or pilot passes; if neither can meaningfully test the work, record why and resolve that gap before dispatch.
- Cooperative claims prevent collisions among participating PointStream jobs. They cannot stop another user starting work later, and free memory cannot guarantee an oversized workload will avoid OOM. If the supervisor detects outside GPU use, it stops only its own affected job, preserves files, and marks timing contaminated.
- Hardware ordering (Ada, A6000, RTX 8000, GV100) is only a fallback heuristic. Prefer compatible hardware and measured performance for comparable workloads.



## Reproducibility and checks

Keep datasets and outputs outside the code tree; do not add `assets/` or `outputs/` symlinks. Preserve the exact code revision and selected patch checksums, input identity, command, GPU UUID, environment, and native encoder/decoder paths and versions with every run. Infrastructure smoke runs are not paper evidence.

Run focused tests for the changed behavior. For this dispatcher and monitor, use:

```bash
python -m pytest -q tests/experiments/test_resource_claims.py tests/experiments/test_gpu_fleet.py tests/experiments/test_job_monitor.py
```

Do not run `scripts/cleanup_merged_worktrees.sh`; its `rm -rf` fallback can discard uncommitted work. Preserve paused or user-owned worktrees.

## Paper repository

The manuscript is in the sibling repository `[../67a9ea6275d3d9785ce57026/](../67a9ea6275d3d9785ce57026/)`, with its own `AGENTS.md`. Make and commit manuscript edits there; keep this repository's evidence references current.