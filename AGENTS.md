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

## GPU command permissions

- Use the local fleet dispatcher as the normal remote execution path: `python -m experiments.jobs.fleet inspect`, `launch`, `status`, and `cancel`. Follow the resource checks, task budgets, snapshot selection, and provenance requirements above. Run from the reviewed PointStream checkout; `python -m` resolves modules from the current environment and is not a security boundary.
- Command approval is separate from task authorization. For network-dependent dispatcher calls in a network-disabled sandbox, submit the exact command through the execution tool's approval mechanism (`sandbox_permissions="require_escalated"` when available). State the purpose, hosts, and resource budget in the justification. Inspect approval or automatic-review results before continuing. A granted approval does not authorize work outside the user's assigned scope or budget.
- Require per-invocation approval for raw `ssh` and dispatcher commands. Do not create blanket `allow` rules for `ssh`, `python`, or the dispatcher. In the local Codex rules, use `decision="prompt"` for these prefixes. Never broaden global permissions or change approval rules merely to make a command run; obtain explicit user authorization for policy changes.
- Submit dispatcher commands directly. Do not hide them inside a generic Python script, shell loop, or retry wrapper: an approval for `ssh` or `fleet inspect` does not transfer to an interpreter that spawns it. Batch independent calls in the tool orchestrator instead. If a probe reports DNS failure, distinguish a sandbox denial from a host-side DNS problem before changing DNS, VPN, or SSH settings.
- Raw SSH is an exception for explicitly authorized, bounded diagnostics. Review the exact host and remote command for that invocation. Do not use it to bypass dispatcher GPU admission, claims, snapshots, supervision, or job budgets. Do not create remote agent sessions, tunnels, forwarding, or credential changes as connection workarounds without specific user authorization.
- After a disconnect, inspect the recorded job ID using `fleet status`; never replay a launch automatically. Cancel only task-owned jobs. Treat full-probe timeouts as unavailable even when a simpler SSH check succeeds.

## Branches and review

Do this work on a scoped branch, not directly on `main`. Commit a coherent change when it is in a state worth keeping, and push that branch. Do not leave finished work only in the local checkout. When focused tests cover the behavior, suggest a pull request and wait for the user to ask before opening it.

## Reproducibility and checks

Keep datasets and outputs outside the code tree; do not add `assets/` or `outputs/` symlinks. Preserve the exact code revision and selected patch checksums, input identity, command, GPU UUID, environment, and native encoder/decoder paths and versions with every run. Infrastructure smoke runs are not paper evidence.

Run focused tests for the changed behavior. For this dispatcher and monitor, use:

```bash
python -m pytest -q tests/experiments/test_resource_claims.py tests/experiments/test_gpu_fleet.py tests/experiments/test_job_monitor.py
```

Do not run `scripts/cleanup_merged_worktrees.sh`; its `rm -rf` fallback can discard uncommitted work. Preserve paused or user-owned worktrees.

## Paper repository

The manuscript is in the sibling repository `[../67a9ea6275d3d9785ce57026/](../67a9ea6275d3d9785ce57026/)`, with its own `AGENTS.md`. Make and commit manuscript edits there; keep this repository's evidence references current.