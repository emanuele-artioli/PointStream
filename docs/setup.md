# Setup and execution

PointStream has a local coordinator and remote Linux compute hosts. Keep the checkout and agent work on the Mac; dispatch CUDA workloads to whichever GPU server passes the current admission checks. SSH aliases `gpu1` through `gpu6` and the pinned server environment are already provisioned.

## Data and environment

Datasets and run outputs live outside the Git tree on the shared data filesystem. The GPU hosts use `/home/itec/emanuele/pointstream-data`; the dispatcher sets `PS_DATA_ROOT` in every remote job. Do not add `assets/` or `outputs/` symlinks to the repository. The Python path resolver accepts `PS_DATA_ROOT`, a checkout-local `.ps-data-root` marker, or the historical repository-root fallback; remote runs always use the explicit shared root.

The GPU hosts provide `/home/itec/emanuele/.conda/envs/pointstream` and native tools such as FFmpeg and `vvencapp`. Do not mutate that pinned environment with ad-hoc package installs. Inspect and record the exact executable paths and versions needed by an experiment; an FFmpeg build string alone does not prove VVC decoding is available.

The local dispatcher uses the Python standard library and SSH. Project tests can run locally with the project dependencies from `pyproject.toml`. Local CUDA is not required.

## Inspect and launch remote work

From the repository root, inspect current host and GPU state:

```bash
python -m experiments.jobs.fleet inspect --hosts gpu1 gpu2 gpu3 gpu4 gpu5 gpu6
```

The probe checks reachability, GPU UUIDs, compute-process lists, memory, utilization, CPU headroom, the shared data root, the pinned Python environment, and native tool versions. A failed or malformed response makes that host unavailable. The default admission limits require no compute process, at most 256 MiB used, at most 5% utilization, free memory of the requested estimate plus a 4 GiB margin, and CPU headroom for the full declared thread allowance. Slurm state and GPU utilization alone do not establish GPU availability.

For experiment-specific binaries, pass `--require-command NAME` (a PATH command such as `ffmpeg`) or an absolute executable path. The dispatcher checks each reachable candidate before selection and rechecks the chosen host before starting the child.

Launch a bounded job to one or more candidate hosts:

```bash
python -m experiments.jobs.fleet launch --hosts gpu5 gpu6 \
  --gpu-memory-mib 12000 --cpu-threads 8 --budget-hours 2 \
  --require-path /home/itec/emanuele/pointstream-data/assets/dataset/alcaraz_highlights/segmentations/scene_000 \
  --require-path /home/itec/emanuele/pointstream-data/outputs/bp21-headroom/clips/alcaraz_highlights/scene_000/window \
  -- /home/itec/emanuele/.conda/envs/pointstream/bin/python -c \
  'import os; from experiments.tier.run import main; raise SystemExit(main(["--tiers", "fast", "--frames", "8", "--out", os.path.join(os.environ["PS_JOB_DIR"], "report.json")]))'
```

`--gpu-memory-mib` is the workload's estimated peak device memory, not a reservation size. Default to one GPU and declare CPU threads honestly. Use `--hosts` to restrict candidates to servers with hardware compatible with the experiment; the fleet cannot infer application-specific GPU constraints from an arbitrary command. The fallback device order is Ada, A6000, RTX 8000, then GV100. When comparable workload timings exist, pass `--prefer-gpu-name SUBSTRING` once per GPU family in measured performance order; the manifest records that preference. The current pilot has no comparable cross-model timing, so use the fallback unless new evidence is available.

The remote job runs from a unique snapshot directory. A clean `HEAD` snapshot is the default; only pass `--include-change PATH` for an intended tracked edit and `--include-untracked PATH` for an intended new file. The manifest records included checksums and excluded dirty paths. Never assume an active checkout on a GPU host matches the Mac revision.

The monitor detaches on the server and owns the child process group, GPU/CPU claims, durable log, and status. An SSH disconnect or laptop sleep does not stop the run. Use the returned job ID to retrieve status or request cancellation:

```bash
python -m experiments.jobs.fleet status JOB_ID
python -m experiments.jobs.fleet cancel JOB_ID
```

The local manifest reports the host and remote `run_dir`. Retrieve result files from that directory with `scp`; keep the copy outside the Git tree. Remote jobs are never silently replayed or moved to another host. If foreign GPU use appears after launch, the supervisor stops only its own child, preserves its files, and marks timing contaminated.

## Reproducibility and operational limits

Record the input manifest/revision and identities, exact command, local Git `HEAD`, selected patch checksum, runtime environment, GPU UUID, and native encoder/decoder paths and versions. Infrastructure probes and short smoke runs validate execution only; they do not create paper evidence.

Atomic shared-filesystem claims coordinate participating PointStream jobs and recheck immediately before child launch. Other users are not prevented from allocating a GPU later. A free GPU and the dispatcher’s memory check cannot guarantee that a workload will fit; choose the estimate from a representative peak and preserve an OOM margin.

The host bootstrap keeps Codex application state, editor servers, and regenerable Python caches on each host's local disk while home is shared over NFS. Keep those runtime directories and cache settings when changing agent instructions. In standalone shells, checkout-specific cache paths can be set as follows:

```bash
PS_CACHE_ROOT="/tmp/pointstream-cache-$(pwd -P | shasum | cut -c1-16)"
mkdir -p "$PS_CACHE_ROOT"
export MYPY_CACHE_DIR="$PS_CACHE_ROOT/mypy"
export RUFF_CACHE_DIR="$PS_CACHE_ROOT/ruff"
export PYTHONPYCACHEPREFIX="$PS_CACHE_ROOT/pycache"
python -m pytest -o "cache_dir=$PS_CACHE_ROOT/pytest" tests/runner/test_tier_end_to_end.py -q
```

Import `sqlite3` before `torch` in entry points that load Torch on the Linux GPU hosts; this is a host ABI workaround, not a blanket import for unrelated modules.

## Verification

For the dispatcher and job monitor, run the focused claim, fleet, and monitor tests from the Mac checkout:

```bash
python -m pytest -q tests/experiments/test_resource_claims.py tests/experiments/test_gpu_fleet.py tests/experiments/test_job_monitor.py
```

Run Ruff 0.11.2 from the pinned server environment against changed Python files. Use the project's applicable CI, static-type, layer, and synthetic-pipeline checks before merging code changes. Documentation-only changes need link and command review plus `git diff --check`; they do not need a GPU run.
