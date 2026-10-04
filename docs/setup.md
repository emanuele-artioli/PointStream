# Setup and execution

PointStream has a local coordinator and remote Linux compute hosts. Keep the checkout and agent work on the Mac; dispatch CUDA workloads to whichever GPU server passes the current admission checks. SSH aliases `gpu1` through `gpu6` and the pinned server environment are already provisioned.

## Data and environment

Datasets and run outputs live outside the Git tree on the shared data filesystem. The canonical roots are `/home/itec/emanuele/Datasets` and `/home/itec/emanuele/Models`. During migration, the GPU hosts retain the legacy `/home/itec/emanuele/pointstream-data` alias; the dispatcher sets `PS_DATA_ROOT` in every remote job. See [storage layout and migration](workflow/storage-layout.md) for the cutover checks and current deployment status. Do not add `assets/` or `outputs/` symlinks to the repository. The Python path resolver accepts `PS_DATA_ROOT`, a checkout-local `.ps-data-root` marker, an existing `~/Datasets`, or the historical repository-root fallback; remote runs always use the explicit shared root.

The GPU hosts provide `/home/itec/emanuele/.conda/envs/pointstream` and native tools such as FFmpeg and `vvencapp`. Do not mutate that pinned environment with ad-hoc package installs. Inspect and record the exact executable paths and versions needed by an experiment; an FFmpeg build string alone does not prove VVC decoding is available.

The local dispatcher uses the Python standard library and SSH. Project tests can run locally with the project dependencies from `pyproject.toml`. Local CUDA is not required.

## Inspect and submit remote work

Use the canonical entry point from the Mac:

```bash
scripts/ps-fleet status
scripts/ps-fleet inspect
scripts/ps-fleet submit /absolute/external/data/job.json
scripts/ps-fleet status JOB_ID
scripts/ps-fleet events JOB_ID
scripts/ps-fleet cancel JOB_ID
```

Inspect existing requests before submitting and use the complete compatible host
pool. Probe failures apply per host; a missing acknowledgement never authorizes
replay. Read [long jobs](workflow/long-jobs.md) for the immutable specification,
shared-inbox doctor checks, workers, recovery and bounded artifact export.

The complete probe checks UUIDs, compute processes, memory, utilization, CPU
headroom, inputs, runtime and native tools. Admission requires no compute process,
used memory <=256 MiB, utilization <=5%, free memory >= the saved estimate plus
4 GiB, and aggregate CPU headroom. Worker selection and launch both recheck this
policy under resource claims. GPU model filters express compatibility, not a
performance guarantee.

One processing path and common arguments serve smoke and full stages. The
workload-specific validator must check outputs; exit zero alone does not pass
smoke. Full promotion requires validation, unchanged identities and enough
remaining budget for its saved estimate. Waiting and execution share an absolute
deadline. Output roots and input/checkpoint identities must be explicit.

The selected code revision and reviewed changes are frozen independently of the
remote primary checkout. Supervisors survive disconnection and Mac sleep, keep
separate smoke/full artifacts and stop only their owned processes on contention.
Do not use unrestricted legacy launches or an inline Python command. CPU-only
work needs a supported CPU admission path with equivalent isolation and claims;
reserving a needless GPU is not a substitute.

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
