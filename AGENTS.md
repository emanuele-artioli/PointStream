# PointStream

PointStream is an object-centric semantic video codec. It splits a video into
foreground (what the viewer watches) and background, and encodes each with the
best tool for it. The paper targets racket sports and egocentric hand-object
video. The submission venue and date are not yet set. Report searches
transparently, including negative results, and scope claims to the evidence.

Read [README.md](README.md) for the layout, [PLAN.md](PLAN.md) for the order of
work, [docs/components.md](docs/components.md),
[docs/resources.md](docs/resources.md), [docs/experiments.md](docs/experiments.md)
and [docs/fleet.md](docs/fleet.md).

## Storage

- Every weight lives in `/home/itec/emanuele/Models`, by family (`YOLO/`,
  `SAM/`, …). Every dataset lives in `/home/itec/emanuele/Datasets`. Code reaches
  them through the `Models` and `Datasets` links in the repository root
  (`scripts/link-storage`; resolution in `src/segmentation/storage.py`).
- Code never downloads into the checkout. A missing weight is an error naming
  the expected path.
- Experiment outputs go under `Datasets`, never in the checkout.
- The links are excluded from git, editor watchers and search, pytest, ruff and
  mypy. Keep them excluded when adding a tool that walks the tree.

## Environments

Before adding a component, audit the repositories and models PointStream will
use and their pinned dependency versions. Use one environment if they are
compatible. Otherwise find the exact conflicts and resolve them with the fewest,
cleanly separated environments, each called through a narrow worker interface.
Record each environment's lock file and the reason it exists.

## Fleet

Remote GPU work goes through `scripts/ps-fleet`
([docs/fleet.md](docs/fleet.md)): one module entrypoint, whole-argument scale
placeholders, a smoke of at most 600 s, a validator with substantive checks,
sha256-identified inputs, a deadline and a budget. Check existing jobs and all
eligible hosts before submitting to the compatible host pool. Uncertain
submissions are inspected, never replayed; cancel only your own jobs. GPU hosts
are gpu1–gpu6.

## Shared home is slow NFS

~200 ms per small-file create, 11–17 ms per open, minutes for a cold
`import torch`. Never do per-file work on it. In fleet jobs, write intermediates
to `PS_SCRATCH_DIR`, pass large or many-file inputs as SHA256-identified archives
in `staged_inputs`, and run from a packed `environment`. Outside the fleet, keep
caches and scratch in `/tmp` or `/dev/shm`. Datasets of many small files are
stored and moved as archives.

## First run of a model on a GPU class

GPUs differ in compute capability (Ada 8.9, A6000 8.6, RTX 8000 7.5, GV100 7.0),
and libraries fall back silently (ONNX Runtime to CPU without cuDNN 9; attention
kernels without flash support before 8.0). The first time a model runs on a GPU
class, a smoke asserts the intended device, execution provider and kernels, and
the result goes into the model–GPU table in [docs/fleet.md](docs/fleet.md).
Later runs read the table instead of re-checking, and pick `gpu_models` from it.

## Experiments

GPU time and training are the constraint; code is cheap. Before a run, write its
decision rule, hypothesis, competing explanation and budget in
[docs/experiments.md](docs/experiments.md). Before building a component, run an
oracle: the cheapest upper bound of what the component could achieve, at the
operating point the decision needs. It must dominate the components it gates.
If it fails the rule, the component is not built. Then run a correctness smoke
in minutes, a bounded pilot on the one axis that matters, and only then the
scaled run. Start every sweep with the fewest settings that give a rough answer
(three, not six), and add points only where the answer is unclear. Every stage
the agent waits on ends within 45 minutes: smokes, oracles and pilots by
scope, and longer runs split into parts that each save every finished item.
Stop when the decision is made.
Rank by value to the paper first and the demo second. Details are in the
[protocol](docs/experiments.md#protocol).

## Evidence

A number is evidence only if it names its job, code revision, inputs and GPU,
and comes from the real component, not a stand-in. Also record the selected
patch checksums, command, environment, and native encoder/decoder paths and
versions. Infrastructure smoke runs are not evidence.

## Baselines

SVT-AV1 as the conventional codec, and a state-of-the-art neural video codec.
VVC only if a correct invocation turns out to be needed.

## Branches, checks and agents

Work on a scoped branch, not directly on `main`. Commit a coherent change when
it is worth keeping, and push the branch. When focused tests cover the
behaviour, suggest a pull request and wait for the user to ask before opening
it. The checks are the ones CI runs:

```bash
ruff check
mypy --config-file pyproject.toml
python -m pytest
```

Pass no paths to ruff or mypy: paths replace the configured file sets.

Codex subagents default to `gpt-6-luna` at `max` reasoning effort unless the
task says otherwise.

## Paper repository

The manuscript is the Overleaf git project `67a9ea6275d3d9785ce57026`
(`https://git.overleaf.com/67a9ea6275d3d9785ce57026`), with its own `AGENTS.md`.
The server checkout keeps a clone at `67a9ea6275d3d9785ce57026/` inside the
repository root (gitignored). Overleaf accepts pushes to `main` only; it rejects
tags. Make and commit manuscript edits there.
