# PointStream

PointStream is an object-centric semantic video codec. It splits a video into
foreground (what the viewer watches) and background, and encodes each with the
best tool for it. The paper targets racket sports and egocentric hand-object
video. The submission venue and date are not yet set. Report searches
transparently, including negative results, and scope claims to the evidence.

Read [README.md](README.md) for the layout, [PLAN.md](PLAN.md) for the order of
work, [docs/components.md](docs/components.md),
[docs/resources.md](docs/resources.md), [docs/experiments.md](docs/experiments.md),
[docs/fleet.md](docs/fleet.md) and [docs/sessions.md](docs/sessions.md) (root
and item sessions, cards, reports).

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
are gpu1–gpu6. Wait on fleet jobs only with `scripts/ps-fleet wait` (returns within 45
minutes); summarize progress to the user at every return.

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

Train/test overlap: a component may be trained on data that overlaps the
evaluation set when that saves real effort; record the overlap and report the
clean subset beside the pooled number. Judges (detectors, metrics' networks)
must not have seen the frames they judge. Paper headline numbers state which
footage was seen in training, and the claim is scoped to it (user, 2026-10-21).

## Evidence

A number is evidence only if it names its job, code revision, inputs and GPU,
and comes from the real component, not a stand-in. Also record the selected
patch checksums, command, environment, and native encoder/decoder paths and
versions. Infrastructure smoke runs are not evidence.

Report comparisons as BD-rate (rate at equal quality, PCHIP over the overlapping
quality range) per metric, against an achievable anchor envelope (the upper
concave hull of the anchor's encodes). When curves do not overlap, report
"better/worse at every rate" instead of a number. Quality differences at equal
rate are secondary.

## Baselines

SVT-AV1 as the conventional codec, and a state-of-the-art neural video codec.
VVC only if a correct invocation turns out to be needed.

## Branches, checks and agents

Work on a scoped branch, not directly on `main`. Commit a coherent change when
it is worth keeping, and push the branch. When focused tests cover the
behaviour and the checks pass locally, open the pull request and enable
auto-merge with a merge commit; it merges once CI passes (user, 2026-10-23).
Leave auto-merge off only when the merge has to wait for the user's timing,
e.g. a change that would break sessions still running, and say so in the
description. The checks are the ones CI runs:

```bash
ruff check
mypy --config-file pyproject.toml
python -m pytest
```

Pass no paths to ruff or mypy: paths replace the configured file sets.

Codex subagents use `gpt-6.1-sol` at `medium` reasoning effort (user,
2026-10-16; Codex's own default) unless the task says otherwise.

## Deciding and asking

Decide alone, and record why, whenever a step can be undone: merges, Overleaf
pushes, paper text, a choice between methods. Ask the user first only when:

- the step cannot be undone: deleting or overwriting anything in `Datasets` or
  `Models`, rewriting history on a shared branch, spending GPU time beyond the
  card's budget, submitting the paper;
- a wrong call, though reversible, would be expensive (much GPU time or work);
- it needs the user's judgment (taste, what the paper should claim);
- you are not confident between the options.

The user switches between several sessions, so a question must stand on its
own: what the work is for, what is done and found, the options with their cost
and risk, your recommendation, and exactly what you need. Explain every
internal name the first time ("part 1a, the test of the hand detector on
blurred frames"). Never make the user scroll back for context
([sessions](docs/sessions.md#asking-the-user)).

## Context and compaction

Long outputs stay in the context until the next compaction and are re-read on
every step. Check a file's length before reading it, then search it or read a
range; send long command output to a file and read its tail and error lines. A
hook blocks whole reads of long files and logs. Scripts print a short summary
(status, key numbers, warning and error counts, path to the full log); read
the full log when the summary shows anything unexpected.

Keep state in files, not only in the conversation: record each fleet job ID in
the experiments entry when submitting it. When a phase ends, tell the user it is
a good moment to `/compact` and give a one-line focus
([sessions](docs/sessions.md#compaction)).

Compact instructions: keep verbatim the card's goal, scope and stop
conditions; every fleet job ID with its state; GPU-hours spent of the budget;
the decision rule's result so far; the root session's address; open questions
for the user. Drop logs, file dumps, wait output and dead ends already
recorded in files.

## Paper repository

The manuscript is the Overleaf git project `67a9ea6275d3d9785ce57026`
(`https://git.overleaf.com/67a9ea6275d3d9785ce57026`), with its own `AGENTS.md`.
The server checkout keeps a clone at `67a9ea6275d3d9785ce57026/` inside the
repository root (gitignored). Overleaf accepts pushes to `main` only; it rejects
tags. Make and commit manuscript edits there.
