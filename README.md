# PointStream

PointStream studies when to send object motion as points, when to send appearance, and when to keep conventional video. Its scene-selective architecture routes shots at the server. The receiver displays video and can expose decoded locations to an application. Tennis supplies the principal evaluation. The repository also contains experiments on egocentric hand–object video.

Start with the [paper figure plan](PLAN.md) for measured figures, gaps and the next experiments. When resuming on another platform, start with the [submission handoff](docs/submission-handoff.md). The [research brief](RESEARCH.md) summarizes the claims. The [codec endpoints](docs/codec-endpoint.md) describe encoding, decoding and inspection.

## Reviewer guide and reproduction

The manuscript's external Shanghai and Sydney rate ladders can be redrawn from
committed numerical receipts without video, model weights, or a GPU. Install
SciPy and Matplotlib, then run from the repository root, with a **new** output
directory:

```sh
card=docs/experiments/joint-operating-range-20261123
shanghai=docs/experiments/shanghai-wider-20261123
PYTHONPATH=. python "$card/compare.py" OUTPUT_DIRECTORY \
  --shanghai-wider "$shanghai/q23-metrics.json" "$shanghai/q17-metrics.json" "$shanghai/q640-metrics.json" \
  --ordinary "$card"/ordinary-*.json --roi "$card"/foreground-roi-*.json
```

This produces `foreground.pdf`, `players.pdf`, `ball.pdf` and `results.json`.
The numerical output reproduced the committed `results.json` byte for byte
on the local workstation. For the twelve-match ball metric table, run:

```sh
python docs/experiments/ball-metric-suite-20261123/summarize.py OUTPUT.json \
  docs/experiments/ball-metric-suite-20261123/match*-result.json
```

The resulting JSON also matched the committed `summary.json` byte for byte.
Use fresh output paths; these checks redraw and aggregate existing measurements,
not independently recompute the experiments. The [replication handoff](docs/submission-handoff.md#redraw-and-prepare-the-replication-release)
links the underlying receipts and the development foreground figure. Original
measurements require restricted footage, source labels, models, native streams,
and the recorded environments; a public replication package is not yet prepared.
Each receipt identifies the inputs and run. Check rights before distributing media.

| Manuscript figures and tables | Evidence and reproduction inputs |
|---|---|
| Figs. 1–2, broadcast visits and system | [Visit timeline](docs/paper/figures/passage-intro-20261122/) and [architecture](docs/paper/figures/system/) |
| Figs. 3–4, foreground comparisons and rate ladders | [Reviewed foreground](docs/paper/figures/reviewed-foreground/) and [joint ladders](docs/experiments/joint-operating-range-20261123/) |
| Figs. 5–6 and 8, TEST ball fidelity and machine localization | [Twelve-match receipts](docs/experiments/cohort-expansion-20261122/) and [ball figure assets](docs/paper/figures/ball/) |
| Fig. 7 and cross-sport table, diffusion pilot and transfer | [Sampler and multisport receipts](docs/experiments/tomm-validation-20261124/) |
| Figs. 9–10, rates and pose coverage | [Byte ledger](docs/paper/figures/complete-wire-cost/) and [coverage and routing audit](docs/experiments/tomm-validation-20261124/) |
| TEST matrix, startup costs and artificial pose gaps | [Match identities](docs/experiments/cohort-expansion-20261122/test-events.md), [Shanghai startup](docs/experiments/test-crop-joint-20261122/match102-reference-cost.json), [Sydney startup](docs/experiments/tomm-reference-price-20261125/), and [Sydney gap diagnostic](docs/experiments/tomm-pose-bridge-20261125/) |

## Research priorities

The [active submission plan](docs/tomm-submission-plan-20261124.md) takes
precedence over older experiment queues. Prioritize completing the external
racket-sport validation and checking pose-gap interpolation for artifacts.
After the submission-critical work, the following remain opportunities rather
than established results:

1. **VISOR:** run a dedicated hand-object pilot with the current keypoint and reusable-appearance architecture.
2. **Subjective assessment:** design a sufficiently powered viewer study using licensed comparison material.
3. **Distilled receiver:** measure fidelity and complete playback cost of fewer-step or TensorRT rendering.
4. **Racket and equipment keypoints:** evaluate a small equipment channel against independent racket and grip labels.

The [archive dependency inventory](docs/submission-archive-inventory.md)
identifies what must be traced before preparing a smaller public export.

## Layout

| Path | Contents |
|---|---|
| `src/segmentation/` | Foreground/background segmentation: SAM 3.1 and YOLOE backends, lossless masks, evaluation, CLI |
| `src/codecs/` | Codec workers (DCVC-UF) |
| `env/`, `environment.yaml` | The environment recipe, patches and locks |
| `experiments/audit/` | Environment-audit smokes |
| `tools/` | Dataset acquisition and samples (`datasets/`), weight placement and MANO conversion (`models/`) |
| `experiments/jobs/` | Fleet dispatcher, workers, resource claims, host-local staging, packed environments |
| `scripts/ps-fleet` | The fleet's one entry point |
| `scripts/link-storage` | Creates the `Models` and `Datasets` links |
| `tests/` | `segmentation/` and `fleet/` |
| `docs/` | [components](docs/components.md), [resources](docs/resources.md), [experiments](docs/experiments.md), [fleet](docs/fleet.md) |

Rules for working in this repository are in [AGENTS.md](AGENTS.md); the order
of work is in [PLAN.md](PLAN.md).

## Setup

```bash
scripts/link-storage
pip install -e ".[dev]"
```

That is enough for the checks. The full environment (SAM 3.1, YOLOE, HaMeR,
WiLoR, HOT3D tooling, DCVC-UF, SVT-AV1) is one prefix built on host-local disk
by `env/build.sh` from `environment.yaml`, with locks in `env/locks/`; fleet
jobs run its packed copy ([resources](docs/resources.md#environments)).

## Checks

```bash
ruff check
mypy --config-file pyproject.toml
python -m pytest
```

## Segmentation

```bash
python -m src.segmentation run --backend yoloe-26n --domain tennis --source CLIP --out OUT
python -m src.segmentation --help
```

## Archive

Everything before this layout is at the tag `archive/pre-reset-2026-10-05`.
