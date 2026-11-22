# PointStream

PointStream studies when to send object motion as points, when to send appearance, and when to keep conventional video. Its scene-selective architecture routes shots at the server. The receiver displays video and can expose decoded locations to an application. Tennis supplies the principal evaluation. The repository also contains experiments on egocentric hand–object video.

Start with the [paper figure plan](PLAN.md) for measured figures, gaps and the next experiments. The [research brief](RESEARCH.md) summarizes the claims. The [codec endpoints](docs/codec-endpoint.md) describe encoding, decoding and inspection.

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
