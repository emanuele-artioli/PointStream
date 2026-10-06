# PointStream

PointStream is an object-centric semantic video codec. It splits a video into
foreground (people, the objects they handle, the ball) and background, and
encodes each with the best tool for it. It targets racket sports and egocentric
hand-object video.

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
