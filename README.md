# PointStream

PointStream is an object-centric semantic video codec. It splits a video into
foreground (people, the objects they handle, the ball) and background, and
encodes each with the best tool for it. It targets racket sports and egocentric
hand-object video.

## Layout

| Path | Contents |
|---|---|
| `src/segmentation/` | Foreground/background segmentation: SAM 3.1 and YOLOE backends, lossless masks, evaluation, CLI |
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

Add `.[yoloe]` for the YOLOE backend. SAM 3.1 runs in its own environment
([resources](docs/resources.md#models)).

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
