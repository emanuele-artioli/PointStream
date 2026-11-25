# Submission archive dependency inventory

Snapshot examined: `4f7fa103` (PR #457), recorded in the local tag
`archive/pre-tomm-clean-2026-11-25`. The working tree has roughly 4,334
tracked `docs/` files, 498 `experiments/` files and 302 `tests/` files. These
are not a list of safe deletion candidates. Receipts preserve failed and
superseded results as well as the current evidence; their original paths and
identities matter. This inventory supports an export, not in-place deletion.

| Area | Active dependency and reason to retain |
|---|---|
| `src/`, `experiments/`, `tools/`, `scripts/` | Codec and evaluation code, fleet entry points and workers. `pyproject.toml` discovers `src*` and `experiments*`; mypy checks `src`, `experiments` and `tests`. Scripts, registrations and experiment documents can refer to workers outside a figure's own folder. |
| `env/`, `environment.yaml`, `pyproject.toml` | Environment recipe, locks, patches, package dependencies and CI tool pins; native encoders, models and source archives are external and identified by experiment receipts. |
| `docs/experiments/` and `docs/paper/` | Source data, figure scripts and PDFs, registrations, checksums, negative results and historical provenance. Joint-ladder reproduction also reads the separate Shanghai wider card. The ball table uses twelve result files. |
| `tests/`, `.github/workflows/`, `pytest.ini` | CI checks and their fixtures. Determine import, subprocess and fixture dependencies before exporting any subset. |
| `docs/fleet.md`, `experiments/jobs/`, `scripts/ps-fleet` | Job specifications, staging and validation machinery; historical job specs may require workers and files outside their card. |
| `Models`, `Datasets` | External symlinks, not repository contents. Model weights, footage and experiment outputs must be obtained under their respective access and license terms. |

A narrowly identified initial **copy-only** export candidate is the numeric
ball summary: `docs/experiments/ball-metric-suite-20261123/summarize.py`, its
12 `match*-result.json` inputs, `summary.json` and the card README. The
aggregator reproduced the checked-in summary byte for byte. This is an
aggregation example, not a complete experiment replication package: inspect
its receipts, licenses, dependencies and machine-local paths before release.

Before any archival removal, create and verify a recoverable complete snapshot,
trace imports, CLI entrypoints, tests, job specs, documentation links and
manuscript figures transitively, and validate the proposed export in isolation.
The inventory retains the historical code and receipts until their transitive dependencies have been checked.
