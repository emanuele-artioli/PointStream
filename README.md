# PointStream

An object-centric hybrid video-codec research project targeting ACM TOMM by
**30 September 2026**. Eligible scenes use reusable background information and
object appearance/motion, with optional generation and corrective residuals;
ineligible scenes use conventional coding. No confirmed rate–quality win has
been established yet. The current search keeps generation off.

## Start here

- `PLAN.md`: concise current state and next task.
- `HANDOFF.md`: next-session entry point and operational cautions.
- `plans/ROADMAP.md`: submission gates, priorities and harness assignments.
- `plans/README.md`: active/parked briefs; `plans/done/`: historical reports.
- `AGENTS.md`: shared project instructions for coding agents.

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | Codec components, contracts and runner |
| `tests/` | Behaviour and regression tests |
| `experiments/` | Tracked experiment drivers and analysis code |
| `scripts/` | Setup, training, test-gate and reproducibility tools |
| `config/`, `manifests/` | Configurations and input descriptions |
| `outputs/` | Generated runs/logs under the data root; not source code |
| `67a9ea6275d3d9785ce57026/` | Manuscript: a separate Overleaf-synced Git repo |

Keep `experiments/` and `outputs/` separate. The former reproduces a study; the
latter contains its results. `DATA.md` explains `.ps-data-root` and external data
storage. Do not merge large generated data into the tracked code tree. Source
footage is YouTube-derived and is not promised for redistribution.

## Agents and tools

Codex reads `AGENTS.md`; a project `.codex/` directory is optional and only
needed for project-specific Codex settings. Shared instructions should not be
duplicated there. `.cursor/rules/host.mdc` points Cursor to host instructions;
`.github/` contains CI and Copilot instructions, not just editor settings.
Claude uses `CLAUDE.md`. The old `.claude/` held temporary worktrees and was
removed when those worktrees were retired; no replacement directory is needed.

See the official documentation for
[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) and
[optional project configuration](https://learn.chatgpt.com/docs/config-file/config-basic).

Codex/Claude handle high-level analysis and delicate integration. Cursor and
VS Code with Antigravity handle bounded routine tasks. Each session receives
one brief and returns the fields in `plans/SESSION-REPORT.md`.

## Environment and checks

Use the existing `pointstream` conda environment on this host. No sudo/apt and
no ad-hoc installs into that pinned environment. Dependencies are declared in
`pyproject.toml`; `environment.yaml` bootstraps GPU binaries. Set `PYTHONPATH`
to the actual checkout and keep regenerable caches on local disk.

```sh
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export MYPY_CACHE_DIR=/tmp/mypy-pointstream RUFF_CACHE_DIR=/tmp/ruff-pointstream
conda run -n pointstream --no-capture-output ruff check
conda run -n pointstream --no-capture-output mypy --config-file pyproject.toml
conda run -n pointstream --no-capture-output python -m src.contracts.layers
conda run -n pointstream --no-capture-output pytest -o cache_dir=/tmp/pytest-pointstream
```

Use executable paths and versions recorded by the experiment protocol. The
background stream, residual and independent reference may use different codec
implementations/presets; do not conflate them. Build the paper from its own repo
with `conda run -n tex --no-capture-output bash tools/build_pdf.sh`.
