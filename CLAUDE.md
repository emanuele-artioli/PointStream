# POINTSTREAM

Object-centric semantic neural video codec. Instead of transmitting pixel
residuals (H.264/HEVC/AV1), the encoder transmits semantic understanding —
actor keypoints, camera poses, a background panorama, and a highly quantized
residual video — and the client reconstructs frames with generative models.
Initial domain is deliberately constrained to tennis (near-static camera,
known background, few actors). The companion ACM TOMM paper lives in
[67a9ea6275d3d9785ce57026/](67a9ea6275d3d9785ce57026/) — a separate nested
git repo (Overleaf sync); don't apply this file's code rules there.

**The core thesis — the Residual Guarantee:** server and client run the
*identical* deterministic `SynthesisEngine` (`src/shared/synthesis_engine.py`),
so the server can compute the true residual against the original and the
client can perfectly restore it. A component earns its place only if it
shrinks the residual payload by **more** than the metadata it adds; benchmark
every addition against the Whole-Frame Residual Baseline. See
`reports/7_implementation_plan.md` for the full framing.

Antigravity/Copilot read their own copies of these rules
(`.agents/rules/pointstream.md`, `.github/instructions/pointstream.instructions.md`).
This file is canonical for Claude sessions; when a convention changes, keep
the other two in sync rather than letting them drift.

## Entry point

Everything runs inside the `pointstream` conda env
(`~/.conda/envs/pointstream`). The CLI takes only two flags — **all other
options live in the config YAML**, one key per former CLI flag:

```
cd /home/itec/emanuele/pointstream   # required: src/ uses `from src....` absolute imports
conda run -n pointstream python src/main.py --input assets/real_tennis.mp4 --config config/default.yaml
```

- Any real end-to-end run **must pass `--input` explicitly** — omitting it
  falls back to a mock source and silently tests nothing real. The standard
  eval video is `assets/real_tennis.mp4`.
- `config/default.yaml` documents every knob (detector/pose/segmenter
  backends, ball extractor, `genai-backend`, codec/CRF/preset,
  `execution-pool`, `evaluation-mode`, …). For an ablation, copy it and edit
  keys; don't grow new CLI flags.
- **Default config caps at `num-frames: 10`** — fine for smoke tests, but a
  real experiment needs a config with `num-frames: null` (all frames).
- There is no config-string mock mode for `detector`/`pose-estimator`/
  `segmenter`/`ball-extractor` — despite what `config/default.yaml`'s
  comments imply, `src/main.py`'s builders only branch on real backend names
  (e.g. `"segmentation"`, anything containing `"yolo"`); an unrecognized
  value either raises or silently falls through to the default backend.
  Mock classes (`MockActorExtractor`, etc.) are wired in only by unit tests
  via direct dependency injection, not through config. For a fast **real**
  smoke run instead, use a config with `execution-pool: inline`,
  `genai-backend: null`, and a small `num-frames` (e.g. `3`) — real
  detector/pose/segmenter backends, no GenAI compositing; ~3 min for 3
  frames on this host.
- Output goes to a timestamped `outputs/<YYYYMMDD_HHMMSS_micros>/` dir. GenAI
  runs (ControlNet/AnimateAnyone) take minutes per 10 frames — background
  long runs or hand them to the `pipeline-runner` agent.

## Architecture rules

- **Strict scaffold** — no monolithic scripts: `src/main.py`, `src/shared/`
  (schemas, `SynthesisEngine`, interfaces, geometry), `src/encoder/`
  (extractors, DAG orchestrator, residual), `src/decoder/` (renderer, GenAI
  compositor/engines), `src/transport/`, `scripts/`, `tests/`, `assets/`,
  `outputs/`.
- **Symmetric synthesis:** never fork `SynthesisEngine` behavior between
  encoder and decoder — the Residual Guarantee breaks the moment server and
  client can disagree. Seeded determinism (`seed` config key) matters for the
  same reason.
- Operate on discrete `VideoChunk`s (~2 s clips); no infinite-stream code
  paths.
- Cross-module data uses the Pydantic models in `src/shared/schemas.py`
  (`VideoChunk`, `ActorPacket`, `FrameState`, keyframe/interpolate/static
  events, `EncodedChunkPayload`, …) — never raw dicts. Every transmitted
  semantic event carries `frame_id` and `object_id`.
- **Every tensor statement carries a shape-hint comment**, e.g.
  `# Shape: [Batch, Frames, Keypoints, Coords]`.
- **Mock-first:** a new extractor/neural module first returns deterministic
  dummy tensors of the exact target shape (see `MockActorExtractor`) so the
  plumbing is proven before heavy weights load.
- DAG execution via `InlineExecutionPool`/`TaggedMultiprocessPool`; tag nodes
  `@cpu_bound` (I/O, FFmpeg) or `@gpu_bound` (PyTorch) from
  `src/shared/tags.py`; pass tensors between processes via
  `torch.multiprocessing` shared memory (`make_shared_cpu_tensor`), not
  pickling.
- All transmission goes behind `BaseTransport` (`.send(payload)` /
  `.receive()`, `src/shared/interfaces.py`); currently only `DiskTransport`.
- Scene-classification routing stays modular: static "Interludes" (crowd
  shots) → traditional fallback codec; active "Exchanges" → semantic pipeline.
- Device-agnostic CUDA: fall back to single `cuda:0` with
  `torch.cuda.is_available()` checks; never hardcode `cuda:1`/multi-GPU in
  library code (multi-GPU tuning lives in scripts/config only).

## Weights

Search `/home/itec/emanuele/Models` first and **symlink** into
`assets/weights/` (see existing symlinks there); `scripts/download_weights.py`
fetches what's missing. Never expose the absolute host path in README or any
user-facing doc — users are told to place weights in `assets/weights/`.

## Environment & host

Shared remote Linux GPU server, **no root/sudo/apt**, headless — save media
and plots to disk, never `cv2.imshow()`/`plt.show()`. `pyproject.toml` is the
one and only source of truth for pip packages (add new deps there, then
`pip install -e .`); `environment.yaml` is strictly the CUDA/PyTorch
bootstrapper; never create a requirements.txt. Known pin: opencv 4.8 /
numpy 1.26.4 ABI coupling (`reports/2_scene_classification_research.md`).
`git push` works via stored credential helper; no `gh`/PRs needed. When
dependencies or structure change, update `pyproject.toml` and `README.md` in
the same pass.

## Concurrent sessions & git hygiene

Multiple Claude sessions — including spawned follow-up tasks from
`spawn_task` chips — often work in this repo at the same time, sharing
this working directory rather than an isolated worktree. A session's
uncommitted edit can be silently overwritten by another session's
read-modify-write cycle on the same file; this isn't a Claude bug, it's
the same hazard as any two processes editing a file without locking.
Confirmed 2026-07-11: an uncommitted `match_orchestrator.py` fix was
clobbered mid-session while a long real-world validation run was in
flight and four spawned sessions were active.

- **Commit a fix as soon as it passes fast checks** (ruff/mypy/unit
  tests) — don't leave it uncommitted while running a slow verification
  (a multi-minute integration test, a real GPU/pipeline run) or while
  moving on to unrelated work. Commit the code under test *before*
  kicking off the slow run, and let that run validate the committed
  state; if it finds a problem, fix and commit again rather than holding
  the fix uncommitted for the run's duration.
- Before committing, a surprising `git diff --stat <file>` showing no
  changes on a file you just edited is the tell that something reverted
  it — re-apply and commit immediately, don't spend time diagnosing why.
- For work that must not be touched by other sessions, prefer
  worktree-isolated agents over same-directory spawned sessions, or
  sequence spawned tasks instead of launching several at once against
  files they might overlap on.

## Testing — this repo has a real suite

- `python scripts/check_coverage_gate.py` is the CI entry point (runs
  `coverage run -m pytest`; threshold 80% in CI, 85% locally, override with
  `POINTSTREAM_COVERAGE_THRESHOLD`).
- Plain `pytest` excludes `integration` and `slow` markers by default
  (`pytest.ini`).
- Lint/type: `ruff check src tests scripts` and `mypy` (config in
  `pyproject.toml`); pre-commit runs both.
- Tests are necessary but not sufficient: after a pipeline change, verify
  with a real `--input assets/real_tennis.mp4` run and show the command +
  the `run_summary.json` numbers, not just "tests pass". Run `/code-review`
  after non-trivial `src/` changes.

## `outputs/` and `assets/` — deletion is unrecoverable

Both are gitignored: `outputs/` holds GPU runs that cost minutes-to-hours to
recompute, `assets/` holds the dataset, raw 4K sources, and the weight
symlinks. A `.claude/hooks/guard-rm.py` PreToolUse hook blocks `rm` against
the whole `outputs/` or `assets/` tree; deleting one specific
`outputs/<timestamp>/` run dir stays allowed. Never test destructive commands
against these real directories.

## This tooling is meant to evolve

`.claude/` (this file, `skills/`, `agents/`, `hooks/`, `settings.json`) is
part of the working setup, not frozen — if a convention gets misapplied
twice, fix the doc right then. Edits to `settings.json`/hooks take effect
next session; skills and CLAUDE.md load fresh each session.

## Reports are the research source of truth

`reports/REPORTS.md` is the dashboard: workstream status, prioritized next
steps, and the catalog of reports (scene classification, SPADE4Tennis,
Animate-Anyone integration, temporal consistency, architecture refactor, the
TOMM action matrix, the implementation plan). **Read the relevant report
before working on that area** — it records what was tried, fixed, and
disproven. After any experiment or diagnosis that produces new knowledge,
fold it back in with the `/update-reports` skill.

## Where to look for more

- Running the pipeline, config knobs, reading outputs → `/run-pipeline` skill
- Residual-Guarantee ablations (baseline vs variants, pays-for-itself
  verdicts) → `python -m scripts.benchmark_matrix run <spec.yaml>` with a
  matrix spec in `config/benchmarks/` (see
  `example_panorama_quality.yaml`); `report` mode re-tables existing runs
- Summarizing/comparing runs, size accounting → `/results-report` skill
- Folding findings into `reports/` → `/update-reports` skill
- TOMM revision checklist workflow → `/reviewer-response` skill
- Long GPU runs / training jobs → `pipeline-runner` agent
- Paper text edits → `paper-editor` agent
