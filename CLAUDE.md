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
every addition against the Whole-Frame Residual Baseline. See the paper repo's
`RESEARCH_LOG.md` for the full framing.

**This file is the only rule file to edit by hand.** `AGENTS.md`,
`.agents/rules/pointstream.md` (Antigravity) and
`.github/instructions/pointstream.instructions.md` (Copilot) are *generated*
from it by `tools/sync_agent_rules.py`, which also inlines the host-wide
`~/.claude/CLAUDE.md` that only Claude loads automatically. Edit CLAUDE.md,
then re-run the script — a pre-commit hook fails the commit if the generated
files are stale.

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
numpy 1.26.4 ABI coupling (recorded in the paper repo's `RESEARCH_LOG.md`).
`git push` works via stored credential helper; no `gh`/PRs needed. When
dependencies or structure change, update `pyproject.toml` and `README.md` in
the same pass.

## Concurrent sessions & git hygiene

This repo gets worked on by multiple agent sessions at once — Claude
sessions (interactive or spawned via `spawn_task` chips, which do run in
their own isolated worktree under `.claude/worktrees/`) and other tools
that read their own copies of these rules here (Antigravity, Copilot; see
"This tooling is meant to evolve" below). Any session working directly in
this checkout's main working directory — not an isolated worktree — can
silently overwrite another session's uncommitted edit on the same file
via an ordinary read-modify-write race; this isn't specific to any one
tool, it's the same hazard as any two processes editing a file without
locking. Confirmed 2026-07-11: an uncommitted `match_orchestrator.py` fix
was clobbered mid-session while a long real-world validation run was in
flight and other sessions were concurrently active on this repo.

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

### Which branch a session works on

A branch only isolates a session if it also has its own working directory —
two agents sharing this checkout share one HEAD, so "make a branch" alone
does not prevent the race above.

- **Substantive code change** (a refactor, a new component, anything
  spanning more than a file or two): take a worktree *and* a branch —
  `git worktree add ../wt-pointstream/<slug> -b <type>/<slug>` with a
  `refactor/`, `feat/`, `fix/` or `exp/` prefix. Push after every logical
  commit so the work survives an SSH drop. When it is green, **suggest**
  merging and let the human do it — never self-merge to `main`.
- **Small fix, doc edit, or a session whose real work is running
  experiments:** stay on `main` in this checkout and commit as soon as fast
  checks pass. Worktrees don't carry `outputs/` or `assets/`, so run-only
  sessions belong here anyway.
- Deleting a branch requires reading it first (`git log main..<branch>`);
  if it is not empty, `git tag archive/<branch>` and push the tag before
  deleting. A worktree with uncommitted changes gets those changes
  committed onto its own branch before removal — never `--force` them away.

## Testing — the suite is a scientific failsafe, not a compile check

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

Three tiers, each catching a different kind of wrong:

1. **Unit tier** — behavior and misuse of pure logic, mocks for anything
   heavy, CPU-only, runs on every push.
2. **Stage-contract tier** — every stage validates its own output as it
   produces it, via validators on the `src/shared/schemas.py` models
   (masks non-empty, panorama and residual dimensions consistent,
   `sizes_bytes` actually summing, a null `psnr_mean` failing the run).
   A broken stage fails *there*, not three stages downstream.
3. **Goal-invariant tier** (`-m invariants`) — checks the *paper's* claim on
   a real run: the Residual Guarantee itself, payload accounting, quality
   floors. Violations are written into that run's own `run_summary.json`
   under `invariant_failures`, and **a run with a non-empty
   `invariant_failures` is never citable** — re-check it before it reaches a
   report or the paper.

**Every diagnosed bug or newly imagined edge case gets a test in the same
session it is diagnosed** — the RESEARCH_LOG dead-end entry and the
regression test are written together. Deleting a test requires saying why
its failure mode is now impossible.

Research code, so keep tests honest and thin: cover envisioned behavior and
plausible misuse of code we own. No tests for unreachable branches, for
third-party library behavior, or for errors a caller cannot produce. **A test
that exists only to raise the coverage number is a defect** — if the gate
fails after deleting padding, lower the gate to the honest number and ratchet
it back up as real tests land.

## Long jobs must checkpoint at least hourly

SSH to this host drops a couple of times a day, and a job can also be killed
by accident. The rule is therefore about *how much progress a kill can cost*,
not about how long a job may run:

- Any job expected to exceed an hour **checkpoints at least every 60 minutes
  of wall clock**, independent of its epoch or step cadence, and its resume
  path is verified *before* it is relied on. A training script that cannot
  checkpoint hourly must be cheap to restart from scratch (well under an
  hour) or be redesigned.
- Long-running scripts append a progress line (step, loss or metric,
  timestamp) to their log **at least every 10 minutes**, so a silent hang is
  visible in minutes and `Monitor` always has something fresh to match on.
- Launch detached (`run_in_background`, or `setsid nohup … < /dev/null &`) —
  never attached to a foreground shell that an SSH drop takes with it. See
  `/train-campaign` for the full launch/resume/tripwire workflow.

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

## The paper is the primary living document

The manuscript (`67a9ea6275d3d9785ce57026/main.tex`) carries the research plan
as machine-readable comment markers (`STATUS/GOAL/HOLE/NOTE/NEXT/CLAIM(anchor):`
— spec in the paper repo's own CLAUDE.md). **Before planning any experiment,
grep the paper's `HOLE()` markers** so we only run what the paper needs. After
a session produces committed, tested results, fold them in with the
`/update-paper` skill.

`67a9ea6275d3d9785ce57026/RESEARCH_LOG.md` is the secondary store: hard
methodology rules, standing results with their real numbers, the codec
anchors, the invalid-G2-campaign record, and the bug / dead-end / superseded
registries. **Read it before re-attempting anything** — it exists to stop us
repeating disproven work. `67a9ea6275d3d9785ce57026/reviewers_comments.md` is
the authoritative TOMM referee checklist.

The 13 numbered reports under `reports/` were consolidated into the paper and
RESEARCH_LOG on 2026-07-21 and the tree was deleted; **do not recreate a
`reports/` tree** (full history via `git log --follow` in the paper repo).

## Where to look for more

- Running the pipeline, config knobs, reading outputs → `/run-pipeline` skill
- Residual-Guarantee ablations (baseline vs variants, pays-for-itself
  verdicts) → `python -m scripts.benchmark_matrix run <spec.yaml>` with a
  matrix spec in `config/benchmarks/` (see
  `example_panorama_quality.yaml`); `report` mode re-tables existing runs
- Summarizing/comparing runs, size accounting → `/results-report` skill
- Folding findings into the paper → `/update-paper` skill
- TOMM revision checklist workflow → `/reviewer-response` skill
- Multi-hour GPU training (launch, resume, tripwires) → `/train-campaign` skill
- Choosing and writing tests for a component → `/test-design` skill
- Long GPU runs / training jobs → `pipeline-runner` agent
- Paper text edits → `paper-editor` agent
