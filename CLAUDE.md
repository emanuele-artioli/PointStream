# POINTSTREAM

Object-centric semantic neural video codec. Instead of transmitting pixel
residuals (H.264/HEVC/AV1), the encoder transmits semantic understanding —
actor keypoints, camera poses, a background panorama, and a highly quantized
residual video — and the client reconstructs frames with generative models.
Initial domain is deliberately constrained to tennis (near-static camera,
known background, few actors). The companion ACM TOMM paper lives in
[67a9ea6275d3d9785ce57026/](67a9ea6275d3d9785ce57026/) — a separate nested
git repo (Overleaf sync) with its own CLAUDE.md and its own conventions; don't
apply this file's code rules there.

**The core thesis — the Residual Guarantee:**

```
size(metadata) + size(residual)  <  size(full-frame encoding at equal quality)
```

Server and client run the *identical* deterministic `SynthesisEngine`
(`src/shared/synthesis_engine.py`), so the server can compute the true residual
against the original and the client can perfectly restore it. A component earns
its place **only if it shrinks the residual payload by more than the metadata
it adds** — measured against the Whole-Frame Residual Baseline (every component
disabled, residual = the whole video). This is the project's only ablation
currency; "looks better" is not a result. Full framing and every measured
verdict: `67a9ea6275d3d9785ce57026/RESEARCH_LOG.md`.

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
  real experiment needs a config with `num-frames: null` (all frames), and any
  number you report must be labeled with the config that produced it.
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

## Experiment methodology — the hard rules

The full set, with the evidence for each, is in `RESEARCH_LOG.md` (paper repo).
The ones that bite most often:

- **Symmetry is the guarantee.** Never fork `SynthesisEngine` behavior between
  encoder and decoder. The encoder computes residuals against the
  *codec-decoded* panorama, never the raw in-memory one — that asymmetry was a
  real bug that made panorama quality a silent no-op. Any new synthesis path
  gets a bit-identity check before results built on it are trusted.
- **Verify a knob is actually wired before ablating it.** Grep its *consumer*,
  not just the config schema. An unwired `residual_block_threshold` produced a
  clean, plausible, entirely fictional ablation table that stood for a day.
- **Infra failure is not a quality result.** Never rank or prune a training
  rung in which an alive variant has no score because it OOM'd or crashed.
- **Held-out gate:** no generative quality claim unless the model was trained
  without `alcaraz_highlights` and `djokovic_zverev`.
- **Scope negative results.** "Conclusively"/"definitively" is banned on
  single-clip, single-architecture experiments — that rule was written and
  violated within a day, and the claim had to be retracted.
- **Preset names are not comparable across codecs.** Compare at matched VMAF
  across a CRF ladder, and state the preset tier.
- **Invalidated runs get `mv`'d** to `outputs/_superseded/<ts>_<reason>/`,
  never `rm`'d.
- **Evaluation must run the decoder's own code path.** Symmetry applies to
  measurement, not just synthesis: `scripts/eval_checkpoint.py` builds
  strategies via `build_genai_strategy`, the same factory the compositor uses.
  A reimplemented inference path once scored ControlNet as text-to-image from
  noise while the decoder ran img2img from the reference crop — fixing only the
  measurement was worth **+6.3 dB PSNR**. If a variant scores near zero while
  others look sane, suspect the measurement before the model.
- **Metrics are scale-specific.** VMAF floor-saturates on 512×512 actor crops
  (it returned exactly 0.00) — use LPIPS/DISTS there, and keep VMAF/FVD for the
  final full frame. Ranking is by **residual bytes**; everything else is a
  diagnostic that explains why a model won or lost.

## Long training runs — never launch attached

The SSH connection to this host drops at least twice a day. Launch every
multi-hour job detached (`setsid nohup … < /dev/null &`, or `run_in_background`
from a Claude session) and verify its resume path *before* it is needed. Kill a
run early on the documented tripwires (nonzero exit, NaN loss, score below the
reference-copy floor, variance collapse) rather than letting it burn a night —
and remember an OOM is an infra failure, never a quality result. Full recipe:
the `/train-campaign` skill.

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
  library code (multi-GPU tuning lives in scripts/config only). **The GPU is
  shared** (48 GB, other processes present) — SPADE at batch 16 / 512 px OOMs.

## Weights

Search `/home/itec/emanuele/Models` first and **symlink** into
`assets/weights/` (see existing symlinks there); `scripts/download_weights.py`
fetches what's missing. Never expose the absolute host path in README or any
user-facing doc — users are told to place weights in `assets/weights/`.
Naming trap: `assets/weights/custom-controlnet` is the fine-tuned **Canny**
checkpoint (there is no `canny-controlnet` path), and
`ip-adapter-controlnet` is architecturally a fourth `ControlNetModel`, not a
diffusers-native IP-Adapter.

## Environment & host

Shared remote Linux GPU server, **no root/sudo/apt**, headless — save media
and plots to disk, never `cv2.imshow()`/`plt.show()`. `pyproject.toml` is the
one and only source of truth for pip packages (add new deps there, then
`pip install -e .`); `environment.yaml` is strictly the CUDA/PyTorch
bootstrapper; never create a requirements.txt. Known pin: opencv 4.8 /
numpy 1.26.4 ABI coupling. `git push` works via stored credential helper.
When dependencies or structure change, update `pyproject.toml` and `README.md`
in the same pass.

## Concurrent sessions & git hygiene

This repo gets worked on by multiple agent sessions at once — Claude
sessions (interactive or spawned via `spawn_task` chips, which do run in
their own isolated worktree under `.claude/worktrees/`) and other tools
that read their own copies of these rules here (Antigravity, Copilot). Any
session working directly in this checkout's main working directory — not an
isolated worktree — can silently overwrite another session's uncommitted
edit on the same file via an ordinary read-modify-write race. Confirmed
2026-07-11: an uncommitted `match_orchestrator.py` fix was clobbered
mid-session while a long validation run was in flight.

- **Commit a fix as soon as it passes fast checks** (ruff/mypy/unit
  tests) — don't leave it uncommitted while running a slow verification
  or while moving on to unrelated work. Commit the code under test *before*
  kicking off the slow run, and let that run validate the committed state.
- Before committing, a surprising `git diff --stat <file>` showing no
  changes on a file you just edited is the tell that something reverted
  it — re-apply and commit immediately, don't spend time diagnosing why.
- For work that must not be touched by other sessions, prefer
  worktree-isolated agents over same-directory spawned sessions.
- **Sweep worktrees before assuming something wasn't built.** A complete,
  tested HNeRV baseline once sat unmerged in a spawned worktree until a manual
  sweep found it. Several `claude/*` and `worktree-agent-*` branches, the `dev`
  and `may26` branches, and two stashes still hold unmerged work — don't delete
  any of them without asking.

## Testing — this repo has a real suite

- `python scripts/check_coverage_gate.py` is the CI entry point (runs
  `coverage run -m pytest`; threshold 80% in CI, 85% locally, override with
  `POINTSTREAM_COVERAGE_THRESHOLD`).
- Plain `pytest` excludes `integration` and `slow` markers by default
  (`pytest.ini`). ~383 tests, ~2 min.
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
against these real directories. **This extends to untracked files anywhere in
the repo** — a scratch file is not in git history either; read it before
removing it.

## This tooling is meant to evolve

`.claude/` (this file, `skills/`, `agents/`, `hooks/`, `settings.json`) is
part of the working setup, not frozen — if a convention gets misapplied
twice, fix the doc right then. Edits to `settings.json`/hooks take effect
next session; skills and CLAUDE.md load fresh each session.

## The paper is the primary living document

The manuscript (`67a9ea6275d3d9785ce57026/main.tex`) carries the research plan
as machine-readable comment markers (`STATUS/GOAL/HOLE/NOTE/NEXT/CLAIM(anchor):`
— spec in the paper repo's CLAUDE.md). **Before planning any experiment, grep
the paper's `HOLE()` markers** — run only the experiments the paper needs:

```
grep -n '^% *\(STATUS\|GOAL\|HOLE\|NOTE\|NEXT\|CLAIM\)(' 67a9ea6275d3d9785ce57026/main.tex
```

After a session produces committed, tested results, fold them in with the
`/update-paper` skill. `67a9ea6275d3d9785ce57026/RESEARCH_LOG.md` is the
secondary store: hard methodology rules, standing results with their real
numbers, the bug registry, the dead-end registry (read it before re-attempting
anything), the superseded-results registry (**check it before citing any
number** — several results here have been retracted), and the asset inventory.
The 13 numbered technical reports were consolidated into it on 2026-07-21
(full history via `git log --follow -- reports/<file>`).

## Where to look for more

- Running the pipeline, config knobs, reading outputs → `/run-pipeline` skill
- Residual-Guarantee ablations (baseline vs variants, pays-for-itself
  verdicts) → `python -m scripts.benchmark_matrix run <spec.yaml>` with a
  matrix spec in `config/benchmarks/`; `report` mode re-tables existing runs
- Summarizing/comparing runs, size accounting → `/results-report` skill
- Folding findings into the paper → `/update-paper` skill
- TOMM reviewer checklist workflow → `/reviewer-response` skill
- Long GPU runs / training jobs → `pipeline-runner` agent
- Paper text edits → `paper-editor` agent
- Evidence, hard rules, past dead ends →
  `67a9ea6275d3d9785ce57026/RESEARCH_LOG.md`
