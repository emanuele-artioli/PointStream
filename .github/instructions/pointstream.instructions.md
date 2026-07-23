---
applyTo: '**'
---

<!-- GENERATED from CLAUDE.md by tools/sync_agent_rules.py — DO NOT EDIT.
     Edit CLAUDE.md and re-run the script; a pre-commit hook checks this. -->

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
numpy 1.26.4 ABI coupling (recorded in the paper repo's `RESEARCH_LOG.md`).
`git push` works via stored credential helper; no `gh`/PRs needed. When
dependencies or structure change, update `pyproject.toml` and `README.md` in
the same pass.

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
  (`pytest.ini`). ~383 tests, ~2 min.
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
symlinks. A `~/.agent-rules/scripts/guard-rm.py` PreToolUse hook (centralized
across projects, configured via `.claude/settings.json`) blocks `rm` against
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
  matrix spec in `config/benchmarks/`; `report` mode re-tables existing runs
- Summarizing/comparing runs, size accounting → `/results-report` skill
- Folding findings into the paper → `/update-paper` skill
- TOMM revision checklist workflow → `/reviewer-response` skill
- Multi-hour GPU training (launch, resume, tripwires) → `/train-campaign` skill
- Choosing and writing tests for a component → `/test-design` skill
- Long GPU runs / training jobs → `pipeline-runner` agent
- Paper text edits → `paper-editor` agent
- Evidence, hard rules, past dead ends →
  `67a9ea6275d3d9785ce57026/RESEARCH_LOG.md`

---

# Host-wide rules

These apply to every project on this host. Claude Code loads them
automatically; they are inlined here for agents that do not.

## Global environment notes

These apply across all projects/sessions on this host, not just one repo's
CLAUDE.md. **This file is the register of things that have gone wrong more
than once** — if a mistake happens twice, it belongs here, phrased as the rule
that prevents it rather than the story of the failure.

## Shared agent rules — single source of truth

Imported by reference (`@` syntax) from each coding agent's own rules file —
currently `~/.claude/CLAUDE.md` and `~/.gemini/GEMINI.md`. Edit **only this
file** for anything that should apply to every agent on this host. Put
agent-specific mechanics (tool names, invocation syntax, that agent's own
conventions) in the importing file instead, not here — this file stays
tool-agnostic so every importer can use it as-is.

### The host

Shared remote Linux **GPU server, no root/sudo/apt**, headless. Home is
`/home/itec/emanuele`. Install extra tooling with conda (Miniconda at
`/usr/local/miniconda3`) into a *separate* env — never into a project's
pinned env, because several forked third-party models are version-sensitive
and a stray `pip install` silently breaks them. Being headless, save media
and plots to disk; `cv2.imshow()`/`plt.show()` never works here.

### Python dependency management

Manage Python packages through `pyproject.toml`, not ad-hoc `pip install` in
the terminal. `environment.yaml` is reserved for bootstrapping heavy
CUDA/GPU binaries only (drivers, PyTorch wheels, compiled packages) — never
fall back to a `requirements.txt` file.

### GitHub CLI (gh)

`gh` is installed at `~/emanuele/bin/gh` (on `PATH` in every shell on this
host) and authenticated as `emanuele-artioli` via `gh auth login`
(credentials in `~/.config/gh/hosts.yml`, not tied to any one project).
Available in every project on this host — install/auth doesn't need
repeating.

**Use it proactively after every push to a repo with GitHub Actions (or
any CI):** don't assume a push landed cleanly or guess at failures from
job/step names alone.

- `gh run list --branch <branch> --limit 3` — find the run a push triggered
- `gh run view <run-id> --json status,conclusion -q '.status'` (poll) or
  `gh run watch <run-id>` — wait for it to finish (`gh run watch` can
  itself flake with a transient "Bad credentials" on the annotations
  call; a `gh run view <run-id>` after that still shows the real job
  status, so don't treat a `run watch` crash as the run having failed)
- `gh run view <run-id> --log-failed` — **the real fix for CI debugging.**
  The unauthenticated GitHub REST API only exposes job/step names and
  conclusions, never log content (log downloads 403 "Must have admin
  rights" even on public repos without an authenticated token) — that
  API alone means guessing at root causes from symptoms. Authenticated
  `gh` gives the exact failing line immediately.

Also usable the same way for `gh pr view`, `gh issue view`, `gh pr create`,
etc. wherever a GitHub-authenticated operation is needed — this isn't
CI-specific.

### Git — never destroy work you have not read

These repos get worked on by several agents at once (Claude sessions,
Antigravity, Codex, Copilot), and unmerged work has genuinely been lost here
before: a complete HNeRV baseline once sat in a forgotten worktree.

- **Read a branch before deleting it.** `git log main..<branch>` and
  `git diff main...<branch> --stat`. If it is not empty,
  `git tag archive/<branch> <branch>` and push the tag *before* deleting.
  Tags are free and make a triage mistake recoverable.
- **A worktree with uncommitted changes never gets `--force`d away.**
  Commit the changes onto that worktree's own branch, tag it, then remove
  the worktree. `git worktree remove` refusing is a warning, not an obstacle
  to route around.
- **"Superseded" needs proof, not a guess.** Compare with `git patch-id`, or
  diff the files against `main` — a branch whose commit message matches one
  on main may still hold changes main never got.
- **A branch alone does not isolate a session** — two agents in one checkout
  share one HEAD. Isolation needs a worktree *and* a branch.

### Research code — tests are a failsafe, not a formality

Cover envisioned behavior and plausible misuse of code we own. Skip tests for
unreachable branches, third-party library behavior, and errors a caller
cannot produce; this is research code and boilerplate slows the iteration
that actually matters. **A test that exists only to raise a coverage number
is a defect** — it makes the gate lie about what is verified. If deleting
padding drops the gate, lower the gate to the honest number and ratchet it
back up as real tests land.

The tests that pay for themselves here are the ones that check *the paper's
claim*, not just that the code runs: an experiment whose result violates the
thing the paper asserts should fail loudly and be marked uncitable, rather
than being caught later by a careful human reading a table.

### Long jobs must checkpoint at least hourly

SSH to this host drops a couple of times a day. Any job expected to run over
an hour checkpoints at least every 60 minutes of wall clock — independent of
its epoch/step cadence — and its resume path is verified *before* it is
relied on. Long scripts also append a progress line to their log at least
every 10 minutes, so a silent hang is visible in minutes rather than hours.
Launch detached; never attached to a shell an SSH drop takes with it.

### Plan mode: split complex plans into parallel-agent waves

When a plan has multiple pieces of work that don't share state, don't execute
it as one linear sequence. Split it into workstreams and hand each to a
subagent working in its own git worktree (a shared checkout with only a
different branch is not isolation — two sessions in one worktree share a
single HEAD). Group workstreams into **waves** ordered by dependency: a wave
starts only once every workstream it depends on has reported results back,
and every workstream within a wave launches together, not one at a time.

**Why:** validated on a multi-part refactor — this surfaced cross-workstream
issues at each wave boundary instead of at the end, and kept parallel agents
from clobbering each other's changes.

**How to apply:** worth it for genuinely multi-part, multi-file tasks where
pieces are largely independent. Skip it for small or sequential tasks — one
file, one clear order of steps — where waves are pure coordination overhead.

### Waiting for long-running commands — never hand-roll a waiter

⛔ **Never write `until ! pgrep -f <pattern>; do sleep N; done` (or any
self-written poll loop) to wait for a job.** The harness runs the loop via
`bash -c "<the whole command string>"`, and that string *contains* the
pattern — so `pgrep -f` matches the watcher's own process and the condition
can never become true. The job finishes, the watcher spins until timeout, and
the completion goes unnoticed. This has already burned >1h of wall clock.
Escaping tricks (`[p]attern`, `pgrep -P`) technically work but are still the
wrong answer: the harness already reports completion, so there is nothing to
poll for.

Pick by duration, not by habit:

- **Finishes in < 10 min** → foreground `Bash` with an explicit `timeout`
  (ms, max 600000). Output arrives in one piece and the harness kills it at
  the deadline, so it cannot hang forever.
- **Longer than that** (GPU restoration, full evaluation passes, big
  backfills) → `Bash` with `run_in_background: true`. It detaches, survives
  across turns, and **re-invokes Claude on exit** with the path to its
  output file. Read that file; do not poll for it.
- **Need progress while it runs** → `Monitor`, with a filter that matches
  failure signatures too (`Traceback|Error|FAILED|Killed|OOM`), not just the
  success marker — a success-only filter stays silent through a crash, and
  silence is indistinguishable from "still running."

`conda run -n <env> …` is not a solution to this. It is still a foreground
command subject to the same 10-minute cap, and without
`--no-capture-output` it buffers all output until exit — so on a long job it
shows nothing and then gets killed. Use it for env activation if convenient,
never as a completion-waiting strategy.

Note: `Monitor`'s progress-matching depends on the logging cadence described
in the shared "Long jobs must checkpoint" rule above — a job that goes quiet
for more than ~10 minutes gives Monitor nothing fresh to match, which looks
identical to a hang.

Same trap, different tool: **`ScheduleWakeup` is not a wait-for-completion
mechanism.** It exists solely to self-pace `/loop` dynamic-mode iterations.
A background agent or background `Bash` job already triggers a notification
the moment it finishes — there is nothing to poll for. Don't call
`ScheduleWakeup` "just to wait" for one; it also fails outright when used
this way (it requires a `prompt` unless `stop: true`), so the mistake
surfaces immediately rather than silently wasting a turn — still worth not
repeating.
