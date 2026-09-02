# POINTSTREAM — rules of engagement

This file is the single source of truth for **this project's** agent rules.

Host-wide rules are not copied here. They live in one file on this machine:

`/home/itec/emanuele/.agent-rules/AGENTS.md`

Follow that file for every session. If it is not already in context, read it
with the Read tool before doing anything else. Cursor-specific mechanics are
in `/home/itec/emanuele/.agent-rules/harness/cursor.md`.

@/home/itec/emanuele/.agent-rules/AGENTS.md

PointStream is an object-centric semantic video codec. The encoder transmits
each salient object's appearance and motion plus a background model and an
optional corrective residual; the client reconstructs frames generatively. The
current cycle is a rewrite into a platform where every component is a config
choice. Target: an ACM TOMM submission, **September 30**.

## Where things are

| You need | Read |
|---|---|
| Status, phases, what to do next | `PLAN.md` |
| The spec for one workstream | `plans/<stream>.md` — read only yours |
| What a component must satisfy | `src/contracts/` — the machine-checkable truth |
| Why the design is what it is | the paper's System Design section |
| A finding worth keeping but not core | the paper's `appendices/` |

**Sessions are scoped.** Read this file, `PLAN.md`, and the one brief for your
workstream. Do not read the whole plan tree — it does not fit, and a session that
needs all of it is scoped too broadly.

The paper lives in `67a9ea6275d3d9785ce57026/`, a **separate git repo** with its
own `AGENTS.md` and its own rules. Commit there when you change it.

## Rules that code cannot enforce

**The paper's headline claims must land where PointStream wins.** A codec paper
whose central result is "we lose to the anchor everywhere" is not a submission.
So finding and naming the regime where an object-centric codec beats a
conventional one is *part of the work*, not a lucky outcome — and when a run
comes back negative, the next question is which axis has not been tried
(content, rate, quality metric, scene count, domain), not whether to write up
the loss. Scope the headline claim to the regime where it holds and state that
boundary plainly; a claim that is true in a named regime is a result, and a
claim that is true everywhere is usually a mistake.

**Searching for the winning configuration is the method, not a compromise.**
Run the axes, see the numbers, and pick the regime where PointStream wins — that
is how this kind of result is found, and there is nothing to apologise for in it.
The only obligation the search creates is to **report it**: say which axes were
tried, what each gave, and where the claim's boundary is. A regime found by
search and reported as found is a genuine finding. The same regime presented as
though it had been predicted is a fabrication, and that is the only version of
this that is forbidden.

Pre-registered bounds serve a different purpose and are not in tension with the
above: they exist to catch a *broken measurement*, not to lock a configuration
choice. Revise a bound whenever it turns out to have been wrong, and record why.

**What is not licensed** is **relaxing the checks once the news is good**, which
is the asymmetry the bounds rule below exists to catch.

**Secondary results may be negative and should be reported as such.** "No engine
beats a pasted keyframe" (`PLAN.md` §2.17) is a real contribution and stays.
What may not happen is the *central* claim resting there. If an honest search
finds no winning regime at all, that is a finding about the approach rather than
about the run — surface it to the user early, while there is still time to
change what the paper argues, not at submission.

**Bound before believing.** Before reading any measured result, write down a
plausible best and worst case and the reasoning behind them. A result outside
that range is an alarm: investigate the measurement before reporting the number.
When a bound turns out to be wrong, **record why it was wrong** — one of ours
fired against a correct result because it had been derived in the wrong units,
and that is as worth knowing as the result.

**Never add a test to raise a coverage number.** A test that exists only to
execute lines makes the gate lie. The gate here is a required-behaviour suite —
a named list of properties that must hold — precisely because padding cannot
satisfy it.

**A flag existing is not a feature working.** Encoders in particular accept
options and ignore them. Before relying on any capability, drive it and measure
that the output changed in the way the option claims.

**Nothing in the pre-rewrite codebase is assumed correct.** Where something
"exists", that is prior art to read, not a foundation to trust.

**Every result carries all three dimensions: size, quality and speed.** Not two
of them, and not speed relegated to a limitations paragraph. A configuration that
is cheaper and better but ten times slower to encode is a different result from
one that is cheaper, better and as fast, and a table that omits the third column
cannot tell them apart. `PLAN.md` §5 item 1 already requires "rate, quality *and*
encode time on the same axes"; this makes it a property of every reported
comparison rather than of one experiment. Wall clock is already recorded per run
— the gap is that it is not reported beside the rate and the quality.

**Report what happened.** If a run failed, say so with the output. If a step was
skipped, say that. Numbers without provenance are not results.

## Environment

**This home directory is an NFS mount that serves ~6 file opens per second.**
Bandwidth is fine (measured 174 MB/s read, 283 MB/s write on one large file);
it is *per-file latency* that is not. Two consequences, and neither is fixed by
caching — a warm import measured 153.5 s against 159.5 s cold:

- **Every Python process pays an import tax proportional to file count**, at
  roughly **10 ms per file**. Measured 2026-08-29 on an idle host: `torch` costs
  **123.9 s** here (11,831 files), 254 s in the `presley` env (23,188 files) and
  252 s in `animate-anyone` (32,844 files). The process spent **3 s of CPU** to
  do it, so 96% is I/O wait. This is not specific to one env, not caused by the
  worktrees (`sys.path` has six entries), and not fixed by caching.
  So **batch work into one long-lived process** rather than many short ones —
  `experiments/tier/run_ladder.sh` pays it once per axis instead of once per
  rung.
- **And do not import `torch` when the work does not need it.** `from
  src.runner import run` costs 30.8 s and loads no torch; the ladder module adds
  another 37 s and still none. A codec-only run that constructs a YOLO backend it
  never uses pays ~800 s for nothing — which is exactly the gap between the
  ladder's first rung (950 s) and its later ones (150 s). Check what a corner
  actually needs before letting it build a perception backend.
- **Editors must not walk `assets/` or `outputs/`.** They hold ~565,000 of the
  checkout's ~580,000 files against ~700 tracked source files. Both are
  gitignored, which stops git tracking them and does nothing about an editor
  walking the filesystem. `.vscode/settings.json` in this repo carries the
  excludes for every VS Code-derived editor (VS Code, Cursor, Antigravity);
  **open a single worktree as the folder**, not the parent directory, or
  repository auto-detection finds all 12 worktrees and follows each one's
  `assets`/`outputs` symlinks back to the same half-million files.

If a session is going to run many separate Python processes, copying the env to
local disk (`/`, not home) is worth the ~3 hours it takes — but measure whether
you actually need it first, because batching usually removes the need.

- `conda run -n pointstream --no-capture-output <cmd>`; imports are absolute
  from the repo root (`from src.contracts... import ...`).
- Before merging: `ruff check`, `mypy --config-file pyproject.toml`, the tests
  for what you touched, and `python -m src.contracts.layers` for import
  direction.
- GPUs are shared; more than one machine is available. Long jobs run detached
  and checkpoint at least hourly.
- `outputs/` and `assets/` are gitignored. Cite paths; never paste their
  contents into the paper.
- Resolve external tools by **path and version**, not by name — this host has
  carried two builds of the same encoder with different capabilities.
