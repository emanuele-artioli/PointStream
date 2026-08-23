<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     AGENTS.md minus two things a Claude session gets elsewhere: the
     host-rules block (already loaded from ~/.claude/CLAUDE.md) and
     every `scope:`-marked section (delivered on demand via
     .claude/rules/). CLAUDE.md imports this instead of AGENTS.md so a
     session pays context only for what it needs every time. -->

# POINTSTREAM — rules of engagement

This file is the single source of truth for **this project's** agent rules.
Host-wide rules (the GPU box, git, measurement discipline) live in
`~/.agent-rules/AGENTS.md` and are inlined at the end of this file so a
session opened only on this repo still sees them. Edit project rules here;
edit host rules there. Do not copy host prose into the project section.

`CLAUDE.md` imports a generated slice of this file — Claude Code does not
read `AGENTS.md`. Cursor, Codex, Antigravity and Copilot's cloud agent read
this file whole, including the host block.

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

**Report what happened.** If a run failed, say so with the output. If a step was
skipped, say that. Numbers without provenance are not results.

## Environment

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
