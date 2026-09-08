# POINTSTREAM — Rules of Engagement

PointStream is an object-centric semantic video codec where every component is a config choice. The encoder transmits each salient object's appearance and motion plus a reusable background model and an optional corrective residual; the client reconstructs frames generatively or from references. Target: an ACM TOMM submission, **30 September 2026**.

## Host Rules

This file is the single source of truth for this project's agent rules. Host-wide rules are not copied here; they live in one file on this machine:

@/home/itec/emanuele/.agent-rules/AGENTS.md

Follow that file for every session. Harness-specific mechanics:
- Cursor: `@/home/itec/emanuele/.agent-rules/harness/cursor.md`
- Antigravity: `@/home/itec/emanuele/.agent-rules/harness/antigravity.md`

The paper lives in `67a9ea6275d3d9785ce57026/`, a **separate git repo** with its own `AGENTS.md`. Commit there when you change manuscript text.

---

## Where Things Are

| Need | Read | Write / Update |
|---|---|---|
| Current assignment | [PLAN.md](PLAN.md) and one linked area | That area's state/actions; [PLAN.md](PLAN.md) only if summary changes |
| Gate dependency | [docs/roadmap.md](docs/roadmap.md) | [docs/roadmap.md](docs/roadmap.md) only when dependency or pass criteria changes |
| Prior decision / failure | Area evidence links, history indexes, PR discussion | PR for session detail; [docs/history/findings.md](docs/history/findings.md) for validity |
| Component behavior | `src/contracts/` and relevant code/tests | Code/contracts/tests in the owned scope |
| Human setup / run | [docs/setup.md](docs/setup.md); [README.md](README.md) for user flow | [README.md](README.md) / [docs/setup.md](docs/setup.md) alongside behavior changes |
| Research evidence | Area protocol and immutable outputs | Run records; area verdict; PR provenance |
| Paper claim | [docs/areas/paper.md](docs/areas/paper.md) and paper `AGENTS.md` | Separate paper commit; evidence references back in paper area |
| Dispatch / report / closeout | [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md) | Prompt in response; PR report; durable area update |

---

## Rules That Code Cannot Enforce

- **The paper's headline claims must land where PointStream wins.** A codec paper whose central result is "we lose to the anchor everywhere" is not a submission. Finding and naming the regime where an object-centric codec beats conventional coding is part of the work. Scope headline claims to the regime where they hold.
- **Searching for the winning configuration is the method, not a compromise.** Run the axes, observe the numbers, and locate where PointStream wins. The obligation is to report the search transparently: say which axes were tried, what each gave, and where the claim boundary lies.
- **Bound before believing.** Before reading any measured result, write down a plausible best and worst case with rationale. A result outside that range is an alarm: investigate the measurement instrument before reporting the number.
- **A flag existing is not a feature working.** Encoders accept options and ignore them. Before relying on any capability, drive it and measure that the output changed in the way claimed.
- **Every result carries all three dimensions: size, quality, and speed.** Not two of them. A configuration that is cheaper and better but ten times slower to encode is a different result from one that is as fast; a table omitting runtime cannot distinguish them.

---

## Completion and Session Boundaries

Before ending each response, check whether the requested issue is solved:
- If solved: record the result and validation in a PR, update the owning area document's current state and next action, and consider whether this is a clean session boundary. Do not create a new standalone report file for a completed session.
- If work remains: state the unresolved decision and produce a scoped continuation prompt when a handoff is useful. A prompt does not replace durable decisions or the area update.
- Keep one PR per independently revertible change; do not open a PR per reply.

### Worktree Cleanup
Cleanup is an explicit check, not a blanket deletion instruction:
- Confirm merge against fresh `origin/main`, clean status, and unique commits/diff.
- Ask before removing any worktree that may host a paused session.
- Never force removal (`--force`) or use `rm -rf` to bypass Git refusal.
- Do not run `scripts/cleanup_merged_worktrees.sh` in its current form.

---

## Environment and Verification

- Data lives outside code; configure via `.ps-data-root` marker or `PS_DATA_ROOT` (see [docs/setup.md](docs/setup.md)). Never create `assets/` or `outputs/` symlinks inside the code repo. Open a single worktree in editors.
- Keep regenerable caches (`.mypy_cache`, `.pytest_cache`, `.ruff_cache`) on host-local disk (e.g. `/tmp`), namespaced by checkout.
- Put `import sqlite3` before `import torch` in all scripts.
- Resolve external tools (`ffmpeg`, `vvencapp`) by path and version explicitly.
- Before merging changes:
  ```bash
  ruff check .
  mypy --config-file pyproject.toml
  python -m src.contracts.layers
  python -m pytest tests/runner/test_tier_end_to_end.py -q
  ```
