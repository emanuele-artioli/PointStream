# POINTSTREAM — Rules of Engagement

PointStream is an object-centric semantic video codec where every component is a config choice. The encoder transmits each salient object's appearance and motion plus a reusable background model and an optional corrective residual; the client reconstructs frames generatively or from references. Target: an ACM TOMM submission, **30 September 2026**.

## Host Rules

This file is the single source of truth for this project's agent rules. Host-wide rules are not copied here; they live in one file on this machine, and each harness loads its own mechanics from alongside it:

@/home/itec/emanuele/.agent-rules/AGENTS.md

The paper lives in `67a9ea6275d3d9785ce57026/`, a **separate git repo** with its own `AGENTS.md`. Commit there when you change manuscript text.

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

## Rules That Code Cannot Enforce

- **The paper's headline claims must land where PointStream wins.** A codec paper whose central result is "we lose to the anchor everywhere" is not a submission. Finding and naming the regime where an object-centric codec beats conventional coding is part of the work. Scope headline claims to the regime where they hold.
- **Searching for the winning configuration is the method, not a compromise.** Run the axes, observe the numbers, and locate where PointStream wins. The obligation is to report the search transparently: say which axes were tried, what each gave, and where the claim boundary lies.

## Task Completion and Setup

For dispatch, completed work, or handoff, read and follow the [session workflow](docs/workflow/session/SKILL.md). It owns PR reporting, area updates, validation, and worktree retirement; ordinary replies do not require a closeout.

Use [docs/setup.md](docs/setup.md) before environment setup or experiment runs, and its verification section before merging. Host-wide cache and import-order rules remain in the host rules above.

Subagents: the host `session` skill and the shared agents (`implementer`, `paper-editor`, `gpu-job-runner`, ...) apply; this project adds none.

Project constraints: data must stay outside the code tree (no `assets/` or `outputs/` symlinks); record the exact native encoder/decoder paths and versions with each run so comparisons are reproducible. Do not run `scripts/cleanup_merged_worktrees.sh` (`INFRA-ACT-01`: its `rm -rf` fallback can discard uncommitted work). Merged-and-clean worktrees: host `git-clean-merged-worktrees` only; ask before removing a worktree that might be a paused session.
