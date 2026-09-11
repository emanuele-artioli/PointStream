# PointStream Area Index & Active Plan

State reconciled **2026-09-11**, main `dc4a0cd`, PR #93 reviewed at `ba5a9a8`; evidence anchors are per row.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | VVC/WebP exists; compact transport repairs in #93 await review fixes | `src/contracts/frozen_procedure.py`; PR #75, #80, #83; `CODEC-ACT-03`, `CODEC-ACT-05` (residual fidelity) |
| [Evaluation](docs/areas/evaluation.md) | Gate A open; Gate B incomplete; overlap artifacts located but identity/evidence verification false | `experiments/tier/protocol.py`; `experiments/tier/resolution_adaptive.py`; `EVAL-ACT-06`, `EVAL-ACT-07` |
| [Data](docs/areas/data.md) | Two candidate matches evaluated; six-source confirmation unmet; preserve exposure history | `manifests/gate_b_confirmation.json`, `assets/confirmation_raw/bp57`; `DATA-ACT-04` |
| [Generation](docs/areas/generation.md) | No validated generator ranking; #93 checkpoint, reuse and control checks require repair | #20/#27/#28 engine roster; `GEN-ACT-05` (repair evaluator), `GEN-ACT-06` (staged search) |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf`; quiet job monitoring implemented (#82) | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; audit caveat added and build validated; competitive result remains open | Paper `55e4bc4`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (recheck page budget) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| `EVAL-ACT-09`, `GEN-ACT-08` | Fresh branch from #93 `ba5a9a8` | `ba5a9a8` | Unassigned — lane A | Checkpoint resolution, complete reuse identity, fail-closed controls and input mapping | Before #93 merge / fresh model evidence | #93 |
| `EVAL-ACT-08` | Fresh worktree from current main | `dc4a0cd` | Unassigned — lane B | Read-only saved-run byte diagnosis; analysis helper if needed | Can start alongside A/C; no automatic diagnostic reuse | #92 dispatch |
| `CODEC-ACT-06` | Fresh worktree from current main | `dc4a0cd` | Unassigned — lane C | Bounded background component probe | Full-codec promotion needs A/B | #92 dispatch |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).

Next dispatch: [parallel evidence repair and component probes](docs/workflow/session/parallel-probes.md).
Give one lane to each Cursor/Antigravity worker, with isolated worktrees and explicit
resource ownership. Lane A repairs validity; B diagnoses saved bytes; C tests the
background mechanism. These assignments are prepared, not launched. Foreground
training, Gate B and paper edits remain off. Follow the
[hypothesis-driven policy](docs/workflow/experiment-design.md).
