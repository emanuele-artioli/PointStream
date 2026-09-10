# PointStream Area Index & Active Plan

State reconciled through **main 6b04eae / PR #89; open PR #88 audited 2026-09-10**; evidence anchors are per row.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | VVC/WebP implementation exists; competitive confirmation not established | `src/contracts/frozen_procedure.py`; PR #75, #80, #83; `CODEC-ACT-03`, `CODEC-ACT-05` (residual fidelity) |
| [Evaluation](docs/areas/evaluation.md) | Gate A reopened; Gate B pass retracted; two-source pilot needs instrument/protocol repair | `experiments/tier/gate_b_confirmation.py`; `outputs/gate-b-confirmation/report.json`; `EVAL-ACT-06` (integrity), `EVAL-ACT-07` (fair anchors) |
| [Data](docs/areas/data.md) | Two candidate matches evaluated; six-source confirmation unmet; preserve exposure history | `manifests/gate_b_confirmation.json`, `assets/confirmation_raw/bp57`; `DATA-ACT-04` |
| [Generation](docs/areas/generation.md) | No generator in audited sweeps; bounded readiness/training pilot permitted before baseline parity | #20/#27/#28 engine roster; `GEN-ACT-05` (repair evaluator), `GEN-ACT-06` (staged search) |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf`; quiet job monitoring implemented (#82) | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; audit caveat added and build validated; competitive result remains open | Paper `55e4bc4`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (recheck page budget) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| — | PR #88 open: Antigravity recovery awaits audited repairs; next dispatch prepared. | — | — | — | — | — |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).

Next dispatch: [Antigravity repair and submission search](docs/workflow/session/submission-search.md). PR #88 remains open with CI and evidence blockers; preserve its implementation progress, repair before merging or ranking. No training or new codec experiments launched by this audit.
