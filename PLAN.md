# PointStream Area Index & Active Plan

State reconciled through **PR #85 / d4252b5; audited 2026-09-09**; evidence anchors are per row.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | Residual transport repaired (full-range $[-255, 255]$); wire ledger reconciled; client scoring source-free; H0 high-fidelity verified | `3d05ce7`; `CODEC-ACT-03`, `CODEC-ACT-05` complete; next: `CODEC-ACT-04` (coded fallback verification) |
| [Evaluation](docs/areas/evaluation.md) | Fail-closed protocol enforced; resolution-adaptive ladders & extrapolation prohibition added; development pilot completed. Gate A open, Gate B incomplete | `4b170da`; `outputs/development-recovery/pilot-frozen/report.json`; `EVAL-ACT-06`, `EVAL-ACT-07` complete |
| [Data](docs/areas/data.md) | Development recovery set evaluated; confirmation holdouts preserved untouched | `manifests/development_recovery.json`, `assets/confirmation_raw/bp57`; `DATA-ACT-04` |
| [Generation](docs/areas/generation.md) | Candidate inventory audited; native temporal smoke verified; evaluator restored; bounded training pilot run; residual guarantee confirmed | `ed1cb3c`; `outputs/development-recovery/diagnostic_matrix.json`; `GEN-ACT-04` complete |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf`; quiet job monitoring implemented (#82) | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; audit caveat added and build validated; competitive result remains open | Paper `55e4bc4`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (recheck page budget) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| RECOVERY-01 | `/tmp/pointstream-recovery` / `antigravity/overnight-recovery` | `da27752` | Coordinator | Overnight recovery workflow integration & verification | Workers A, B, C merged | Active PR pending |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).

Next dispatch: [eight-hour residual/evaluation/generator recovery](docs/workflow/session/overnight-recovery.md). Prepared for user launch; no overnight jobs started by this documentation change.
