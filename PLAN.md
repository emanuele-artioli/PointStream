# PointStream Area Index & Active Plan

State reconciled through **PR #73 merge / 74953ab; reviewed 2026-09-08**; evidence anchors are per row.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | VVC low-delay background streaming and WebP appearance confirmed on held-out content; Gate B passed | `src/contracts/frozen_procedure.py`; PR #75, #80, #83; `CODEC-ACT-03` |
| [Evaluation](docs/areas/evaluation.md) | Gate A passed; Gate B held-out confirmation passed across 1080p and 720p matches | `experiments/tier/gate_b_confirmation.py`; `outputs/gate-b-confirmation/report.json`; `EVAL-ACT-05` |
| [Data](docs/areas/data.md) | Diagnostic corpus characterized; held-out confirmation candidate matches verified and materialized | `manifests/gate_b_confirmation.json`, `assets/confirmation_raw/bp57`; `DATA-ACT-04` |
| [Generation](docs/areas/generation.md) | Optional; generation off in primary search; pasted-reference baseline beats generative engines | #20/#27/#28 engine roster; `GEN-ACT-01` (SAM3 evaluation, previously deferred D2) |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf`; quiet job monitoring implemented (#82) | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; 30 pages measured (appendix 4 pages over budget); no premature winning headline | Paper `87d9e52`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (trim appendix) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| `GATE-B-CONFIRMATION` | `/home/itec/emanuele/pointstream` / `antigravity/gate-b-confirmation` | `deceb8d` | Antigravity | `CODEC-ACT-03`, `DATA-ACT-04`, `EVAL-ACT-05` | Gate A Passed / PR #83 | Complete |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).
