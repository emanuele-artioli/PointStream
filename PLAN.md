# PointStream Area Index & Active Plan

State reconciled through **main 6199d3e**, Wave 2 GPU pilots 2026-09-10 (gpu5); those pilots are superseded as generator/codec evidence (see evaluation/generation). Evidence anchors are per row.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | VVC/WebP implementation exists; competitive confirmation not established | `src/contracts/frozen_procedure.py`; PR #75, #80, #83; `CODEC-ACT-03`, `CODEC-ACT-05` (residual fidelity) |
| [Evaluation](docs/areas/evaluation.md) | Gate A open; Gate B incomplete; Wave 2 residual direction worked but all codec comparisons were invalid/unscorable | `outputs/development-recovery/`; `EVAL-ACT-06`, `EVAL-ACT-07` |
| [Data](docs/areas/data.md) | Two candidate matches evaluated; six-source confirmation unmet; preserve exposure history | `manifests/gate_b_confirmation.json`, `assets/confirmation_raw/bp57`; `DATA-ACT-04` |
| [Generation](docs/areas/generation.md) | Wave 2 pix2pix matrix is invalid for generator comparison: client selected pasted placements | diagnostic-pix2pix-rq32.json; `GEN-ACT-05`, `GEN-ACT-06` |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf`; quiet job monitoring implemented (#82) | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; audit caveat added and build validated; competitive result remains open | Paper `55e4bc4`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (recheck page budget) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| `CODEC-ACT-05`, `EVAL-ACT-06`, `GEN-ACT-05` | `cursor/wave1-generation-transport-repair` | `6199d3e` | Repair invalid Wave 2 generator/transport evaluation | Explicit generation intent; compact mask wire; diagnostic provenance | Do not train until repaired evidence shows generated pixels reach the client | — |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).

Next: repair client generation (appearance must condition, not paste) and compact mask transport, then rerun bounded development pilots. The original pix2pix matrix is invalid for generator comparison. The ~41.5 MB cost is raw full-frame mask transport, not a ledger mismatch. Do not rank models or change gate status from the unrepaired pilots. Training remains unlaunched.
