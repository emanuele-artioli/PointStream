# PointStream Area Index & Active Plan

State audited **2026-09-12**, code main `4fa7cd3`; [audit](docs/history/antigravity-audit-2026-09-12.md) supersedes broad Wave 1/2 claims.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | Registered panorama candidate shows +6–7 dB geometry compensation; 94% saving unverified at matched quality; residual demand to be evaluated in CODEC-ACT-07 | PR #96 (`0bdf0ef`); `outputs/development-recovery/wave2-background-probe/`; `CODEC-ACT-07` (production integration) |
| [Evaluation](docs/areas/evaluation.md) | Useful ledger/prototypes retained; verdict, reuse and schema gaps remain after #98 | `diagnostic-pix2pix-alcaraz.json`, `diagnostic-pix2pix-federer.json`; `EVAL-ACT-10` |
| [Data](docs/areas/data.md) | Two candidate matches evaluated; six-source confirmation unmet; preserve exposure history | `manifests/gate_b_confirmation.json`, `assets/confirmation_raw/bp57`; `DATA-ACT-04` |
| [Generation](docs/areas/generation.md) | Sparse first-frame injection tested; broad superior-model claims retracted; generation-off is conservative baseline with models preserved | `outputs/development-recovery/diagnostic-pix2pix-alcaraz.json`; `GEN-ACT-09` |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf`; quiet job monitoring implemented (#82) | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; audit caveat added and build validated; competitive result remains open | Paper `55e4bc4`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (recheck page budget) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| `EVAL-ACT-11` | Fresh worktree from current main | `4fa7cd3` | Next session, unassigned | Data-derived verdicts, actual conditioning identity, result schema, checkpoint factory validation | Scope repairs to next claim; no broad rerun | Handoff |
| `CODEC-ACT-07` | Fresh worktree from current main | `4fa7cd3` | Next session, unassigned | Background removal/representation evaluation with actual geometry transport and total-codec headroom | One costed card; generation can remain off | Handoff |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).

Start with [the evaluation handoff](docs/workflow/session/evaluation-handoff.md).
The user's background-first rate–distortion–computation plan is authoritative.
Keep useful old evidence and implementations; do not replay the old Wave 2
matrix, fixed GPU assignment or training schedule. Gate A remains open; Gate B
incomplete. No experiment jobs were launched by the audit/cleanup session.
