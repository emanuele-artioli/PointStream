# PointStream Area Index & Active Plan

State reconciled **2026-09-11**, main `914acaa` (after Wave 1 merge of PR #93, #95, #96); evidence anchors are per row.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | Wave 1 probe (`CODEC-ACT-06`) proved registered panorama overcomes still deficit (+7 dB) for 32–120 kB, saving 94% vs legacy plate | PR #96 (`0bdf0ef`); `outputs/development-recovery/wave2-background-probe/`; `CODEC-ACT-07` (production integration) |
| [Evaluation](docs/areas/evaluation.md) | Client validity merged (#93); byte diagnosis merged (#95); Wave 2 diagnostic matrix passed across both scenes | `diagnostic-pix2pix-alcaraz.json`, `diagnostic-pix2pix-federer.json`; `EVAL-ACT-10` |
| [Data](docs/areas/data.md) | Two candidate matches evaluated; six-source confirmation unmet; preserve exposure history | `manifests/gate_b_confirmation.json`, `assets/confirmation_raw/bp57`; `DATA-ACT-04` |
| [Generation](docs/areas/generation.md) | Fail-closed client identity and dynamic start frames enforced; Wave 2 diagnostic matrix verified pix2pix altered pixels and reconciled wire | `outputs/development-recovery/diagnostic-pix2pix-alcaraz.json`; `GEN-ACT-09` |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf`; quiet job monitoring implemented (#82) | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; audit caveat added and build validated; competitive result remains open | Paper `55e4bc4`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (recheck page budget) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| `CODEC-ACT-07` | Dedicated worktree from current main | `914acaa` | Coordinator | Production integration of compact registered panorama background strategy | Wave 1 complete | Wave 2 dispatch |
| `EVAL-ACT-10` | Dedicated worktree from current main | `914acaa` | Coordinator | Wave 2 diagnostic matrix and overlap ladder execution on development scenes | `CODEC-ACT-07` | Wave 2 dispatch |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).

Wave 1 complete across all three lanes:
- Lane A: Reconstructed client and diagnostic identity verified; PR #93 merged.
- Lane B: Disjoint byte ledger reconciled; background rate floor isolated; PR #95 merged.
- Lane C: Background representation probed; registered panorama promoted; PR #96 merged.

Wave 2 dispatch: [Wave 2 pilot plan](docs/workflow/session/wave2-pilot-plan.md).
GPU 1 (`CUDA_VISIBLE_DEVICES=1`) is assigned for Wave 2 pilots. GPU 0 is reserved.
Follow the [hypothesis-driven policy](docs/workflow/experiment-design.md).
