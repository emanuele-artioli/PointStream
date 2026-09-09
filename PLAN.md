# PointStream Area Index & Active Plan

State reconciled through **PR #73 merge / 74953ab; reviewed 2026-09-08**; evidence anchors are per row.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | VVC low-delay background streaming and WebP appearance implemented; C0–C3 rate ladder proven | PR #75, #80 (`CODEC-ACT-01`, `02` completed); `CODEC-ACT-03` (payload ledger) |
| [Evaluation](docs/areas/evaluation.md) | Gate A 192-frame evaluation complete; winning regime established below AV1 floor and beating VVC low-rate collapse | PR #79, #81, outputs/gate-a-vvc-webp-n96-run2; `EVAL-ACT-01/02/03` completed; `EVAL-ACT-04` (anchor presets) |
| [Data](docs/areas/data.md) | Diagnostic corpus exists; fresh sources provisional; long 96/192-frame sequences characterized | #56/#57 audit, #60 shortlist, #63 acquisition; `DATA-ACT-01` (manifest packaging), `DATA-ACT-04` (reserve confirmation sources) |
| [Generation](docs/areas/generation.md) | Optional; generation off in primary search; pasted-reference baseline beats generative engines | #20/#27/#28 engine roster; `GEN-ACT-01` (SAM3 evaluation, previously deferred D2) |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf`; quiet job monitoring implemented (#82) | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; 30 pages measured (appendix 4 pages over budget); no premature winning headline | Paper `87d9e52`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (trim appendix) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| `OVERNIGHT-GATE-A` | `/tmp/pointstream-gate-a-overnight` / `antigravity/gate-a-overnight-run-2` | `5848423` | Antigravity | `CODEC-ACT-01/02`, `DATA-ACT-01/04`, `EVAL-ACT-01/02/03/04` | PR #80 / `5848423` | Completed / Under PR |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).
