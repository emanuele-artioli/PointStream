# PointStream Area Index & Active Plan

State reconciled through **PR #73 merge / 74953ab; reviewed 2026-09-08**; evidence anchors are per row.
Submission Target: **ACM TOMM — 30 September 2026** (hard deadline). See [docs/roadmap.md](docs/roadmap.md) for gate criteria.

---

## 1. Area Status Index

| Area | Reconciled State to Carry Forward | Evidence Anchor / Next Action |
|---|---|---|
| [Codec](docs/areas/codec.md) | Offline canonical canvas and explicit fallback exist; lean background (VVC/SVT) & WebP appearance proposed | #36/#45/#50/#52 implementation; #64 seed; #70–#72 proposals; `CODEC-ACT-01` (connect VVC stream) |
| [Evaluation](docs/areas/evaluation.md) | 48-frame diagnostic complete; Gate A not passed; fast two-tier eval (in-memory PSNR vs full VMAF) proposed | #65/#66 plumbing; #69 result; #72 fast-eval proposal; `EVAL-ACT-01` (piped VMAF), `EVAL-ACT-02` (rate sweep) |
| [Data](docs/areas/data.md) | Diagnostic corpus exists; fresh sources provisional; long 96/192-frame sequences characterized | #56/#57 audit, #60 shortlist, #63 acquisition; `DATA-ACT-01` (manifest packaging), `DATA-ACT-04` (reserve confirmation sources) |
| [Generation](docs/areas/generation.md) | Optional; generation off in primary search; pasted-reference baseline beats generative engines | #20/#27/#28 engine roster; `GEN-ACT-01` (SAM3 evaluation, previously deferred D2) |
| [Infrastructure](docs/areas/infrastructure.md) | External data isolated; gpu6 imports fast; cleanup helper flagged for unsafe `rm -rf` | #32/#35 data root, #53 recovery, #68 cleanup audit, #73 profiling; `INFRA-ACT-01` (repair cleanup script) |
| [Paper](docs/areas/paper.md) | Separate repo; 30 pages measured (appendix 4 pages over budget); no premature winning headline | Paper `87d9e52`; `PAPER-ACT-01` (plate table), `PAPER-ACT-02` (crop table), `PAPER-ACT-03` (trim appendix) |

---

## 2. Active Assignments

| Task ID | Worktree / Branch | Base Rev | Owner | Allowed Scope | Dependencies | Active PR |
|---|---|---|---|---|---|---|
| `OVERNIGHT-GATE-A` | `/tmp/pointstream-gate-a-overnight` / `antigravity/gate-a-overnight-2026-09-09` | `0c45b92` | Antigravity | `CODEC-ACT-01/02`, `DATA-ACT-01/04`, `EVAL-ACT-01/02/03/04` | PR #74 / `0c45b92` | Pending |

---

## 3. Session Routing

Agents receive one assigned area from the table above. Read [AGENTS.md](AGENTS.md), this file, and your assigned area document. Detailed dispatch and closeout procedures are in [docs/workflow/session/SKILL.md](docs/workflow/session/SKILL.md).
