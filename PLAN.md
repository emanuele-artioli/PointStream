# PointStream — current plan

Updated 3 September 2026. **ACM TOMM submission: 30 September, hard deadline.**
Evidence freeze: 20 September. Read this, `AGENTS.md`, and one assigned brief.
`plans/ROADMAP.md` defines the submission gates and ordering.

## Objective

Find and confirm a size–quality win against both AV1 and VVC in a named tennis
regime. Always measure and report computation time, but accept slow computation
during the search; optimize speed after confirming a win. The core design is
scene selection, reusable background, compact object appearance/motion,
receiver reconstruction, optional correction and conventional fallback.
It offers a compression opportunity, not a guaranteed win by construction.

## Current state

| Area | Verified | Remaining |
|---|---|---|
| Integration | PR #52 integrated BP44–BP46; #53 added repaired recovery; #54 archived old plans | Keep one current main and clearly bounded sessions |
| Background | Offline canonical canvas per compatible context, adjusted transforms and reference reuse | Long-scene rate–quality evidence; no causal canvas construction |
| Recovery | Identity checks, durable preparation/scene snapshots, quality preservation and cumulative timing; interrupted-run tests pass | Native checkpoint-gap verification; no mid-codec resume |
| References | Metric direction/span checks and low-rate machinery | Same-frame slowest-preset AV1/VVC pilot and curves |
| Fallback | Explicit conventional route with route-byte accounting | Automatic end-to-end mixed-scene scheduler is not validated |
| Data | Diagnostic tennis inputs available | Confirmation corpus remains incomplete |
| Generators | No confirmed improvement over the pasted-reference control | Training parked until background and lean payload can win |
| Evidence | No confirmed first-domain rate–quality win | Gate A, independent confirmation, core ablations, learned-codec comparison |
| Paper | Separate Overleaf Git repo; PDF build works | Final curves, supported headline claim, conclusion and page-budget reduction |

The historical native preflight is not E1 evidence. Engineering smoke tests do
not establish a codec win or compliance with the hourly native checkpoint budget.

## Next work

1. **Cursor — `plans/BP49-native-reference-pilot.md`:** native `bg-crf51`
   recovery-budget rerun, then a bounded slowest-preset AV1/VVC pilot on the
   same decoded source hashes. Use new output directories.
2. **Codex:** review the complete report and operational limits before broader E1.
3. **Cursor — BP45:** staged low-rate/duration search after that approval.
   BP32–33 investigate byte costs and long-span scaling; a short-span diagnostic
   does not settle the long-scene hypothesis.
4. **Antigravity — BP46:** audit and complete independent confirmation footage.
   Historically used diagnostic videos are not held-out confirmation data.
5. **Antigravity — `plans/PAPER-NEXT.md`:** manuscript scope, provenance and
   page budget; **Codex** reviews delicate claims and negative findings.
6. After a candidate win: freeze the regime, confirm it, then core ablations,
   a credible learned-video-codec comparison and an independent second domain.

All sessions return `plans/SESSION-REPORT.md` fields. Codex (or Claude if used
again) handles high-level analysis and delicate integration; Cursor and VS Code
with Antigravity handle routine, bounded work. No broad batch is authorized by
this cleanup. Preserve all search outcomes and failures.

## History and working layout

The former long PLAN is `plans/done/RESEARCH-HISTORY.md`. Old citations to PLAN
sections 2–8 refer to that historical record, not this current plan. Archived
briefs retain their original validity warnings; moving them does not make a
superseded result citable. `plans/README.md` indexes current and deferred work.

Use `/home/itec/emanuele/pointstream`, not removed worktrees. The paper at
`67a9ea6275d3d9785ce57026/` has independent Git history. The incomplete extraction
edit is preserved, not merged, at tag
`archive/bp46-incomplete-extraction-2026-09-03` (`c8898e6`).

`experiments/` is tracked runnable code; `outputs/` is generated data under the
configured data root. Keep them separate; see `DATA.md`. Codex uses `AGENTS.md`;
no project `.codex/` is needed without project-specific settings.
