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
| Integration | BP51/BP52 reviewed; confirmation and batch-stop repairs included in PRs #57/#58 | Keep one current main and clearly bounded sessions |
| Background | Offline canonical canvas per compatible context, adjusted transforms and reference reuse | Long-scene rate–quality evidence; no causal canvas construction |
| Recovery | BP49/BP52 native short-pair checkpoint budget passed; batch stops on alarms/control mismatch | Longer-run budget still unverified; no mid-codec resume |
| References | Same-frame AV1 QP63 and VVC QP63/51/39 diagnostic points; legal-neighbour QPs and resume | Frozen-regime curves; separate semantic encoder/client timing |
| Fallback | Explicit conventional route with route-byte accounting | Automatic end-to-end mixed-scene scheduler is not validated |
| Data | All seven current videos are development/training-used; manifest accepts zero confirmation matches | Acquire at least six genuinely fresh audited matches |
| Generators | No confirmed improvement over the pasted-reference control | Training parked until background and lean payload can win |
| Evidence | No confirmed first-domain rate–quality win | Gate A, independent confirmation, core ablations, learned-codec comparison |
| Paper | Separate Overleaf repo; build and current 21+5-page budget checked; BP51/BP52 evidence notes | Final curves, supported headline claim, conclusion and rebalance as results land |

BP52's three CRF points have balanced coded ledgers, fresh metric controls and
no recorded alarms. CRF51 reproduced BP49; coarser backgrounds saved bytes but
lost quality, with no winning point established. This is one diagnostic scene
pair, not confirmation. Separate semantic encode/decode time is still missing:
runner wall time includes reconstruction and scoring. Historical outputs stay
immutable; code changes require a new measurement identity/output directory.

## Next work

1. **Cursor — `plans/BP53-background-transport-scale.md`:** transport a
   half-resolution background, restore the canonical size before warping,
   verify geometry/recovery, then the explicitly bounded diagnostic batch.
2. **Antigravity — `plans/BP54-fresh-confirmation-sources.md`:** verified fresh
   match/source shortlist and acquisition plan; no downloads or annotation yet.
3. **Codex:** review BP53 geometry, charged metadata and decoder symmetry;
   approve any broader curve/longer-context search only after its report.
   BP32–33 remain the cost/long-span follow-up, not disproved by this short pair.
4. **Codex:** retain ownership of the timing contract: encoder computation,
   client decode/reconstruction and evaluation scoring need separate clocks.
   No speed claim or full publication-ready comparison until instrumented.
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
