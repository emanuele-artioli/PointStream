# PointStream — Gate-A experiment handoff

Trigger: BP56/BP57 review and integration formed a clean session boundary, and
the user requested a fresh Codex session for the decisive next steps. The TOMM
deadline is 30 September 2026; evidence freezes 20 September. There is no
resource outage and no PointStream job is running.

## Overall task

PointStream is an offline/buffered object-centric hybrid video codec. The paper
must find a named tennis regime where its rate--quality curve beats both AV1 and
VVC, report computation time, confirm the frozen result independently, explain
it with core ablations, and submit by 30 September. Read `AGENTS.md`, `PLAN.md`,
`plans/SUBMISSION-READINESS-2026-09-05.md`, then `plans/ROADMAP.md`.

## Current state, verified 5 September

- Code repo: `/home/itec/emanuele/pointstream`, `main`; PRs #63/#64 merged at
  `60a18f7`. This handoff/status audit is the only later local change until its
  commit is recorded.
- BP56: merged and CI-green at `ee17a8a`. Preserve
  `outputs/bp56-background-effort/`. `good/cpu-used=4/CRF63` strictly dominates
  the realtime CRF51 PointStream control on one development pair. No current
  AV1/VVC ranking or generalized win is licensed.
- BP57: merged and CI-green at `cdd3e95`. Two provisional external sources and
  seven visually checked long shots exist. They are not accepted confirmation
  data and cannot confirm a native-4K-specific claim.
- Paper repo: `67a9ea6275d3d9785ce57026/`, independent `main` at `f3421c8`. The verified build has 27 pages total: 22 through references plus 5 appendix pages.
- Merged worktrees remain at `/home/itec/emanuele/pointstream-bp56` and
  `/home/itec/emanuele/pointstream-bp57`. Ask the user before removing them;
  branch deletion is the user's operation.

## Running or queued

No experiment or download was launched. Before GPU work, check
`ps -u emanuele -o pid,etime,args` and `nvidia-smi`; do not kill unknown jobs.
BP56 full points took about 66--72 minutes, mostly because of scoring, so longer
runs need a revised explicit budget.

## Open decisions

1. Can Gate A yield negative BD-rate or low-rate-boundary dominance against both
   anchors by 10 September?
2. Can semantic encode/decode timing be separated without changing coded output
   or delaying Gate A?
3. If full-frame VMAF remains negative, activate the already declared
   salient-object-quality thesis after 10 September?
4. Keep Gate B at at least six independent videos? If so, at least four more
   fresh matches are needed beyond BP57.

## Immediate next steps

1. Verify repo states and rebuild the paper.
2. Write one bounded Gate-A brief: freeze development hashes, 48/96/192/384-frame
   contexts, coherent PointStream rate settings seeded by BP56 CRF63,
   slowest-preset AV1/VVC segmented and continuous controls, metric bounds/nulls,
   timing fields, checkpoints, stop rules and total budget.
3. Review it at Codex level, then dispatch routine implementation/run work to
   Cursor. Do not launch a broad grid first.
4. Draft a separate BP57 extraction/annotation/eligibility brief and obtain
   authority before GPU annotation or more acquisition.
5. Adjudicate Gate A by 10 September. Pass: freeze and confirm. Fail: activate
   the predeclared salient-object-quality thesis.

## Landmarks

- `PLAN.md`, `plans/SUBMISSION-READINESS-2026-09-05.md`
- `plans/ROADMAP.md`, `plans/SESSION-REPORT.md`
- `plans/BP56-background-effort-report.md`
- `plans/BP57-acquisition-report.md`, `manifests/bp57-acquisition-pilot.json`
- `plans/BP55-timing-boundaries.md`
- `67a9ea6275d3d9785ce57026/AGENTS.md`
- `67a9ea6275d3d9785ce57026/sections/evaluation.tex`
