# PointStream — current plan

Updated 7 September 2026. **ACM TOMM submission: 30 September, hard deadline.**
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
| Integration | PRs #65/#66 merged at `606cf53`; Gate A preflight passed | Launch bounded 48-frame native run |
| Background | Offline canonical canvas per compatible context, adjusted transforms and reference reuse | Long-scene rate–quality evidence; no causal canvas construction |
| Recovery | Checkpoint budget, heartbeat, 55-min subprocess timeout and retry limits enforced | Longer-run budget verification; no mid-codec resume |
| References | SVT-AV1 preset 0 and ffmpeg/libvvenc `slower` probe verified | Native curve generation under continuous and segmented access |
| Fallback | Explicit conventional route with route-byte accounting | Gate 2 fallback control in native 48-frame run |
| Data | Two fresh provisional sources acquired; seven sampled long shots visually checked | Validate them and add enough independent matches for Gate B |
| Generators | No confirmed improvement over the pasted-reference control | Training parked until background and lean payload can win |
| Evidence | Gate A dry-run preflight complete (`identity: 9b63be50…`) | Gate A native run, confirmation, core ablations, DCVC-RT |
| Paper | Separate Overleaf repo; 27 pages (22 main + 5 appendix), within budget | Final curves, supported headline claim, timing and confirmation |

PR #65 and #66 merged at `606cf53`. Gate A implementation identity, disjoint timing
boundaries (`encoder_seconds`, `client_seconds`, `evaluation_seconds`), tool floor
probes (SVT-AV1 preset 0, libvvenc `slower`), rate ladder (C0–C3), and independent
client-side raw stream decode are implemented and tested.

The final permitted 48-frame preflight passed cleanly from commit `606cf53` with
identity fingerprint `9b63be50…`. Source hashes for `alcaraz_highlights` (`scene_000`,
`scene_028`) match. No native curve was encoded in the preflight; native controls
(`object_stream_off`, `conventional_fallback`) remain pending in the bounded 48-frame run.

Completed briefs and reports (BP32, BP33, BP38, BP39, BP40, BP43, BP45, BP49, BP51–BP57)
are archived in `plans/done/`. The dispatch entry point for execution is
`plans/DISPATCH-GATE-A-48FRAME-RUN.md`.

## Next work

1. **Dispatched session (`plans/DISPATCH-GATE-A-48FRAME-RUN.md`):** launch and
   adjudicate the bounded 48-frame native run (`--frames 48 --native --authorize-native`)
   with Gate 2 controls, native slowest-preset AV1/VVC curves, and PointStream rungs C0–C3.
2. **Gate A duration progression (by 10 September):** if 48 frames demonstrates
   favorable amortization and subprocess timeout projection holds (<55 min), advance
   sequentially (96 -> 192 -> 384 frames) and fit payload slope. If no crossover by
   10 September, activate the pre-registered salient-object fallback thesis.
3. **Antigravity/Cursor:** validate and annotate BP57 candidate shots; acquire and
   audit >=4 additional independent matches for Gate B confirmation (n>=6).
4. **Antigravity — `plans/PAPER-NEXT.md`:** manuscript scope, provenance and
   page budget; **Codex** reviews delicate claims and negative findings.
5. After a candidate win: freeze the regime, confirm on Gate B, then run Gate C
   core ablations, DCVC-RT baseline, and second domain before evidence freeze on 20 September.

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
