# PointStream — Gate-A execution handoff

Trigger: PR #66 merged at `606cf53`; 48-frame preflight passed cleanly with identity
`9b63be50…`. The TOMM submission deadline is 30 September 2026; evidence freezes
20 September. There is no resource outage and no PointStream job is currently running.

## Overall task

PointStream is an offline/buffered object-centric hybrid video codec. The paper
must find a named tennis regime where its rate--quality curve beats both AV1 and
VVC, report computation time, confirm the frozen result independently, explain
it with core ablations, and submit by 30 September. Read `AGENTS.md`, `PLAN.md`,
`plans/GATE-A-LONG-CONTEXT-2026-09-05.md`, and `plans/ROADMAP.md`.

## Current state, verified 7 September

- Code repo: `/home/itec/emanuele/pointstream`, `main` at `606cf53` (PRs #65 and #66 merged; CI green).
- Gate A preflight: passed from clean merge commit `606cf53` (identity: `9b63be50…`).
  - Source RGB hashes match frozen `alcaraz_highlights` (`scene_000`, `scene_028`).
  - Tools verified: SVT-AV1 preset 0 and ffmpeg/libvvenc `slower`.
  - Disjoint timing boundaries (`encoder_seconds`, `client_seconds`, `evaluation_seconds`) and client-side raw bitstream decode verified.
  - Checkpoint budget, 55-minute subprocess timeout, heartbeat, and fail-closed stop conditions verified.
- Native execution: no native curve points were run in preflight. Native controls (`object_stream_off`, `conventional_fallback`) remain pending and must pass in the bounded 48-frame run.
- Paper repo: `67a9ea6275d3d9785ce57026/`, independent `main` at `f3421c8`. Verified build has 27 pages total: 22 through references plus 5 appendix pages (within TOMM budget).
- Worktrees: `/home/itec/emanuele/pointstream-gate-a` (branch `antigravity/gate-a-long-context-2026-09-05`, merged in PR #66). Ask user before removing.

## Running or queued

No experiment is currently running. Before GPU work, check `ps -u emanuele -o pid,etime,args`
and `nvidia-smi`; do not kill unknown jobs.

## Immediate next step: 48-Frame Native Run

The turnkey prompt for dispatching the 48-frame native run and recording results is:
[`plans/DISPATCH-GATE-A-48FRAME-RUN.md`](plans/DISPATCH-GATE-A-48FRAME-RUN.md).

Pass that document to the executing session. The session will execute the bounded
run, verify Gate 2 controls, record reference and PointStream curves, and document
the session results directly into the template in that file.

## Subsequent gates

1. If 48 frames demonstrates favorable amortization, advance sequentially (96 -> 192 -> 384 frames) by 10 September.
2. If no full-frame VMAF crossover by 10 September, activate the pre-registered salient-object fallback thesis (`ROADMAP.md` §6.1).
3. Confirm frozen winner on Gate B (>=6 independent matches; BP57 + 4 new sources) by 14 September.
4. Core ablations (Gate C), DCVC-RT baseline and second domain (Gate D) by 18 September.
5. Evidence freeze 20 September; submit by 30 September.
