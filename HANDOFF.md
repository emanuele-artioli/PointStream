# PointStream — next session

Trigger: the user requested a clean handoff after merging recovery and cleaning
the repositories. Updated 3 September 2026. **Do not reopen PR #53 or #54.**

PointStream is an object-centric hybrid codec targeting ACM TOMM by the hard
30 September deadline. First establish a size–quality win against AV1 and VVC;
report time throughout. No win has yet been confirmed. Evidence freezes on
20 September. Read `AGENTS.md`, `PLAN.md`, then the one assigned brief.

## Verified starting point

Code PR #52, #53 and #54 are merged; #54 reached main at `3ba0e0b`. All five
auxiliary worktrees have been removed. Use `/home/itec/emanuele/pointstream`.
Check `git status -sb`, `git log -1` and `git worktree list` before starting;
the handoff-cleanup commits follow that integration base.

The paper is a separate repository at `67a9ea6275d3d9785ce57026/`. Check its Git
state separately. Paper cleanup `a0c2d93` is pushed to Overleaf `main`.
Its PDF build uses the `tex` conda environment, not the pinned
`pointstream` environment. Historical evidence is in `plans/done/`, including
`plans/done/RESEARCH-HISTORY.md`; archived launch instructions are not current tasks.

The incomplete extraction edit was archived at pushed tag
`archive/bp46-incomplete-extraction-2026-09-03`, not merged. Do not cherry-pick it
wholesale: it removes validation and has broken syntax. No source footage,
dataset, or experiment result was deleted during cleanup.

## Immediate next task

**Cursor:** execute `plans/BP49-native-reference-pilot.md`. First verify native
checkpoint gaps on the BP47 frames in a NEW output directory. Then, only after
that operational gate, run the bounded slowest-preset AV1/VVC pilot on the same
decoded frames. Return the complete report to Codex before curves or broad E1.

**Antigravity:** BP46 confirmation-footage work and `plans/PAPER-NEXT.md` can
proceed separately. Do not alter an active experiment's implementation.

## Open decisions and limits

- Can every native checkpoint gap stay below one hour? Scene checkpoints do not
  resume a killed codec mid-bitstream. Escalate an oversized stage to Codex.
- Does a winning rate–quality regime exist? The reference pilot and curves have
  not answered that. Generation stays off during the initial search.
- Which slowest preset does the pinned VVenC binary actually support? Record
  executable path/version and driven preset, not a generic preset-list guess.
- Is there enough independent eligible confirmation footage at the chosen
  duration? The corpus remains incomplete.
- The manuscript still needs final evidence, a conclusion and page-budget work;
  clean handoff does not mean submission-ready evidence.

## Jobs and operational details

No experiment was launched by this cleanup session. Check
`ps -u emanuele -o pid,etime,args` and GitHub CI before launching anything; do not
stop another user's job. Set `PYTHONPATH` to the actual checkout and keep caches
on local `/tmp`. `.ps-data-root` resolves assets/results outside the code tree.

Landmarks: `PLAN.md`, `plans/ROADMAP.md`, `plans/SESSION-REPORT.md`,
`plans/BP49-native-reference-pilot.md`, `plans/PAPER-NEXT.md`, `DATA.md`, and the
paper's own `AGENTS.md`. Build/test details and remaining structural issues are
in `plans/done/BP50-handoff-cleanup.md`.
