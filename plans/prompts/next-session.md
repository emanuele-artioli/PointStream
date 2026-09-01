# Prompt — cleanup, then BP31, then wave 9

Paste below the line. Supersedes the older content of this file.

---

You are picking up PointStream, an object-centric video codec targeting ACM TOMM
on **30 September**. Three jobs, in order. The third depends on the second's
result and must not be planned before it.

**Read first:** `/home/itec/emanuele/.agent-rules/AGENTS.md` · this repo's
`AGENTS.md` — in particular the rule now at the top of *Rules that code cannot
enforce*, which is the standing direction for what this project's claims must
do.

## 1. Cleanup (small, do it first)

- **Remove the `pointstream-w8-e` worktree** and its merged branches. Everything
  in it is on `main`. Follow `AGENTS.md`: read the branch before deleting
  (`git log main..<branch>`), tag anything unmerged, never `--force` away a
  worktree with uncommitted changes. As of the last check it was clean and fully
  merged, but re-check rather than trusting this sentence.
- **Retire `plans/prompts/wave8-resume-note.md`.** It warned the wave-8 session
  that its worktrees were stale and that `make_background` had changed
  load-bearingly. Those worktrees are gone and the wave is closed, so the note is
  spent. The one fact in it worth keeping is already in `PLAN.md` §2.23 and
  guarded by `tests/runner/test_background_stream_stage.py`; confirm that before
  deleting, then delete.

## 2. BP31 — the run the paper depends on

**`plans/prompts/next-session-bp31.md` is the full prompt. Use it.** It is long
because the run has been got wrong before; do not summarise it to yourself.

In one line: all three plate levers have moved and none has been priced in a
ladder, so sweep the plate codec cheaply, then re-run the paired ladder with
panorama + cross-scene stream + the winning codec all on, over N scenes, with the
anchor given the same footage.

**The standing direction applies most sharply here.** If the ladder still shows a
gap, that is a mid-point and not a conclusion — the prompt's "If the gap has not
closed" section lists the untried axes, cheapest first, and the most promising is
that §2.20 ran on the most *static* clip of eight, which is the friendliest case
for the anchor and the worst for an object-centric codec. Report the search
honestly and scope the claim to the regime that works.

## 3. Wave 9 — plan it only after BP31 reports

Its shape depends on the answer, which is why it is not planned yet:

- **If a winning regime is found:** the remaining `PLAN.md` §7 P0 items are 4
  (the core ablation lattice, still un-run), 6 (generalization on the
  general/DAVIS profile) and 7 (Evaluation and Conclusion sections, abstract
  reconciled with what was measured). Scope the wave around those, and around
  making the winning regime's claim airtight.
- **If several axes are exhausted and none wins:** stop and talk to the user
  before writing anything up. That is a finding about the approach and it changes
  what the paper argues — it must not be discovered at submission time.

**When you do scope a wave:** give each stream files that no other stream owns.
The last wave split `BP30` across two PRs purely because one stream owned
`src/runner/stages.py` and another needed it, which is also what left a stale
worktree able to silently revert a load-bearing change. One PR per independently
revertible change; over-splitting burns the Copilot review budget.

## Housekeeping that applies throughout

- `mypy --config-file pyproject.toml` now covers `experiments/`. CI is the faster
  authority here (~3m30s against 15-25 minutes locally), and it typechecks
  `tests/` — passing paths on the command line **overrides** the config's file
  list rather than adding to it.
- Long runs detached; `conda run` swallows pytest's summary and exits 0, so use
  `--junit-xml`.
- Confirm CI green with `gh` before saying it is.
