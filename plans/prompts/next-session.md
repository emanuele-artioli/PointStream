# Prompt — pick up a wave-9 work item

Paste below the line. Supersedes the older content of this file (the cleanup and
BP31-scoping prompt it used to hold; both are done, and `plans/BP24-ladder-report.md`
cites this filename for that history).

---

You are picking up PointStream, an object-centric video codec targeting ACM TOMM
on **30 September**.

**Read first, in order:** `/home/itec/emanuele/.agent-rules/AGENTS.md` · this
repo's `AGENTS.md` — in particular *Rules that code cannot enforce* · **`plans/ROADMAP.md`**
· then **exactly one brief** from it. Do not read the whole plan tree; a session
that needs all of it is scoped too broadly.

## Where the project stands in four sentences

PointStream is measured end to end against the codec it is built on and it
**loses**: BD-rate **+90.97%** against an av1 anchor at two scenes with the
cross-scene background stream on. The plate — one still image — is **88–91% of
the payload at every rung**, so the rate problem is not the object stream, and
the residual is the one component with a measured favourable trade (+0.9% rate
for +5.40 dB). A parallel session owns the hunt for a winning regime (BP31,
worktree `pointstream-w9-a`, PR #45) and is about to run a ten-scene, six-video
ladder. `plans/FORK-bp31.md` holds the three papers that result, one per outcome,
written before that run reports.

## Pick one

**If nobody has done them yet, `BP32` and `BP33` come first** — they are cheap,
they are mostly arithmetic over data already on disk, and either can change what
the expensive campaign should be spent on. `BP32` reconciles the ~150 BD-rate
points between the headroom the motivation measured and what the system
delivers. `BP33` is the observation that every ladder in this project has run at
eight frames per scene while the cache holds forty-eight, on a cost that is paid
once per scene.

Otherwise take any wave-9 item from `plans/ROADMAP.md` §2. They are file-disjoint
and none of them touches what PR #45 holds.

## Rules that keep costing time when they are skipped

- **Bound before believing, and two-sided.** Write the band to
  `outputs/<brief>/bounds-before-run.json` before the first encode. A result
  outside it is an alarm to investigate, not a number to report.
- **When the news is good, add a check rather than stopping.** These checks get
  applied to disappointing results and skipped on exciting ones.
- **Report per video with the spread.** BP30 drew two conclusions from one video
  and both inverted at five.
- **One thing at a time.** Span and scene count are both amortisation axes on the
  same fixed cost and a two-axis sweep run once will not separate them.
- **Drive the flag, do not read it.** This project has twice found a config axis
  that reached nothing; a null result is only readable beside a control where the
  same probe does separate the arms.

## Housekeeping

- `ruff check` (no paths — passing paths *overrides* the project's file set
  rather than adding to it, and that trap has been hit here in both ruff and
  mypy), `mypy --config-file pyproject.toml` with no paths, the tests for what
  you touched, and `python -m src.contracts.layers`.
- CI is the faster authority for mypy (~3m30s against 15–25 minutes locally).
  Confirm green with `gh run view <id> --log-failed` rather than assuming.
- `conda run` swallows pytest's summary and exits 0 — use `--junit-xml`.
- Long runs detached, checkpointing at least hourly.
- One PR per independently revertible change.
