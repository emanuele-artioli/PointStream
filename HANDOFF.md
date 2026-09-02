# PointStream — status and handoff, 2026-09-02

Replaces the 2026-08-23 handoff, which described a project two waves back. The
session start hook surfaces this file, so a stale one actively misleads.

Target: **ACM TOMM, 30 September.** Twenty-eight days.

## Read, in this order

1. `AGENTS.md` (project) and the host rules it imports.
2. **`plans/ROADMAP.md`** — what is left, in what order, with the dependency
   graph and file ownership.
3. **`plans/FORK-bp31.md`** — the three papers, one per outcome of the run in
   flight, written before it reported.
4. `PLAN.md` for the system and the measurements; `plans/README.md` for the
   brief index. Then **one** brief. Do not read the plan tree.

## The situation in six sentences

The platform is built, runs end to end, and is measured: sixteen component axes,
a runner, region-scoped metrics with calibration invariants, and a paired
rate-distortion ladder against conventional anchors. **PointStream loses to the
codec it is built on** — BD-rate **+90.97%** against an av1 anchor at two scenes
with the cross-scene background stream on, down from +116.8% with a single-frame
plate. The cause is not the object stream: **the plate is 88–91% of the payload
at every rung**, and PointStream's entire non-plate payload is under a third of
the anchor's total rate. **No generative engine produces a usable player** —
every one of eight loses to pasting the keyframe — so the shipped configuration
is plate + warps + pasted crops + a corrective residual, and the residual is the
one component with a measured favourable trade (+0.9% rate for +5.40 dB).
`AGENTS.md` requires the paper's headline claim to land where PointStream wins,
so **finding that regime is the work**, and one session is on it now. Everything
else divides into work that is true whatever it finds and work whose shape
depends on it.

## Who is doing what

| | |
|---|---|
| **Parallel session** | BP31, worktree `pointstream-w9-a`, branch `wave9/bp31-ladder`, **PR #45 open**. Owns `src/runner/stages.py`, `src/components/background/**`, `experiments/tier/**`, `tests/runner/**`, `plans/BP31-*`. Next: extract more cached scene windows, then a ten-scene six-video ladder. |
| **Anyone else** | wave 9 in `plans/ROADMAP.md` §2. File-disjoint from the above by construction. |

## The two things to do first, and why

Both are cheap, both are mostly arithmetic over data already on disk, and either
can change what the expensive campaign should be spent on.

- **`BP32` — the rate budget.** The motivation measured 22.9% ± 3.0% of av1's
  bitrate in the foreground and 34–69% of the background's rate recoverable by a
  plate. The system delivers +90.97%. Those are the same claim measured twice and
  they are ~150 BD-rate points apart, and nobody has written down where the
  difference goes.
- **`BP33` — span.** Every ladder in this project has run at **eight frames per
  scene**; the BP21 cache holds **forty-eight**, and `PLAN.md` §2.14's headroom
  was measured over those forty-eight. The plate is paid once per scene whatever
  the scene's length. Three separate reports already record this as "the least
  favourable amortisation a fixed plate cost can get" and none acted on it.

**Tell the BP31 session the answer before its extraction campaign commits to a
frames-per-scene value.**

## State of the tree

- `main` is green; CI on the last five pushes succeeded. `ruff check` clean.
- One PR open (#45). Three worktrees: `pointstream`, `pointstream-w9-a`, and a
  Claude scratch worktree.
- **Local `pytest` and `mypy` were unable to start** in the `pointstream` env —
  both read config out of `pyproject.toml` and Python 3.10 has no `tomllib`, so a
  missing `tomli` made them fail inside their own argument parsing. Fixed
  2026-09-02: installed, and added to the `dev` extra.
- **CI's ruff step omitted `experiments/`** (it passed explicit paths, which
  *overrides* the project's file set rather than adding to it — the same trap
  mypy hit on 2026-08-30). Fixed 2026-09-02: the step now passes no paths.
- The `pointstream` env carries two invalid distributions in `site-packages`
  (`-` and `-umpy`), residue of an interrupted numpy install. Harmless today.
- Untracked cruft that a guard would not let this session delete:
  `.pytest_cache`, `.ruff_cache`, the `__pycache__` tree, and an empty
  `src/decoder/` holding only stale bytecode. All gitignored and regenerable.

  ```bash
  rm -rf src/decoder .pytest_cache .ruff_cache && find . -name __pycache__ -type d -not -path './.git/*' -prune -exec rm -rf {} +
  ```

## Standing hazards

- **The asymmetry**: these checks get applied to disappointing results and
  skipped on exciting ones. When the news is good, add a check.
- **Bound before believing, two-sided**, written to
  `outputs/<brief>/bounds-before-run.json` before the first encode.
- **Per video with the spread.** BP30 drew two conclusions from one video and
  both inverted at five.
- **A flag existing is not a feature working.** Two config axes here reached
  nothing at all and looked fine.
