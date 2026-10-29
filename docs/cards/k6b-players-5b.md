# Card: K6b (players) 5b, why two tennis windows lose

Effort: high (reading an ambiguous result, proposing a scope rule).
Paste everything below the line.

---

You are an **item session** for PointStream (repo
`emanuele-artioli/PointStream`), running in Codex on the server checkout, so
the fleet and `~/pointstream-data` are reachable. Report in your
item file and your last reply.

## Read first
AGENTS.md, `docs/sessions.md` (including "Codex"), PLAN.md's tennis bullet
in "Where PointStream stands" and item K6b, and `docs/experiments/k6b-players.md`
(steps 2–5) and `docs/experiments/k6.md`. Step 5's outcome is on branch
`origin/claude/k6b-players-zeroshot` (commit `d0a00e5`) and may not be on
main yet: merge that branch into yours if `k6b-players.md` lacks the step 5
outcome.

Branch: `codex/k6b-players-5b` from `origin/main`.

## Background (for the item file and the user)
K6b tests whether tennis players can be sent as coded keypoints and redrawn
by a generator (AnimateAnyone) more cheaply than pixel codecs (SVT-AV1,
DCVC-UF) at equal quality. Judged on K6's windows (240-frame clips from five
broadcast matches), a window is won on a metric (LPIPS, DISTS on the
players) when the generator beats the codecs' envelope at the keypoint rate;
the bar is 75% of window-metrics. Steps 3, 4 and 5 all reach 67% on six
windows, and the same two windows lose every time:
- `rublev_auger-aliassime/052`: judged on only 32 of its 240 frames (the
  frames K8b, the court registration, recognises as court view); the rest is
  a close-up. Highest keypoint rate (9.8 kbps).
- `alcaraz_hurkacz/045`: small players; lost by 0.003 LPIPS in step 3 and
  0.03 in step 5.
The user's view (2026-10-24): step 5 did well apart from these two; if the
problem is in them, PointStream may work well within a scope that excludes
such windows.

## Goal
Explain, with measurements, why 045 and 052 lose, and say whether a
**content-blind scope rule** exists: one an encoder can compute before
coding, from the video alone (for example: the share of a window's frames
registered as court view, the players' height, the share of frames where
both players are fitted), that separates windows where the generator route
is expected to win from windows that go to the pixel coder.

## Steps (CPU only)
1. Write the question, the measurements and how a scope rule would be
   chosen in a new entry at the end of `docs/experiments/k6b-players.md`,
   before reading any new number.
2. Per window, for all 8 of K6's windows (and the development windows
   008 and 033): registered-frame share, player heights, the pose
   estimator's miss rate, camera motion and cuts, keypoint rate, the
   envelope's strength at that rate (its LPIPS/DISTS), the reference crop's
   distance to the judged frames, and steps 2–5's per-window margins.
   Read saved reports and inputs only.
   Check first whether 052 crosses a shot cut (court view to close-up):
   K4 chose windows by time, not by shot, so a window can straddle a cut.
   If it does, test the obvious content-blind rule, "a window is one shot"
   (shot-boundary detection is standard at an encoder), and find which
   frames the generator's reference and the judged frames come from.
3. Say which measurement explains the losses, if any. If a rule follows,
   state it with its threshold, and show the score of steps 2–5 under it
   **beside** the pooled score, never instead of it.

## Honesty rules (the user decides any claim)
- A scope rule found on these windows is a hypothesis, not a result: the
  windows that suggested it cannot confirm it. Say what fresh windows would
  confirm it (court-view windows from the archive not used to train or
  choose anything in steps 3–5; list what exists) and what a confirmation
  run would cost. Do not run it.
- Step 5's verdict stands as recorded. Nothing here re-scores it as a pass.
- Any scope must be applied to every method on the same windows,
  including the pixel anchors.

## Scope and budget
In: the analysis above, a small script and a focused test if code is needed.
Out: new GPU runs, new fits, changing earlier entries, editing other items' text in PLAN.md. GPU: none. Stop when the goal is
reached, a decision belongs to the user, or you run out of ideas.

## Ledger
PLAN.md's dispatch ledger is the single record of open work. When you start,
set your row's status to `running <date>` and push that commit at once. When
you stop, set it to `reported <date>: <one-line outcome>; <GPU-hours used> of
<budget>`, and update your item's bullet in "Where PointStream stands" in one
or two lines. Edit only your own row and bullet.

## Report
Run `ruff check`, `mypy --config-file pyproject.toml`, `python -m pytest`.
Commit, push, open the pull request with auto-merge. Put the report (format
in `docs/sessions.md`, "The report") at the end of your entry and at the end
of your last reply. Include: which
measurement explains each loss, the proposed scope rule (or none), steps
2–5's scores under it beside the pooled ones, and the confirmation run's
windows and cost. Then tell the user you are ready to be archived once the
pull request has merged.
