# Card: K6b (players) 6, Wan-Animate

Effort: high (an environment audit for a 14B model, a new oracle rule).
**Start only after K6b players 5b has reported and the user approves.**
Paste everything below the line.

---

You are an **item session** for PointStream (repo
`emanuele-artioli/PointStream`), running in Codex on the server checkout, so
the fleet is reachable. Report in your item file and your last reply.

## Read first
AGENTS.md (Environments, Fleet, First run of a model on a GPU class),
`docs/sessions.md` (including "Codex"), `docs/resources.md#environments`,
`docs/fleet.md` (model–GPU table), PLAN.md item K6b, and
`docs/experiments/k6b-players.md` (steps 2–5, its "Not in this step"
note on step 6, and 5b's analysis), `docs/experiments/h3e.md` (Wan2.1-14B
on hands: 27 min per 81 frames on Ada, lost to scene drift).

Branch: `codex/k6b-players-6` from `origin/main`.

## Background (for the item file and the user)
K6b tests whether tennis players sent as coded keypoints (2–3 kbps) and
redrawn by a generator beat pixel codecs at equal quality on K6's windows
(bar: 75% of window-metrics, LPIPS and DISTS on the players).
AnimateAnyone reached 67% in steps 3–5 (fitted per window, per match, with
a tennis base); the same two windows lose. Step 6 tries a newer generator,
Wan-Animate (Wan2.2, 14B, open weights; arXiv 2509.14055). Like step 2 it is
conditioned on the same unbilled reference crop of each player and the coded
keypoints: "zero-shot" means no fitting, not no appearance. Speed is
secondary while quality is what is being tested (user, 2026-10-24).

## Goal
Decide step 6's zero-shot rung: does Wan-Animate, with one unbilled
reference per player, the coded keypoints and R granted as matte (as steps
2–5), beat K6's envelope at the keypoint rate on ≥ 75% of window-metrics?
Use the windows and any scope rule the user fixed after 5b, stated before
any score; otherwise the same 6 windows as steps 2–5.

## Steps
1. Environment audit (CPU): Wan-Animate's code and weights, their pins
   against the existing environments (`pointstream`, `pointstream-jcv`),
   memory on 48 GB cards; the fewest environments, each behind a narrow
   worker; lock file and reason recorded. Weights into `Models`, never the
   checkout.
2. First-run smoke per GPU class (device, attention kernels), into the
   model–GPU table.
3. The rule, hypothesis, competing explanation and budget in a new entry at
   the end of `docs/experiments/k6b-players.md`, before the scored run.
   Budget: set after the audit; ask the user if it exceeds 6 GPU-hours.
4. The run, one window per stage (≤ 45 minutes each), every finished
   window saved.

## Scope and budget
Out: fitting (the overfit and match rungs get their own rules after this
one, with the user's approval), hands, editing PLAN.md's "Where PointStream
stands" or the ledger. Stop when the rule decides, the budget is spent, a
decision belongs to the user, or you run out of ideas (for example the
model does not fit the cards).

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
of your last reply. Then tell the
user you are ready to be archived once the pull request has merged.
