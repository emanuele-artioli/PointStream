# Card: W7 Tier 1 — the five remaining windows, batched by stage

Model: Sonnet, effort high. Runs locally on the Mac, in `~/Desktop/PointStream`, where
`scripts/ps-fleet` reaches gpu1–gpu6. Root session (report address):
`session_01MbpYHe92aChsjy7MqXQ6jZ` (cloud; it receives `SendMessage` but cannot reply,
so questions go to the user in this session).

## Goal
Run W7 Tier 1's other five windows (al034, al043, med000, med024, med019) through the
whole chain, and record every stage's outputs **as written** in
`docs/experiments/w7-untouched.md`: hybrid quality and rates (judge), SVT anchors at
the hybrid's rate (W1 scoring), and DCVC at the hybrid's rate (`w5_dcvc`). W7 tests
whether PointStream's player-region win (paper's foreground-first claim) holds on
footage no experiment has used. **Do not compute the W7 decision rule or interpret
the quality numbers:** the root session does that from your records.

## Read first
AGENTS.md; docs/fleet.md (submitting, waiting, validators); in
`docs/experiments/w7-untouched.md` the rule entry (top), the first-window entries (how each
stage ran, every failure and its fix), and the last three entries (2026-10-31: batch
specs, operator order, anchor scope).

## How
- Plan: `python -m experiments.tennis.w7_specs plan docs/experiments/w7-specs/windows.json`.
  Specs: `python -m experiments.tennis.w7_specs batch STAGE docs/experiments/w7-specs/windows.json OUT --windows A,B ... NAME=PATH:SHA`.
  The operator order and the upstream archive names are in the 2026-10-31 batch entry.
  Upstream hashes come from `sha256sum` of the upstream job's `full/published.tar` on the server.
- Before submitting: `scripts/ps-fleet status` and `inspect` on all hosts (other sessions
  share the pool). Run independent stages in parallel across hosts (the five generation
  jobs, rgb diagnostic with the anchors chain).
- Record each job ID in the experiments entry **when you submit it**, and its outcome (host,
  smoke + full GPU seconds from `execution.json`, archive sha256, validator checks) when it
  ends. Keep a running GPU ledger line.
- Wait only with `scripts/ps-fleet wait` (≤ 45 min per wait).
- A failed job is never replayed: fix the cause (code, validator or spec), add a test, commit,
  and submit a new job. Same rule as the first window. Fixes go into code with tests
  (`ruff check`, `mypy --config-file pyproject.toml`, `python -m pytest`).
- W1 encodes: `--target` = the judge's scale-4 KPS2 `total_kbps` for that window. DCVC:
  `--bracket W=KPS2,KPS1` totals of the window's hybrid. Record the chosen CRFs and DCVC
  points (rate only) before the scoring job runs.
- First in-fleet runs to watch: DCVC `--bracket` (never ran on a GPU), multi-window smokes of
  rgb, receiver, judge and native base, and Med windows' times (50 fps, untimed). After the
  first Med anchors and generation jobs, rerun `plan` with the measured Med factor and
  record the revised total.

## Scope, budget and stop conditions
- In scope: the five windows, the stages in `plan` (W1 SVT anchors: plain half, plain quarter
  and half +20 outside R only). Out: Tier 2, any refit, any change to the W5 recipe, the
  first window (done), any new anchor family.
- **GPU budget: 22,783 GPU-s (6.33 GPU-h)**, the remainder of the 8 GPU-h Tier 1 card,
  failed smokes and retries included, unless the user raises it in this session. The plan
  projects 20,246–22,940 s, so there is almost no slack.
- Stop and report when: every window has judge, W1 scoring and DCVC outputs recorded
  (goal); the next job would push the ledger past the budget (stop **before** submitting it,
  so finished windows stay complete); a decision belongs to the user (scope, budget); or a
  failure you cannot fix within scope.
- Prefer finishing whole windows over half-finished stages when the budget is tight: finish
  the Al windows' remaining stages before starting a Med window's generation.

## Branch and finish
Branch `claude/w7-tier1-batch` from `origin/main`. Edit only `docs/experiments/w7-untouched.md`
(append entries at its end) and the code you fix. Commit as you go. At a stop condition,
open the pull request with auto-merge (AGENTS.md) and send the report
([docs/sessions.md](../sessions.md#the-report)) to the root session with `SendMessage`.
If it can't be reached, end your last reply with the report so the user can paste it there.
