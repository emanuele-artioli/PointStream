# Continuation, 2026-10-31: W7 closed; work moves to Codex

From here on the work runs in Codex sessions on the Mac (user, 2026-10-31). The cloud root
session is wrapped up: no job, timer or process of it remains. There is no root session now, so
each Codex session owns its own dispatch-ledger row in PLAN.md
([sessions](sessions.md#codex)).

## Where things stand
- **W7 is closed: the foreground-first claim holds on untouched footage**
  ([outcome entries](experiments/w7-untouched.md), at the end of the file).
  - Tier 1, six untouched windows of familiar matches: player gap LPIPS +0.103 [+0.041, +0.165], DISTS +0.085.
  - Tier 2, three windows of nadal_kyrgios, a match the base never saw, with its own billed adapter:
    LPIPS +0.072 [+0.005, +0.140], DISTS +0.070 [+0.048, +0.091].
  - The whole frame is worse than the best anchor in every window, and DCVC-UF is the strongest anchor.
  - Weak points to carry into the paper:
    - the Tier 2 LPIPS interval clears zero by 0.005 (n = 3);
    - the Medvedev cluster alone (n = 3) does not pass;
    - the generated players flicker (W6);
    - adapters are billed over 2 h (W9).
- **GPU state:** no W7 job is running. W7 used about 10.5 GPU-h.
- **Fleet issues open:**
  - gpu3 still runs the old fleet release, and its restart and doctor check failed on environment-probe timeouts (Codex, 2026-10-31).
  - The A6000s on gpu2 were memory-full from other users at the time.
  - DCVC-UF decoding is intermittently nondeterministic on the same bitstream. It is guarded by the two-process decode check; no fix has been tried.
- **Tools that now exist:**
  - `experiments/tennis/w7_batch.py` (`w7_specs plan|batch`): stage-batched fleet specs for manifest windows.
  - `experiments/tennis/w7_analysis.py`: the player-gap rule from result archives.
  - Both are reusable for any new window set.

## Next, in order of value to the paper
Each item starts with its rule entry in `docs/experiments/` before any GPU time (AGENTS.md, Experiments).

1. **Paper text for W7.** Put the two-tier result, its scope and the weak points above into the manuscript.
   This is the Overleaf project in AGENTS.md, which has its own AGENTS.md. No GPU.
2. **W6b: temporal smoothing of the generated crops** (CPU, saved W6 outputs).
   - Flow-guided blend of each crop with its warped predecessor, at two or three blend weights.
   - Score flicker and per-frame player LPIPS/DISTS together, since smoothing can trade sharpness for stability.
   - The rule's starting point is the last entry of `docs/experiments/w6-motion-pose.md`.
3. **W9: adapter break-even length L\*** (PLAN.md, W9).
   - Encode anchor points at higher rates (SVT half up to about 130 kbps, DCVC on Al above 33 kbps): CPU encodes plus about 0.3 GPU-h of scoring.
   - This replaces the arbitrary 2-hour amortisation in every W7 number.
   - Then the base-only hybrid at minute 0, and adapters fitted on the first 2, 5 and 10 minutes of nadal_kyrgios. The Tier 2 split is already by time, and its windows and manifest are staged.
4. **W11: racket oracle** (CPU on W6 outputs). Paste the source racket into the hybrid and measure the player-region and whole-frame gain before building a racket path.
5. **W10: stronger generator / larger base.** Run an oracle first: the new backbone zero-shot and overfit on one window, at the W1 operating point.
6. **Housekeeping:**
   - Fix gpu3's environment probe and restart its worker.
   - A 25-frame DCVC determinism repeat costs about 100 GPU-s per run, if the nondeterminism ever blocks a result.
   - The first Codex session that delegates to subagents works through the Codex validation list in `docs/sessions.md` (AGENTS.md says so).

## How Codex sessions should run (lessons from 2026-10-31)
- **Don't end a turn just because a command is running.**
  - For a fleet job, wait with `scripts/ps-fleet wait` (≤45 min).
  - For a CPU command on a server outside the fleet, check its state, then make one bounded blocking wait (≤45 min).
  - If it is still running after that, check once more before deciding.
- **Commit and open the PR as you go.** A fix that sits uncommitted in a worktree is lost with it.
- **Local test failures.** About 12–13 pytest failures and 7 mypy errors also fail on unchanged `main` on the Mac (missing packages, macOS paths). Compare with `origin/main`; CI is the check.
- **Decisions the user must make** (budget beyond a card, scope) go to the user in that session, with the context the question needs.
