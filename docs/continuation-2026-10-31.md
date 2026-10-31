# Continuation, 2026-10-31: W7 closed; work moves to Codex

**Later same-day audit:** W7's historical closure below is superseded. The
reference-bank billing code counted encoding variants as independent crops,
undercharging the modeled bank by four. The corrected saved-evidence Tier 1
LPIPS gap is +0.0965 [0.0349, 0.1581], DISTS +0.0810 [0.0529, 0.1091],
conditional on resolved anchors. Tier 2 LPIPS is +0.0671 [−0.0011, 0.1354]
and DISTS +0.0667 [0.0440, 0.0894]: the original two-metric rule does not pass.
Several SVT curves need higher-rate points; these conditional comparisons are
optimistic while families are unresolved. The unseen-match result is
inconclusive. See [W9](experiments/w9-adapter-break-even.md) and the current
PLAN ledger. Original results are preserved. No new GPU job was submitted.

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

## Later Codex continuation: corrected accounting and approved anchor repair

The billing fix and reproduced corrected evidence are on `codex/paper-continuation-2026-10-31`, PR #315, commits `002c89cc`, `4ce348a6`, `2da6447d`. The manuscript has been fast-forward pushed to Overleaf main as `9a4491f4b33173fcaaf6fc47dbb5627f9656a8d7`; it states the conditional familiar-match result and inconclusive unseen-match result, not submission readiness. The reviewed working PDF is `/private/tmp/pointstream-w7-manuscript/build/main.pdf` (23 pages). See `docs/paper/readiness-2026-10-31.md`.

The user subsequently approved a **new independent anchor-repair budget of 5,400 GPU-seconds and 5,400 workload CPU-seconds**, with pilots and cost stops, no training. This restores nine comparisons lost at corrected rates; original missing DCVC/ROI comparisons are explicitly outside this repair. No GPU job had been submitted and none of this new budget had been spent when approved. Runtime enforcement, frozen hashes/specifications and existing-job/host preflight precede dispatch. The W9 experiment entry owns IDs, budget accounting and results. Do not use old closed budget balances. Full fleet status through gpu2 timed out after 120 seconds; this is a management-path limitation, not evidence that existing jobs stopped or the fleet is unavailable.

The separate W6b CPU timing pilot uses session `5929` on gpu2, fresh directory `/local/users/emanuele/pointstream/w6b-metric-pilot2-20261031/`, with a 20-minute timeout; inspect its recorded state in `docs/experiments/w6-motion-pose.md`, do not replay it. It has no GPU authority. Its timing data are not smoothing-quality evidence.
