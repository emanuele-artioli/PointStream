# Continuation, 2026-10-31: W7 closed; work moves to Codex

**Resume from the [2026-11-01 continuation](continuation-2026-11-01.md):** the
approved W8 coverage amendment ran two cost-stopped requests; both are terminal,
no full upper-anchor result exists, and an exclusive-GPU cap amendment is pending.
The older summaries below are retained as history, not current authority.

**Latest continuation, 2026-11-01:** user approved the separate W8 witness and
confirmed 23 pages of body/references plus five additional appendix pages.
The full PLAN/evidence audit and expanded negative-results appendix are done;
Overleaf `e366440` is 22+3 pages. W8's actual fixed-payload rate screen stopped
unresolved before any fleet/GPU submission (46.3151 kbps versus one family's
45.8233 upper bracket). Local tensor diagnostics do not establish a fresh
receiver. See [current readiness](paper/readiness-2026-11-01.md) and
[W8 outcome](experiments/w8-wire-audit.md); old pending-approval statements below
are historical. No unchanged card may be replayed from this older handoff.

**Later same-day audit:** W7's historical closure below is superseded. The
reference-bank billing code counted encoding variants as independent crops,
undercharging the modeled bank by four. The corrected saved-evidence Tier 1
LPIPS gap is +0.0965 [0.0349, 0.1581], DISTS +0.0810 [0.0529, 0.1091],
conditional on resolved anchors. Tier 2 LPIPS is +0.0671 [−0.0011, 0.1354]
and DISTS +0.0667 [0.0440, 0.0894]: the original two-metric rule does not pass.
All nine resulting SVT coverage gaps have now been repaired; Tier 2 has all
four families bracketed and remains inconclusive at two hours. Three original
Med comparisons remain missing in Tier 1. The conditional duration model passes
about 133–198 min within measured coverage, not globally. All repair jobs are
terminal (1,043/5,400 GPU-s; ≤1,924/5,400 CPU-s). See
[the completed continuation](#completed-repair-resume-here),
[W9](experiments/w9-adapter-break-even.md) and the current PLAN ledger.
Original results are preserved; no research job is running.

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

The separate W6b CPU timing pilot **completed normally**, session `5929` on gpu2, fresh directory `/local/users/emanuele/pointstream/w6b-metric-pilot2-20261031/`. Its process is absent. It used 89.82 CPU-s plus a conservative 10 CPU-s for a terminated faulty startup (99.82/1,800 total); wall 1,112.46 s. Even warmed metric timings project 2,473 CPU-s for the required 240 metric pairs before composition, flow, flicker and controls. The scale gate stops; there is no smoothing-quality result or running W6b job. See `docs/experiments/w6-motion-pose.md`; do not replay either pilot. Any new full smoothing proposal needs separate authority and a packed host-local environment.

Latest manuscript main is `fc0343f989ac038ee8d37462bcff9b47fe5a67cb`, including the corrected W7 figure and conditional caption, still 23 pages. Code checkpoint `ac2b82ba` adds an optional bounded read-only status timeout; all existing mutation/probe deadlines remain unchanged. The single extended read through gpu5 completed: 1,536 records, no pending/running requests or duplicate W9 repair; one older gpu3 attention remains unresolved and untouched. Five hosts qualify; gpu3's normal 60-second probe timed out.

First CPU pilot **`20261031T211106Z-6eb4ea04` failed smoke before encoding**: restricted source revision `0c00247c` omitted the transitive `tools` import. Its request and wait are terminal, no full stage or GPU ran. Charge 36 CPU-s conservatively, retain 100 CPU-s preparation reserve inside the 5,400 CPU-s cap, GPU 0/5,400. Preserve its diagnostics at `/private/tmp/w9-pilot-failed/` and never replay it. Before a fresh replacement, validate the complete tracked-code import closure in an isolated snapshot, preventing fallback to an editable original checkout; include `tools`. Deadline remains `2026-10-31T22:30:00Z`; W9 records every future request ID and wait handle. Parent must review CPU evidence and a frozen GPU spec before GPU dispatch. W11's read-only annotation audit found no eligible pixel mask linked to the W6 frames; the historic 3% measurement is a body-hull proxy, not independent racket recall.

Replacement **`20261031T211655Z-098385bc` completed** on gpu6 CPU, from exact frozen `0c00247c` full tracked source (clean isolated checkout `/private/tmp/pointstream-w9-source-0c00247c`). Smoke native/source/stream/affinity/search/projection gates pass; full quarter search completed in 17.04 s, total conservative campaign charge 45 CPU-s. Its full stage was not independently validated, so it is not promoted as a quality result. The new CPU profile is no longer needed for this repair: all nine missing upper-rate encodes were recovered as unscored search trials in the original W7 archives, with hashes, byte counts and native/source/ROI metadata verified. Exactly six window selection manifests and original three-control identities per window are in `docs/experiments/w9-inputs/saved-trial-scoring/`.

Current W9 conservative CPU ledger is **421/5,400 CPU-s** (including bounded audit/preflight charges), GPU **0/5,400**, no GPU request submitted. Both CPU requests are terminal; do not replay either. W9 owns immutable data manifests, specs, execution and its experiment entry; `paper_w7` implements the reviewed exact-point GPU scoring profile/tests, parent reviews before any GPU dispatch. Score only nine recovered upper points plus eighteen original controls, using the same 80 ticks and original regions; retain legacy scorer behavior. Merge verified old and new raw policy rows by stream identity before corrected-rate interpolation, never use the old cached `at_target`. Require full-output validation before scientific acceptance. PR #315 auto-merge is enabled; a figure-only mypy name collision was fixed in `1edc67ae` without changing the rendered PDF. Current deadline 22:30 UTC is recorded in W9.

### GPU repair pilot checkpoint (21:42 UTC)

PR #315 merged into main as `d14e9678`; active branch is now `codex/w7-anchor-repair-2026-10-31`, draft PR #316. Reviewed scorer/audit/data specification is committed as `9a74db87`, with a clean frozen source checkout `/private/tmp/pointstream-w9-source-9a74db87`. Its full tracked source passed isolated production-packed import validation, one CPU, GPU hidden, every project module under the frozen root. Full local tests: 1,255 passed, 27 skipped, the same 13 failures as unchanged main; Ruff passes, mypy the same seven baseline errors. No new local regression was found.

**GPU pilot `20261031T214253Z-429aabab` submitted pending**, nad000, exact source `9a74db87`, source snapshot SHA prefix `2ca2f023`, normalized spec prefix `a63ba987`, selection spec prefix `b53be31`. The W9 entry holds full identities and its one bounded wait handle. CPU ledger **541/5,400 CPU-s**, GPU **900/5,400 reserved**; no other GPU request authorized yet. Pilot must pass the independent full-output audit and budget projection before the remaining five windows. Fresh all-queue preflight found 1,538 records, no pending/running or duplicate scoring request; older gpu3 attention remains unresolved and untouched. GPU2's A6000s were below the shared free-memory threshold, GPU5/6 Ada had capacity; compatible pool remains GPU2/5/6, GPU3 excluded after normal probe timeout. Do not replay any request after interruption.

W8's read-only wire audit is in `docs/experiments/w8-wire-audit.md`: current charges are a modeled research rate, not a complete fresh-decoded payload. The next distinct gate is a receiver-only witness with actual framing and tensor bootstrap counted. No such experiment or budget has been authorized by the anchor repair. W11 has no linked independent racket pixel masks; its 3% measurement is only a proxy. W6b smoothing has no quality result and is stopped at the CPU scale gate. These limitations remain beside the repaired comparison.

### GPU scale checkpoint (21:50 UTC)

Pilot `20261031T214253Z-429aabab` completed gpu5 Ada and passed all 13 smoke checks plus independent 12-check full-output audit. All 80 ticks and raw controls were recomputed; maximum control discrepancy 5.23e-5 against 1e-3. Audit `/private/tmp/w9-gpu-pilot-audit/audit.json`, SHA256 `61d38bdef643cb7f5326fd58fefc7a402e8c6c749642a0cfa05a3b67ae45b11d`; output archive `28aeb71d62c64d642f1c5157af6bca691cbb4f6f0c3ebdfdea94887a88177ba4`. Conservative charge 168 GPU-s (167.03-s supervisor duration); smoke/full command 18.70/127.95 s, shared timing contaminated. Source remains `9a74db87`.

After parent review, five independent remaining jobs were submitted once, all pending at the checkpoint:
- al028: `20261031T215041Z-2f1dee96`
- al034: `20261031T215039Z-d1695e8d`
- al043: `20261031T215041Z-3ab89852`
- nad001: `20261031T215041Z-74e3424a`
- nad002: `20261031T215042Z-994d0829`

W9 owns **one bounded batch wait** for these five IDs and records its handle in the experiment entry. No additional requests or retries are authorized. Every result requires its own smoke gate and fresh-process full/raw-control audit before merging quality. Budget **168 charged + 4,500 reserved / 5,400 GPU-s**, CPU **541/5,400** plus bounded final audits. Every job's cap is 900 GPU-s, deadline 22:30 UTC; worst total 4,668 leaves 732 seconds unallocated. GPU5/6 Ada had capacity; GPU2 remains in the compatible pool but was below the shared free-memory threshold. PR #316 CI passed on exact head `d0d06624067128f0fe9447ef43f16791291e2465`; the final scientific results are still pending.

### Completed repair: resume here

All six GPU jobs listed above are **complete and independently full-audited**; all completion/contention events are acknowledged, waits ended, no task-owned scoring job or reservation remains. Both CPU requests are terminal. Final conservative cost **1,043/5,400 GPU-seconds** and **at most 1,924/5,400 workload CPU-seconds**, no transfer of the unused balance to other experiments. Older gpu3 attention remains unrelated, unresolved and untouched.

Root independently verified every archive/audit hash pair, directly recomputed the original t confidence limits, and recomputed both complete derived outputs from the verified original/repair archives with exact structural agreement. Exact analysis code hashes and local Python/NumPy/SciPy versions are in `docs/experiments/w9-inputs/analysis-provenance.json`; GPU production code is `9a74db87`. Accepted mapping `repair-accepted.json` SHA256 `8fa9109e9a18e23d1d50600c468aa3cdfdfe5d58c2f622a8bc6adb680a74dea6` pins all six result and audit identities.

Final two-hour result `w7-repaired-2h.json`, SHA256 `5245d83df6f8ee16ade28174f6f3c46a94b42ae4d32520c2165fa602e83e9d86`: familiar-match Tier 1 LPIPS +0.09650 [0.03490, 0.15810], DISTS +0.08099 [0.05289, 0.10909], conditional on three original missing Med comparisons. Unseen-match Tier 2 has all four anchor families bracketed, LPIPS +0.06713 [-0.00110, 0.13537], DISTS +0.06668 [0.04395, 0.08941]; it still does not pass the frozen two-metric rule at two hours. The new SVT points leave DCVC as the strongest player anchor where present; every whole frame remains worse on both metrics. The Med-only cluster does not pass; excluding the missing-DCVC window still gives a positive five-window sensitivity.

All-assets duration model `w7-repaired-all-assets-intervals.json`, SHA256 `6d00095001d2e149db33b714cd523df2a6e605ef2bea69cae142dfd47a214c37`: conditional joint common coverage 119.5765–198.3917 min. Certified failure through 132.7005 min, numerical transition unresolved through 132.7014 min, certified pass through 198.3917 min. Report roughly **133–198 min (2 h 13 min–3 h 18 min)**, not subsecond experimental precision, a global minimum L*, or all longer matches. Strict all-four-family joint coverage remains unresolved because the original Med comparisons are missing. This assumes a reusable twelve-crop bank and installed decoder weights; it does not verify a continuous stream, actual full framing or live adaptation.

Overleaf main is now **`a3e2888dff74d6c07e1cd73d10b5dc711bd994b0`** with repaired Figure 3, rounded duration sensitivity and W8 limitations. Working PDF `/private/tmp/pointstream-w7-manuscript/build/main.pdf`: 23 pages, empirical checker passes, final results pages visually reviewed by parent and delegate. PR #316 contains the repair; latest derived code/data and this closure are being finalized on `codex/w7-anchor-repair-2026-10-31`.

**Next decision:** the proposed [W8 fresh-decoder witness](cards/w8-fresh-decoder-witness.md) needs separate user authority: at most 3,600 GPU-s and 3,600 workload CPU-s, CPU payload/input oracle first, then one source-hidden receiver witness at the 80 saved ticks if its rate and projection gates pass. No implementation/job has begun under that proposal. W6b smoothing likewise has no new authority beyond its stopped CPU pilot; W11 lacks eligible linked independent racket masks. The paper is a qualified working draft, not submission-ready.

CPU accounting clarification: the earlier 641-second figure was the standalone CPU/audit subtotal. GPU smoke/full child user+system snapshots add a conservative 1,283 CPU-s (1,282.600081 recorded), producing **≤1,924/5,400 total workload CPU-s**, still within the cap. The cumulative full snapshots include the smoke validator; summing smoke again deliberately overcounts it. Receipt `w7-workload-cpu-accounting.json` SHA256 `914d2ff5b90395f09bec6deedbf40afd83aa5faefc12c4009ed5d6d2cc4c95ea` pins all records and scope. Long-lived fleet infrastructure is excluded as previously agreed. No GPU or CPU workload remains active.

Final publication state: PR #315 merged `d14e9678`; repair PR #316 merged **`5af6199f0519189c848004dbf58ed5c6f4b5586c`** after CI passed on exact `60a77b1a1d50ddbd41f8594494514a5839df00d0`. Final local checks: Ruff passes; pytest 1,265 passed, 27 skipped and the same 13 failures reproduced on unchanged Mac main; mypy the same seven baseline errors. The working tree was clean after publication. The user has been asked for the separate W8 witness authority; it remains pending, with no new receiver implementation or computation started. No recurring model monitor or task-owned wakeup was created.
