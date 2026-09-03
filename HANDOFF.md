# PointStream handoff — 3 September 2026

Trigger: the user requested a clean session boundary after the recovery changes
are ready to merge. This is that bounded engineering handoff, not a request to
launch broad E1. Read `plans/BP48-recovery-validation.md` for the next gate and
the exact report expected from Cursor.

Target: **submit a defensible ACM TOMM manuscript by 30 September 2026. The date
is hard because the author's contract ends then.**

## Read in this order

1. `AGENTS.md`
2. `plans/ROADMAP.md`
3. `plans/TERMINOLOGY.md`
4. one assigned brief only
5. `PLAN.md` only when historical evidence is needed

Every dispatched task follows `plans/SESSION-REPORT.md`.

## Current state

PR #52 is merged: `origin/main` is `68a03dc` as last fetched on 3 September.
Recovery work is on `codex/recovery-safety` in `/tmp/pointstream-recovery`, based
on Cursor's `e08641c` and `fadf9f6`, followed by Codex's `a155df2` and the approved
regressions. Check `gh pr list --head codex/recovery-safety` for the current PR
and `gh pr checks <number>` before merge. Do not mistake local validation for
completed GitHub checks. The recovery report records the test results and the
known 80.33% coverage / 81% local-buffer discrepancy; the CI policy remains 77%.

Preserved worktrees: primary `wave10/bp46-manifest` at `21c0681` has an unrelated
edit to `experiments/long_scenes/extract.py`; do not overwrite it. BP44 is
`cursor/bp44-canonical-canvas` at `02418fe`; BP45 is `cursor/m1-bp45` at `7b263f3`;
Cursor's E1 worktree is `cursor/e1-native-preflight` at `fadf9f6`. Integration is
`codex/bp44-bp46-integration` at `68a03dc` in `/tmp/pointstream-integration`.
Ask before removing any worktree; remote branch deletion remains human-only.

- The contracts, components, end-to-end runner and BP31 multi-scene experiment
  are merged on `main`; PR #45 passed tests, lint and typing.
- The best current system result is **+90.97% BD-rate against AV1** on one tennis
  video, two scenes and eight frames per scene. PointStream is about 20× the
  reference codec's end-to-end wall time in that run.
- Increasing scene length from 8 to 16 frames improved the single-point
  PointStream/AV1 byte ratio from 2.17× to 2.01×. This confirms amortization but
  does not establish the long-run slope.
- At 24 frames, independently built scene panoramas acquire different
  dimensions. Predictive background-sequence coding requires equal dimensions.
  BP44 now implements an offline canonical background canvas per compatible
  context. Synthetic 48-frame static/pan and context-reset runner tests pass;
  native-resolution E1 preflight completed, but the repaired native checkpoint
  budget and independent reference pilot still need verification.
- The current ~43 dB tests are high quality. The first winning-regime search
  moves to ultra-low bitrate and long eligible scenes.
- No reconstruction model beats the reference-image paste control. Generation
  training is parked until the background and lean non-generative payload can
  win on rate--quality.
- The second domain, learned-codec baseline and speed optimization begin only
  after the first-domain rate--quality result is confirmed.

## Optimization order

1. Beat AV1 and VVC on size at matched quality in a named tennis regime.
2. Confirm on at least six independent videos/matches.
3. Explain it with a core component ablation matrix.
4. Add DCVC-RT and one independent domain.
5. Optimize speed on the frozen winning configuration.

Time is measured and shown from step 1, but it is not a gate until step 5. A
slow win must be framed as offline or compute-intensive, never live.

## Immediate work

1. **Codex:** inspect current PR review and CI at the recovery branch's head;
   merge when ready. Preserve Cursor's original commits/worktree.
2. **Cursor:** execute the bounded native recovery-budget rerun specified in
   BP48, using new output directories and the same BP47 decoded source hashes.
3. **Cursor, only after that gate:** run one slowest-preset AV1/VVC runtime and
   recovery pilot on those frames; report all three dimensions and failures.
4. **Codex:** review before expanding to curves or broad E1. No win is yet
   established; confirmation footage remains incomplete.

## Running work and open decisions

No experiment was launched by this recovery session. No owned local encode or
test job remains at handoff; check `ps -u emanuele -o pid,etime,args` and the PR
checks for current state. Other users' GPU jobs are not ours to stop.

Can the native rerun keep every gap between durable checkpoints below one hour?
If not, stop expansion and ask Codex to scope a smaller recoverable unit. A killed
codec process cannot resume mid-bitstream. A hard-killed attempt's time is only
a lower bound and must not become a complete size/quality/time comparison.

Does the pinned VVenC binary expose `placebo`, or only `slower`? Verify against
the actual binary; record the slowest supported preset, never infer it from a
generic preset list. Neither codec's reference curves have been run on these
native inputs. Whether a winning regime exists remains an experimental question.

## Landmarks and host details

- `plans/BP48-recovery-validation.md`: validation commands and dispatch brief.
- `plans/BP47-e1-preflight.md`: historical input/output provenance, not permission
  for broad expansion.
- `src/runner/recovery.py`, `chunk_checkpoint.py`, `run.py`: recovery contract.
- `tests/runner/test_recovery.py`: approved interrupted-run regressions.
- `experiments/tier/low_rate_sweep.py`: point selector and operational gate.
- Set `PYTHONPATH` to the worktree, use conda `pointstream`, keep caches in `/tmp`.
  Imports from outside a worktree otherwise resolve to the editable main tree.
- This session needed escalated shell calls because the sandbox's `bwrap` binary
  was missing. `gh` is restored at `/home/itec/emanuele/bin/gh`.

## Historical first wave

Original wave:

- **M1:** extend AV1/VVC to their lowest useful rate, fix metric-direction
  typing, and calibrate the primary VMAF comparison;
- **B1:** implement canonical background canvases and context resets;
- **D1:** extract and validate 2/4/8/16-second eligible tennis scenes;
- **M2:** produce an exact byte ledger and fit per-additional-frame cost from at
  least three durations;
- **P1:** restore reproducible manuscript rendering and keep a live page budget.

After the current gates, run E1, the low-rate × duration search. See `plans/ROADMAP.md` for dates,
dependencies, harness assignment and the required report from every session.

## Standing safeguards

- Write two-sided bounds and null controls before reading a result.
- Every result reports rate, all declared quality axes, encode time and decode
  time, even while only rate--quality gates the search.
- The same source frames, resolution, frame rate and colour convention go to
  PointStream, AV1 and VVC.
- Report every searched configuration. A scoped win found by search is valid;
  presenting it as predicted is not.
- YouTube-derived source footage is not redistributed. The paper and artifact
  must not promise a releasable dataset without a rights review.
- The evidence freezes on 20 September.
