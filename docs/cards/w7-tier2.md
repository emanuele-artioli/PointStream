# Card: W7 Tier 2 — a match the tennis base never trained on (nadal_kyrgios)

Model: Sonnet, effort high. Runs locally on the Mac in `~/Desktop/PointStream`, where
`scripts/ps-fleet` reaches gpu1–gpu6. Root session (report address):
`session_01MbpYHe92aChsjy7MqXQ6jZ`. If `SendMessage` can't reach it, end your last
reply with the report so the user can paste it there.

## Goal
Run W7's Tier 2 test and record every result as written in
`docs/experiments/w7-untouched.md`: does PointStream's player-region win hold on a
match the tennis base never saw? The match is `nadal_kyrgios`. Its own adapter is
fitted on the first half of the broadcast, and three evaluation windows come from the
second half. Finish by running `experiments/tennis/w7_analysis.py` on the three windows
(see its `inputs`/CLI; extend `W7_INPUTS` or pass paths for Tier 2). Commit the result
JSON as `docs/experiments/w7-specs/w7-tier2-result.json`. **Do not interpret the
quality numbers**: the root session applies the rule.

## Read first
AGENTS.md; docs/fleet.md. In `docs/experiments/w7-untouched.md`:
- the rule entry (top: Tier 2, decision rule, narrowing);
- "W7 CPU stage" (Tier 2 lineage verdict, nadal source file and hash, the CPU plan for scene scan and K8b registration);
- "W7 Tier 2 decisions";
- the reference-bank decision;
- the Tier 1 batch entries (every stage, failure and fix);
- the 2026-10-31 entries (Tier 1 outcome, Tier 2 budget).

The Tier 1 card `docs/cards/w7-tier1-batch.md` shows the working style that succeeded.

## Frozen choices (written by the root session before any Tier 2 extraction)
- **Windows.** The seeded draw (seed 20261030) ranges over shots of ≥240 frames starting in the second half (frame ≥ 57,753 of
  115,505), no cut inside a window. In each drawn shot, take the first 240-frame interval the frozen registration rule admits.
  Draw in order until three windows are admitted. List every rejected shot and why; never pick one by hand.
- **Adapter.** Fit on first-half shots only, with the frozen step-5 recipe. Copy the Alcaraz adapter fit (`20261023T221650Z-3d7cbe2f`)
  and the jobs that prepared its fit data (Al 002/006/011, three clips):
  - same base (`20261023T194934Z-b0dcbb19` published.tar), rank, steps, learning rate, precision and 4-bit export;
  - **three** first-half fit clips, drawn with the same seed from first-half shots ≥240 frames that pass registration, in draw order.
  The adapter is billed in the bitstream exactly as the Al/Med adapters are. No evaluation frame may be in any fit input; record the
  frame ranges.
- **Pipeline per window:** the frozen W5 recipe, exactly as Tier 1 ran it:
  - scale-4 paid SVT base, KPS2 keypoints, the 4-bit adapter, players generated on every third frame;
  - reference = the window's own frame-0 crops (`window_frame0`), billed;
  - anchors: SVT plain half, plain quarter and half +20 outside R, plus DCVC-UF with the `--bracket` path.
  - DCVC: if a window's DCVC fails decode acceptance, one rerun of the frozen spec is allowed. If that fails too, the window is
    reported without DCVC (as med000) and you tell the user.
- **Exposure to disclose:** K8 and K8b used nadal court marks for registration design; no generator input contains nadal frames.

## Order and stop points
1. **CPU stage** (no GPU): scene scan and K8b registration (on the server under `/local/users/emanuele`, never `/tmp`), the draw of
   three evaluation windows and three fit clips, extraction and staging (sha256-verified), a `windows.json` manifest for Tier 2, and a
   nadal entry for the adapter in `window_manifest.py` (with tests).
   - Look up the GPU seconds of the Al adapter fit and its fit-data jobs from their `execution.json`.
   - Make a GPU projection: fit-data stages + adapter fit + 3 windows at Tier 1's measured per-stage costs (nadal is 50 fps, so use the
     Med windows' measured times).
   - Record it all. **If the projection exceeds 18,000 GPU-s (5 GPU-h), stop and report** before any GPU job.
2. **Adapter:** fit data, then the fit, then validation (the fit job's own checks). Record the adapter size in bytes, which is billed.
3. **Windows:** the Tier 1 chain for the three windows with the new adapter, batched by stage (`w7_specs plan`/`batch` with the
   Tier 2 manifest).
4. **Analysis:** `w7_analysis` on the three windows; commit the result JSON and a dated entry with the archive hashes and the
   printed summary as written.

## Budget and scope
- **18,000 GPU-s (5 GPU-h)**, approved by the user on 2026-10-31. Failed smokes, reruns and the adapter fit are included.
  Keep a running ledger in the entry. Stop **before** a job that would cross it.
- Out of scope: any change to the recipe, a different adapter rank or fit length (that is the narrowing test if Tier 2 fails),
  more than three windows, Tier 1 windows, W8.
- Same rules as Tier 1:
  - record each job ID when you submit it;
  - never replay a failed job: fix it with a test and submit a new one;
  - wait only with `scripts/ps-fleet wait`;
  - check the fleet and all hosts before submitting.

## Branch and finish
Use a branch from `origin/main`. Edit `docs/experiments/w7-untouched.md` (append entries at its end), the Tier 2 specs and
manifest, and the code you fix (with tests: `ruff check`, `mypy --config-file pyproject.toml`, `python -m pytest`). Open pull
requests with auto-merge as you go. At a stop condition, send the report ([format](../sessions.md#the-report)) to the root session.
