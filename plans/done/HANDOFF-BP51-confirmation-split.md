# BP51 — integrate the confirmation-contamination audit

Archived dispatch. BP51 and its review repairs are complete; current work is BP54.

Trigger: user dispatch from Codex to Antigravity after PR #55 merged.
PointStream targets TOMM by 30 September. Independent confirmation must use
matches not used for development or training. The diagnostic search continues
in parallel; do not change its inputs or implementation.

## Start and ownership

Read root AGENTS.md, PLAN.md, plans/SESSION-REPORT.md, BP46-long-tennis-scenes.md
and this brief. Fetch origin; base a NEW branch and separate worktree on current
origin/main containing PR #55 merge 35a57163bd80fb259f68aae2241e495b27c6cf6b.
Do not reuse Cursor's checkout or change its HEAD. Respect local dirty files.
Use an absolute PS_DATA_ROOT for the existing external data root. Check active
jobs before doing work; do not stop other sessions' jobs.

Own: manifests/bp46_long_tennis_scenes.json, experiments/long_scenes/{schema,verify,extract}.py
only as needed for split metadata/validation (not frame extraction algorithms),
tests/experiments/test_long_scenes.py, plans/BP46-long-tennis-scenes.md,
and a new plans/BP51-confirmation-audit.md. Do not edit shared PLAN.md/HANDOFF.md,
low_rate* code, source frames, cached crops, historical outputs or the paper.
Report central status updates for Codex to integrate.

## Work

1. Bring the audit out of the private walkthrough into the versioned report.
   Verify each claim against actual records; cite file paths. The training meta
   assets/dataset/pointstream_aa_meta.json has 114 tracks: alcaraz_highlights 20,
   federer_djokovic 20, djokovic_zverev 16, alcaraz_perricard 14,
   djokovic_federer 20, sinner_alcaraz 20, alcaraz_ruud 4. Recompute these counts;
   several counts in your previous walkthrough were wrong.
2. All seven existing assets are development/diagnostic material. Set zero
   accepted confirmation videos and update scene roles consistently without
   changing interval hashes, coordinates, eligibility measurements or cached
   frames. Keep the two designated E1 diagnostic videos unchanged; choose a
   clear development-only role for other previously used videos.
3. Record match/source identity and prior use, not just filename disjointness.
   Include training and earlier headroom/background experiments. Compilation
   footage may contain multiple matches. Ambiguous event names must be marked
   unresolved, not guessed (e.g. "ATP Finals / Tokyo" is not a verified event).
4. Ensure the verifier distinguishes diagnostic readiness from independent
   confirmation readiness. Strict confirmation must reject contaminated,
   unresolved or duplicate-match candidates even if filenames differ and the
   count reaches six. Use explicit provenance metadata; do not hard-code seven
   filenames as the definition of contamination. Keep historical manifests
   readable without silently treating missing provenance as clean.
5. Update the extraction/manifest-generation metadata path so regeneration
   cannot silently restore the old false confirmation assignments. Do not run
   full extraction or overwrite historical output mirrors. Report the new
   authoritative manifest path and any stale mirrors explicitly.
6. Write a fresh-data acquisition brief: at least six distinct, verified,
   previously unused matches, source identifiers, rights/access constraints,
   and annotation/eligibility checks at 48/96/192/384 frames. No downloads or
   annotation campaign in this task. No source-footage redistribution promise.

## Verification and return

Follow test-design: propose the behavior tests and obtain approval before adding
tests. Cover contaminated/unknown/duplicate match rejection, a genuinely clean
synthetic confirmation set, regeneration preserving split roles, and existing
diagnostics remaining loadable. Do not add tests just for coverage.
Run targeted tests, ruff, relevant mypy and import-direction checks. Expected
real-data verdict: diagnostics ready, zero accepted independent confirmation
matches, strict confirmation fails honestly. No expensive codec/GPU runs.

Commit, push your branch and open one PR; do not merge/delete branches or
worktrees while the other session runs. Report at plans/BP51-confirmation-audit.md:
commit/PR, exact commands, audit sources, corrected counts and identities,
manifest delta, test outcomes, unresolved identities and acquisition needs.
The audit is not complete if only the private walkthrough changes.
