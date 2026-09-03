# BP52 — bounded background CRF search, then reference bracketing

Archived dispatch. BP52 and its review repairs are complete; current work is BP53.

Trigger: user dispatch from Codex to Cursor after PR #55 merged.
Question: can stronger background quantization reduce PointStream's dominant
payload enough to approach independent AV1/VVC rate--quality points on the
BP49 diagnostic pair? This is exploration, not confirmation or a promised win.

## Start and ownership

Read AGENTS.md, PLAN.md, plans/SESSION-REPORT.md, BP49-native-reference-pilot.md
and this brief. Fetch origin; create a NEW branch and separate worktree from
origin/main containing merge 35a57163bd80fb259f68aae2241e495b27c6cf6b. Do not
reuse Antigravity's checkout or change its HEAD. Check jobs/resources first.
Own only experiments/tier/low_rate_plan.py, an optional focused BP52 driver under
experiments/tier/, narrowly needed low_rate_sweep.py plumbing, corresponding
tests, and plans/BP52-background-crf-search.md. Reuse existing measurement and
recovery machinery. No changes to src/, long_scenes/, manifests, paper, shared
PLAN.md/HANDOFF.md, reference identity policy or codec adapters.

Snapshot and record the input manifest before measuring. Antigravity's parallel
split audit must not change this experiment's frames or code. Set PYTHONPATH to
your worktree and PS_DATA_ROOT=/home/itec/emanuele/pointstream-data. Keep caches
on local disk and use the pinned pointstream conda environment.

## Fixed experiment

alcaraz_highlights scene_000 + scene_028, 48 frames each, native 3840x2160,
24 fps, shared alcaraz_highlights_main_court context. Full decoded RGB hashes:
scene_000: 388665774c91f980c3bf0e329d6f4e3bd7123398e99e9192854540723cc60fd6
scene_028: e2491f5772cab6d89bd8f32af5d691e97dcde1df3a060aa831f9c7a2371d9aeb
Require exact matches before encoding. Preserve the current injected objects,
appearance JPEG quality 40/downscale 2, motion max_points 16, canonical canvas,
reference policy and full-resolution delivery. Generation and residual off.
Background remains libaom-av1 cpu-used 8 realtime. Do not confuse this with
SVT-AV1 preset 0, which is the independent reference encoder.

Output: NEW data-root outputs/bp52-background-crf/ directory. If it already
exists with another identity, stop and choose a documented new suffix. Never
modify outputs/bp49-native-recovery or copy its checkpoints to bypass identity.

## Small batch and gates

1. Add only selectable background points needed for CRF 63 and 57, preserving
   the original bg-crf51 control. Verify the live background encoder accepts
   these values and the requested CRF reaches the actual command. No preset,
   resolution, duration, scene-count or representation changes in this batch.
2. Before any long run, follow test-design: propose behavior tests for approval,
   reuse existing tests, then run lint/types/targeted tests/layer checks. Freeze
   and commit implementation BEFORE measurements; record content digest, tool
   paths/versions/effective arguments. Code changes require a new output identity.
3. Write bounds and run fresh native-resolution metric controls before trusting
   rankings: identical, mild blur/noise, severe degradation, unrelated natural
   content. Use existing calibration fixtures and absolute scales; do not
   assume severe and unrelated must be ordered. Check identity > mild > severe
   and mild > unrelated for higher-is-better metrics. Record full control paths.
4. Run PointStream CRF 51, then 63, then 57. Three points maximum. CRF51 is a
   fresh regression/control point, not a copy of the old JSON. Compare it with
   BP49's matching configuration; investigate unexplained differences before
   continuing. Confirm that quantization changes transmitted background bytes
   and decoded content; a flag being accepted is insufficient.
5. If all three points and measurement checks are valid, run independent
   references at AV1 preset 0 and the live verified slowest VVC preset (BP49:
   ffmpeg libvvenc slower; placebo rejected). Start QP63 for each codec. Then
   add up to THREE VVC QPs adaptively to bracket the candidate quality range:
   start QP51, choose the next between available bounds or move to QP39 if all
   VVC points remain below candidates. Never compare equal QP as equal quality.
   Continuous and segmented access patterns run for every selected QP. Maximum
   five codec/QP settings total: one AV1, four VVC, hence ten pattern results.
   Stop rather than exceed that cap or force a BD-rate from inadequate overlap.
   Compare shared-context PointStream to continuous references; segmented
   references are an access-pattern diagnostic, not a matched PointStream arm.
6. Same implementation may reuse per-pattern/QP checkpoints. Request the full
   accumulated QP list when writing the final aggregate; --qp with only the new
   point otherwise produces a report containing only that requested subset.

## Bounds, accounting and stop conditions

Write bounds BEFORE results. For the CRF51 control carry BP49's quantitative
bands and compare to its recorded outputs. For stronger degradation do NOT
reuse its quality floor or byte floor: expected diagnostic bands are 0--98 VMAF,
8--45 dB Y-PSNR, 0--1 SSIM; positive coded bytes below 50 MB. These are alarms,
not acceptable-quality targets or proof of a correct instrument. No finite
quality outside the range is silently accepted. Negative SSIM or another
out-of-band result requires investigation, not automatic deletion.
Frame count/dimensions/hash and byte-ledger equality are exact invariants.
Carry scene-local late-frame bands VMAF [-25,+8], Y-PSNR [-8,+3] dB. Preserve
and investigate alarms; no joined-scene delta used as a scene drift measure.

Run detached. Progress at least every ten minutes; durable checkpoint gap at
most one hour INCLUDING preparation/scoring. A killed codec cannot resume
mid-bitstream. Stop expansion if a gap exceeds one hour. Set a watchdog with
TERM then KILL grace, record every timeout/failure and all retry time; never
count a failed point as completed. Cap total experiment wall allocation at
eight hours, including controls and retries, and stop after the current bounded
stage when the remaining budget cannot safely fit another. Do not kill other
sessions' jobs. Native BP49 success does not license unbounded longer runs.

Report TOTAL transmitted bytes plus background/reference/motion/metadata parts,
full-frame VMAF/Y-PSNR/SSIM and encoder/decoder time. Retain separate preparation,
codec, scoring and attempt wall times; evaluation time is not codec encode time.
Do not add or promote LPIPS without native-resolution calibration. One sequence
is n=1 independent experimental unit: no significance or generalization claim.

## Return and out of scope

Write plans/BP52-background-crf-search.md using SESSION-REPORT fields: commit/PR,
exact reproducible commands, source/code/tool identity, all submitted/succeeded/
failed counts, controls/bounds/alarms, complete per-point size--quality--time
tables, accounting, checkpoint gaps, outputs and licensed conclusions.
Return even if no candidate improves. Explain whether the next plausible lever
is background resolution, background encoder effort, or longer contexts; do not
implement those alternatives here. No training, fresh data acquisition, full
Cartesian grid, broad E1, confirmation, paper claim, source-code refactor or
automatic mixed-scene scheduler. Push one PR and return to Codex for review.
