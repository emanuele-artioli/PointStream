# BP56 — bounded background encoder-effort pilot

Cursor executes in a NEW branch/worktree from updated origin/main; Codex reviews
prefix semantics and evidence. Read AGENTS.md, PLAN.md, SESSION-REPORT.md,
done/BP53-background-scale-report.md and done/BP53-measurement-provenance.md.
Deadline 30 September; evidence freeze 20 September. No broader sweep authorized.

## Delivered

Measurement complete 4 September 2026. Report: `plans/BP56-background-effort-report.md`.
Native outputs: `outputs/bp56-background-effort/`. Stopped before expansion.

## Question and scope

Does spending more background encoder effort help the short-pair rate–quality
gap? Background coding still uses realtime libaom cpu-used 8; this is not the
slow SVT-AV1 reference. Do not change reference presets, metrics, foreground,
source selection, scene length, geometry, residuals or generation. Keep scale 1.
Own narrow background config/stream option plumbing, focused tests, one bounded
driver and plans/BP56-background-effort-report.md. Retain defaults unchanged.
No global CODECS monkeypatch in production; config, provenance and checkpoint
identity must include every effective background option.

## Gate 1: small synthetic capability proof

Resolve installed encoder path/version and supported options. Candidate is
libaom good-quality usage, cpu-used 4, lag-in-frames 0, bf 0; this is deliberately
a bounded higher-effort test, NOT the slowest-background claim. Keep rate control,
pixel format, threading and reference mode unchanged. If unsupported, stop;
do not silently select a substitute or expand into a preset grid.

Before native work, encode textured/static/translated synthetic background
sequences of 2, 3 and 4 frames. Assert every already-emitted packet stays exactly
unchanged as prefixes grow. Exercise last-reference chains, same-size context
reset and byte-only client restore; compare against encoder reconstruction.
Drive the option and record output/command changes. A different flag alone is
not evidence it is used. If prefix stability fails, stop for Codex design review;
do not waive the check, re-encode old bytes for free or switch protocol.

Use test-design approval for new behavior tests. Include changed-option resume
rejection and default compatibility. Run full ruff/mypy, touched tests and
import-direction checks. Commit/freeze before native measurement. Any code edit
afterward needs a new output identity; never overwrite identity.json to resume.

## Gate 2: bounded native pilot

Use EXACT BP53/BP52 source RGB hashes: alcaraz_highlights scene_000 + scene_028,
48 frames each, native 3840x2160, 24 fps, same ordered context/injected objects.
No generation or residual. Use last-reference mode and canonical canvas.
Fresh output directory outputs/bp56-background-effort/; never rewrite BP49–BP53.

At most THREE points, in this order:
1. Original realtime cpu-used 8, CRF51, scale 1: regression control.
2. Candidate good-quality cpu-used 4, CRF51, scale 1.
3. Candidate good-quality cpu-used 4, CRF63, scale 1, only if runtime permits.

Fresh native identical/mild/severe/unrelated metric controls precede points.
Write bounds first: control quality/source/geometry exact BP53; reconcile all
wire metadata differences explicitly. Candidate VMAF 0–98, Y-PSNR 8–45 dB,
SSIM 0–1, bytes >0 and <=50 MB (broad diagnostic bounds, not hoped-for gains).
Retain per-scene late-frame bands and ledger/invariant checks. Alarm means save
partial report and STOP before another encode. Do not force three completions.

Total active-work allocation <=8 hours including controls, retries and native
preparation. Use repaired persistent attempt budget. Unknown crash gaps mean
budget compliance is unresolved: stop for review, not another allocation.
One native encode attempt <=1 hour; abort/time out and report if it cannot fit.
Use subprocess timeouts/watchdog with durable prior progress, not an assumption
that a preset finishes. Honor the shorter of stage and remaining batch limits.
Leave enough time for scoring; BP53 scoring nearly exhausted the hourly gap.
Do not clear longer runs just because this pair succeeds. Launch detached,
progress <=10 minutes, elapsed heartbeats <=60 seconds; create logs before tee
and propagate both pipeline failures. No fourth point or automatic restart loop.

## Evidence and return

Compare immutable BP52 continuous AV1 QP63 / VVC QP51 and QP39 only after checking
full source identity, reference commands/presets/color and metric code identity.
Background ffmpeg version equality alone is not enough. If verification fails,
return unranked diagnostics; no extra reference encodes are authorized here.
Report bytes, full-frame quality and available stage/attempt times, including
new background coding/decoding effort. Semantic encoder/client times remain null
where unavailable; do not infer them by subtracting metrics (BP55 owns design).
No BD-rate, statistical generalization or speed claim from one pair.

Return one PR and the report with exact commands, frozen digest, source/tool
identities, controls, prefix/client proof, success/failure counts, timing bounds,
byte ledger, output paths and next recommendation. A loss leaves longer context
and foreground/background quality allocation open; return to Codex, do not launch.
