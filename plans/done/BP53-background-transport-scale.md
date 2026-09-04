# BP53 — half-resolution background transport

Archived 4 September 2026 after review and integration. Historical instructions
below are not a current dispatch; follow PLAN.md and the active plan index.

Active. Cursor implements and runs the bounded diagnostic; Codex owns design
changes and final evidence interpretation. Read AGENTS.md, PLAN.md,
SESSION-REPORT.md and BP52-background-crf-search.md. Deadline 30 September;
evidence freeze 20 September. Trigger: quantization-only search found no win.

## Isolation and scope

Start a NEW branch/worktree from current origin/main after PRs #57/#58 and the
integration docs are merged. Check git status and active jobs. Do not reuse old
BP51/BP52 worktrees; preserve outputs. Use the external PS_DATA_ROOT and set
PYTHONPATH to the new checkout. No agent should modify that checkout during runs.

Own narrowly: src/contracts/config.py; background strategy/artifact/config
binding and stream-state code needed for transport scaling; runner background
view/metadata handling only if needed; corresponding tests; one focused
experiment driver and plans/BP53-background-scale-report.md. Do not change
perception, foreground, frame selection, source manifests, metrics, encoder
presets, residuals, temporal policy, or unrelated reference code.

## Design decision already made by Codex

Add background.transport_scale with default 1.0. First implementation supports
only 1.0 and 0.5 under panorama-stream; reject unsupported combinations rather
than silently ignoring the setting. This changes coding resolution, not scene
registration, canonical geometry, output resolution or object coordinates.

1. Build the SAME full-resolution canonical background and transforms as BP52.
2. Immediately before stream coding, downsample the background with INTER_AREA
   to an explicitly recorded even-sized coded raster. Define deterministic
   rounding (floor to the nearest positive even width/height); reject too-small
   inputs. The stream's decoded prediction history remains in coded coordinates.
3. After decoding, restore to the exact original canonical width/height with
   INTER_LINEAR before exposing the background view to warp/compositing.
   Homographies still map frame to original canonical coordinates and remain
   unchanged. No scaled-coordinate transform shortcut in this first experiment.
4. Both encoder reconstruction and an independently exercised client decoder
   use only decoded payloads plus charged metadata to do this same restoration.
   Serialize original/coded dimensions and required reconstruction policy in
   the actual wire/ledger path, not only in a Python attribute or report. Count
   every added byte. No receiver-only access to source arrays or encoder state.
5. Preserve context resets, prefix-stability checks and coded decoded-history
   semantics. Checkpoint/restore must retain scale, coded dimensions and original
   geometry; mismatches fail closed. Never mix scales within a reference chain.
6. Scale 1.0 follows the old path without resampling. Preserve its reconstruction
   exactly; any new metadata bytes must be explicitly accounted, not hidden.

If the existing transport cannot carry the required information without a larger
protocol change, stop and return the concrete design conflict to Codex. Do not
quietly implement reduced-coordinate warping or a new architecture.

## Verification gate before native runs

Use test-design and request approval before new tests. Required behavior:
scale-one compatibility; invalid scales rejected; static/pan geometry preserved;
decoded half-scale chain restored identically on encoder/client paths; odd/even
rounding; context reset; snapshot/resume equivalence and changed-scale rejection;
complete byte accounting including dimension metadata. Use synthetic translated
patterns/objects to catch coordinate shifts, not merely equal output shapes.
Run ruff, mypy, targeted tests and import-direction checks. Commit/freeze the
implementation before measurement; an edit afterward requires a new directory.

## Bounded native experiment

Same BP52 alcaraz_highlights scene_000 + scene_028, 48 frames each, native 4K,
24 fps, same full RGB hashes and fixed injected objects. Output root NEW
outputs/bp53-background-scale/. Never rewrite BP49/BP52 artifacts.

Run exactly three PointStream settings: scale 1.0 CRF51 (regression control),
scale 0.5 CRF51, scale 0.5 CRF63. Generation/residual remain off. All other BP52
settings fixed. Half-scale payload reduction is not itself a quality-matched win.

Before measuring, write two-sided bounds and fresh native metric controls using
BP52 fixtures. Identity/dimensions/duration and ledger balance are exact; control
quality must reproduce BP52, with any explicit metadata delta reconciled. For
half scale, inherit broad diagnostic bands (VMAF 0–98, Y-PSNR 8–45 dB, SSIM 0–1,
positive total bytes below 50 MB). Do not inherit BP43's optimistic pixel-ratio
prediction as a pass requirement. Alarm -> save partial report, stop expansion.
Retain per-scene late-frame bands and all recovery identity checks.

Compare to BP52 continuous AV1 QP63 and VVC QP51/QP39 only after proving source,
codec/preset/color settings and measurement code are unchanged. Reference JSONs
may be cited as immutable prior diagnostics, never transplanted into a new
checkpoint identity. If comparability cannot be established, run those three
codec/QP settings freshly (six pattern results with existing CLI), within budget.
No interpolation/BD-rate claim when overlap/point count is inadequate.

Run detached, progress every ten minutes, checkpoint gaps <=1 hour including
preparation and scoring, overall wall budget <=8 hours including controls and
retries. Log directory exists BEFORE tee; propagate both logger/runner failures.
Do not start a point that cannot fit the remaining allocation. No more points,
no half-vs-quarter grid, no longer contexts or slower background presets yet.

## Timing and report

Report complete bytes, full-frame quality and available wall/stage times. Record
downsample/upsample and background coding/decoding overhead where separable;
do not relabel runner/scoring wall as semantic encode time. Separate end-to-end
encoder/client timing remains Codex's next instrumentation decision if not already
available. Unknown timings stay null with a reason; no real-time/speed ranking.

Return one PR and BP53-background-scale-report.md with exact commands, frozen
commit/digest, source/tool identity, all controls/alarms, submitted/success/failed
counts, size-quality-time ledger, geometry/resume proof and output paths.
n=1 diagnostic pair, no standard error/generalization claim. Stop for Codex
review before broad curves. A loss keeps the next hypotheses open: longer
contexts and more background encoder effort, tested separately.
