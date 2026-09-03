# BP49 — native recovery verification and reference pilot

**Active. Owner:** Cursor executes; Codex reviews operational limits and evidence.
Read `AGENTS.md`, `PLAN.md`, this brief and `done/BP47-e1-preflight.md`.
Recovery is merged. Do not reimplement it or reuse pre-repair checkpoints.

## Scope and order

1. Check resources and pin tool paths/versions. Inputs: `alcaraz_highlights`,
   `scene_000` + `scene_028`, 48 frames each, native 3840×2160 at 24 fps, shared
   main-court context. Compare FULL decoded RGB hashes against records under
   the data root's `outputs/bp47-e1-preflight/`; abbreviated hashes are not enough.
2. Write two-sided bounds and metric controls before reading results. Rerun only
   PointStream `bg-crf51`, generation/residual off, canonical canvas, injected
   objects, in a NEW output directory. The old preflight is not E1 evidence.
3. Run detached with progress and durable checkpoints. Verify saved preparation,
   scene completion, delivered dimensions, complete byte ledger, per-scene
   last-minus-first quality alarms and cumulative timing. The largest gap
   between durable checkpoints must be at most one hour. If a stage cannot fit,
   stop expansion and report it to Codex; do not silently accept uncheckpointed
   multi-hour work. A codec cannot resume mid-bitstream.
4. Only after the operational gate, run a bounded one-point pilot for each
   reference at its slowest supported preset. SVT-AV1 uses preset 0. Verify
   whether the actual VVenC binary supports `placebo` or only `slower`; record
   the preset actually driven. Bound runtime before launch and report a timed-out
   pilot as failed, not as a completed rate point.
5. Return to Codex before curves or broad E1. Do not launch training, a second
   domain, or a full Cartesian sweep from this brief.

## Fair comparison

Both arms use identical decoded frames, duration, resolution, frame rate and
colour conventions. Keep continuous-context and independent-segment access
patterns separate. Continuous references may retain prediction across scene
joins wherever PointStream retains its background reference; reset at the same
context boundaries. Do not force a keyframe per tennis point in that arm.

The PointStream background encoder is ffmpeg libaom-av1 with realtime usage and
cpu-used 8. It is not the SVT-AV1 preset-0 reference. Record background, residual
and reference provenance separately. Residual and generation remain off here.

Always report size, declared quality axes and time. A slow size–quality win is
acceptable during search; an unknown total runtime is not a complete comparison.
Retries accumulate time. After a hard kill, preserve the labelled lower bound
instead of reporting only the retry's duration. Failed or unusable points cannot
count as completed successes. A single unmatched-quality point proves no win.

## Required report

Use `SESSION-REPORT.md`: commit, exact commands, executable paths and versions,
effective presets, full source hashes, output/log/checkpoint paths, bounds and
controls, submitted/succeeded/failed counts, all timing and checkpoint-gap
fields, per-scene quality alarms, byte accounting and explicit go/no-go.
State skipped work and recovery failures. Preserve historical outputs unchanged.

BP45's staged search follows acceptance. BP46 confirmation-footage work and
`PAPER-NEXT.md` can proceed independently without changing an active run.
