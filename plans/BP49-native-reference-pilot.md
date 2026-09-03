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

## Delivered

**Outcome: complete.** Operational gate **go**. One-point reference pilot **complete**
(not timed out). Gate A / a size–quality win / E1 evidence: **not licensed**.
Return to Codex. No curves or broad E1 launched.

**Commit:** this session commits the bounded-pilot harness only. Measured JSON
under `outputs/bp49-native-recovery/` is left as written. `HANDOFF.md` is not
part of that commit.

### Inputs and tools

- Video `alcaraz_highlights`, `scene_000` + `scene_028`, 48 frames each, 24 fps,
  3840×2160, shared `alcaraz_highlights_main_court`.
- Full decoded RGB SHA-256: scene_000
  `388665774c91f980c3bf0e329d6f4e3bd7123398e99e9192854540723cc60fd6`, scene_028
  `e2491f5772cab6d89bd8f32af5d691e97dcde1df3a060aa831f9c7a2371d9aeb`. Both match
  `outputs/bp47-e1-preflight/`. `all_hashes_match_bp47: true`.
- ffmpeg `/opt/local/bin/ffmpeg` n7.1.1-56-gc2184b65d2.
- SVT-AV1 `/opt/local/bin/SvtAv1EncApp` v1.8.0, **preset 0**.
- VVenC via ffmpeg `libvvenc`. Live binary **rejects** `placebo` and `veryslow`;
  **driven preset `slower`**. Same result as `codec-floor.json` `selected_preset`.
- PointStream background: ffmpeg `libaom-av1` `-cpu-used 8 -usage realtime`
  (not SVT-AV1). Residual and generation off.
- CUDA hidden. Output root:
  `/home/itec/emanuele/pointstream-data/outputs/bp49-native-recovery/`
  (new directory; BP47 preflight left unchanged).

### Bounds and controls (written before the corresponding encodes)

- PointStream: `bounds-before-run.json` at 2026-09-03T10:28:00Z.
- Reference pilot: `reference-pilot-bounds-before-run.json` at 2026-09-03T11:42:00Z.
  Hard timeout 14400 s. Metric controls from BP45 calibration (identical VMAF
  97.54, unrelated 0.0).

### PointStream `bg-crf51`

Command (conda env `pointstream`, `PYTHONPATH` = checkout):

`python -m experiments.tier.low_rate_sweep --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48 --codec av1 --point bg-crf51 --skip-compare --skip-fallback --out …/sweep-alcaraz_highlights-scene_000+scene_028-n48-av1.json`

| Quantity | Bound | Observed | Verdict |
|---|---|---|---|
| frames | 96 | 96 | held |
| resolution | 3840×2160 | 3840×2160 | held |
| coded bytes | 80 000–50 000 000 | 474 313 | held |
| residual bytes | 0 | 0 | held |
| VMAF | 15–97 | 77.417 | held |
| Y-PSNR dB | 16–45 | 33.003 | held |
| SSIM | 0.72–0.995 | 0.967 | held |
| run_seconds | 30–10800 | 3669.4 | held |
| max_checkpoint_gap_s | ≤3600 | 1664.0 | held |
| hourly_checkpoint_budget_met | true | true | held |
| completion | 1/1/0 | submitted 1, succeeded 1, failed 0 | held |

Ledger: panorama 445 513 (0.9393), metadata 20 201, actor_reference 8 599,
residual 0. Parts sum to 474 313.

Per-scene last−first (band VMAF [−25, +8], Y-PSNR [−8, +3] dB): scene_000
VMAF +0.962 / PSNR +1.028 dB; scene_028 VMAF +6.135 / PSNR −0.303 dB. Alarms:
none. `recovery_alarm`: null. Joined-across-scenes deltas are diagnostic only.

Checkpoints: `prepared/`, `chunk_00/done`, `chunk_01/done`. Largest gap is
assembly scoring 1664 s after both scene checkpoints. Heartbeats every 600 s.
A killed codec still cannot resume mid-bitstream. Attempts: 1. Timing complete.
Log: `sweep.log`.

### Reference one-point pilot (QP 63)

Command:

`timeout 14400 python -m experiments.tier.low_rate_references --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48 --qp 63 --out-dir …/bp49-native-recovery`

Completion: submitted 4, succeeded 4, failed 0. Exit 0. Wall 2889 s (under 4 h).
Per-encode wall 708–724 s (under the 3600 s per-encode alarm). `encode_seconds`
are codec time; wall includes lossless mux, decode, and VMAF.

| Arm | Preset | Bytes | VMAF | Y-PSNR | SSIM | encode_s | decode_s |
|---|---|---:|---:|---:|---:|---:|---:|
| AV1 continuous | 0 | 109 198 | 82.814 | 36.284 | 0.971 | 97.7 | 8.9 |
| AV1 segmented | 0 | 129 081 | 85.645 | 37.159 | 0.975 | 105.3 | 12.9 |
| VVC continuous | slower | 16 270 | 6.832 | 24.386 | 0.842 | 99.4 | 13.0 |
| VVC segmented | slower | 16 445 | 8.297 | 24.340 | 0.841 | 96.0 | 16.1 |

All four `usable=true`. Source hashes identical to the PointStream run.
VVC `placebo` was not driven.

### Go / no-go

- **Go:** native recovery hourly-checkpoint budget on this 48-frame pair.
  Scene checkpoints and 10-minute heartbeats held. Do not treat assembly
  scoring (27.7 min) as a reason to stop; it is under one hour.
- **No-go for a win:** PointStream 474 313 B / VMAF 77.42 versus AV1 QP 63
  continuous 109 198 B / VMAF 82.81. Smaller-and-better AV1 at one unmatched
  point proves no win. VVC QP 63 is a coarsest-rate point (VMAF ~7), not a
  matched-quality comparison.
- **No-go for curves / E1 from this session.** Codex decides the next QP walk
  and whether Gate A is even in reach at this residual-off operating point.

**Skipped:** full independent QP walk, BD-rate, fallback-equivalence, training,
second domain, Cartesian sweep.

**Reproduction:** same two commands above. Chunk and per-pattern JSON under
`*.points/` resume finished work. Historical `outputs/bp47-e1-preflight/` was
not overwritten. Expanding this pilot's QP set in the *same* directory is now
allowed only when frames, preset and implementation match; a code change still
requires a new directory so the measured implementation stays frozen. Neighbour
QPs such as 62 are legal `--qp` values and are not limited to the sparse walk.

**Next:** Codex review. BP45 staged search waits on that acceptance.
