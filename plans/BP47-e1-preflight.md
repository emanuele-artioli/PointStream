# BP47 E1 native-resolution preflight

**Roadmap:** E1 preflight (not the broad search).  
**Outcome: complete.** One native 4K PointStream point passed with source identity and delivered-payload accounting. Gate A is still open. This is not E1 evidence.

## Setup

- **Commit:** `68a03dc` (PR #52 merge). Worktree dirty only with the `--point` selector and this report.
- **Branch / worktree:** `cursor/e1-native-preflight` at `/home/itec/emanuele/pointstream/.claude/worktrees/e1-preflight`
- **Starting `main`:** not BP44/BP45/BP46 branches; new output directory.
- **Point:** `bg-crf51` — panorama-stream CRF 51, residual off, JPEG q=40 ds=2, 16 motion points, generation off, objects injected, canonical canvas.
- **Input:** `alcaraz_highlights` `scene_000` + `scene_028`, 48 frames/scene, 24 fps, 3840×2160. Both clips `context_id=alcaraz_highlights_main_court`.
- **Source hashes (full decoded RGB):** scene_000 `388665774c91f980…`, scene_028 `e2491f5772cab6d8…`. Shapes `(48, 2160, 3840, 3)` each.
- **Implementation digest:** `f4a3f563ca406626…`
- **Tools:** Background *stream* coding is ffmpeg ``libaom-av1`` with
  ``-cpu-used 8 -usage realtime -lag-in-frames 0 -bf 0`` (not SVT-AV1).
  Residual was **off**. `/opt/local/bin/SvtAv1EncApp` SVT-AV1 v1.8.0 preset `0`
  is the independent AV1/VVC reference (and residual) configuration, unused on
  this PointStream point. ffmpeg n7.1.1-56-gc2184b65d2 at `/opt/local/bin/ffmpeg`.
  CUDA hidden (`CUDA_VISIBLE_DEVICES=`); injected objects skip YOLO.
- **Output:** `/home/itec/emanuele/pointstream-data/outputs/bp47-e1-preflight/`
- **Skipped on purpose:** AV1/VVC reference curves and fallback-equivalence. Those are the next batch, not this gate.

## Bounds (written before the encode)

File: `outputs/bp47-e1-preflight/bounds-before-run.json`

| Quantity | Low | High | Observed | Verdict |
|---|---:|---:|---:|---|
| frames | 96 | 96 | 96 | held |
| resolution | 3840×2160 | 3840×2160 | 3840×2160 | held |
| coded bytes | 80 000 | 50 000 000 | 474 313 | held |
| VMAF | 15 | 97 | 77.417 | held |
| Y-PSNR (dB) | 16 | 45 | 33.003 | held |
| SSIM | 0.72 | 0.995 | 0.967 | held |
| run_seconds | 30 | 7200 | 4382.7 | held |
| completion | 1/1/0 | 1/1/0 | submitted 1, succeeded 1, failed 0 | held |

No bound was revised. No numeric alarm.

## Result

**Completion:** submitted 1, succeeded 1, failed 0. Sweep exit 0. `usable=true`, `is_rate=true`, `e1_evidence=false`.

**Byte ledger** (transport total 474 313 B; parts sum to total):

| Category | Bytes | Share |
|---|---:|---:|
| panorama | 445 513 | 0.9393 |
| metadata | 20 201 | 0.0426 |
| actor_reference | 8 599 | 0.0181 |
| residual | 0 | 0 |

**Quality** (full-frame): VMAF 77.417, Y-PSNR 33.003 dB, SSIM 0.967.

Late-frame: BP45 `late_frame_quality_change` is last-minus-first **per scene**
(VMAF [−25, +8], Y-PSNR [−8, +3] dB). The preflight JSON recorded last-minus-first
on the **concatenated** 96 frames (scene_000 first vs scene_028 last): VMAF
82.70 → 70.25 (Δ −12.45), Y-PSNR 33.62 → 31.83 dB (Δ −1.80). That joined VMAF
delta is inside the numeric band but is **not** the quantity the bound names.
Per-scene rot was not stored on that run. The sweep now scores each scene
separately and applies the BP45 band to those deltas.

**Time:** PointStream `run_seconds=4382.7` (73.0 min). Process wall 5009.6 s (83.5 min), including 4K load. `encode_seconds` / `decode_seconds` are `None` (PointStream wall is `run_seconds`). Resume wall 24.9 s.

**Checkpoint:** identity written at start; `bg-crf51.json` written after the point. Re-running the same command printed `resume bg-crf51`, did not print `pointstream_e1 start`, reused 474 313 B / VMAF 77.417, exit 0 in 25 s. A crash *during* the 73 min encode would still lose the point: per-point checkpoints cannot resume an interrupted encoder subprocess.

**Hourly budget:** this 48-frame native point does **not** finish inside 60 minutes.
Accepting lost work on a mid-encode crash conflicted with the hourly-checkpoint
rule. The runner now writes a durable checkpoint after each *chunk* (scene),
prints stage timings, and emits a still-running line at least every ten minutes
while a stage is blocked. A killed ffmpeg subprocess still cannot resume
mid-bitstream. These changes alone do not prove the hourly budget. The follow-up
in `BP48-recovery-validation.md` saves preparation, verifies actual interrupted
scene recovery, preserves quality and accumulates attempt time. A new native
run must check its recorded maximum checkpoint gap before batch expansion.

## Commands

```
PYTHONPATH=$PWD PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  conda run -n pointstream --no-capture-output \
  python -m experiments.tier.low_rate_sweep \
    --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48 \
    --codec av1 --point bg-crf51 --skip-compare --skip-fallback \
    --out …/bp47-e1-preflight/preflight-alcaraz_highlights-scene_000+scene_028-n48-av1.json
```

`--point` unit tests: 4 passed. Logs: `preflight.log`, `preflight-resume.log`.

## What this licenses

The native two-scene 4K PointStream path delivers the right shape, hashes the source, writes a ledger that sums, scores usable VMAF/PSNR/SSIM, and resumes from a per-point checkpoint. This historical run does NOT authorize broad E1 expansion. First pass the native recovery-budget and slowest-preset reference pilot gates in `BP48-recovery-validation.md`, using new output directories.

It does **not** license a PointStream vs AV1/VVC ranking, a Gate-A decision, or E1 evidence. Fallback-equivalence and both access-pattern reference curves were not run.

## Next

Independent AV1/VVC 48-frame native curves on the same source hashes, starting
with a slowest-preset runtime/recovery pilot, then the staged sweep. Do not
treat a killed libaom subprocess as resumable; do treat a finished scene as
resumable.
