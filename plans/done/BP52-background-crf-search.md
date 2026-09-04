# BP52 — bounded background CRF search

Archived 4 September 2026 after review and integration. Historical instructions
below are not a current dispatch; follow PLAN.md and the active plan index.

## Outcome

Review repair (after measurement): the batch saves a partial report and stops
before the next encode on any point alarm or CRF51 size/quality control mismatch.
Missing control data also stops the batch. Host timing variation is not a quality
regression; the comparison now reads run_seconds from the correct field.
Targeted checks: 43 tests passed, lint and mypy passed. These are driver repairs,
not a rerun: measured outputs and their original implementation remain unchanged.
The new implementation requires a new output directory.

**Complete.** The exact BP49 pair was rerun at native 3840×2160 with a fresh
`bg-crf51` control, then `bg-crf63` and `bg-crf57`. All three PointStream
points succeeded. The fresh CRF51 ledger and quality are byte/metric identical
to BP49. Stronger background quantization moves the live background payload,
but neither tested point beats the independent AV1 QP63 anchor; VVC QP51/QP39
bracket the candidate quality range without licensing a BD-rate claim.

The independent reference bracket also completed: AV1 QP63 and VVC QP63,
QP51, QP39, both continuous and segmented. No point timed out or failed.
This report stops the batch for Codex review. No downscaling, longer scene,
full ladder, training, confirmation, paper edit, or extra VVC point was run.

Implementation commits: `29b1c3d`, `128090c`.  
PR: [#58](https://github.com/emanuele-artioli/PointStream/pull/58).

## Scope and identity

- Base: `origin/main` at merged PR #55,
  `35a57163bd80fb259f68aae2241e495b27c6cf6b`.
- Worktree: `/home/itec/emanuele/pointstream-bp52`.
- Branch: `cursor/bp52-background-crf-search`.
- Data root:
  `/home/itec/emanuele/pointstream-data`.
- Input: `alcaraz_highlights`, `scene_000` and `scene_028`, 48 frames each,
  native 3840×2160 RGB, 24 fps, shared context
  `alcaraz_highlights_main_court`.
- Full decoded RGB SHA-256:
  - `scene_000`:
    `388665774c91f980c3bf0e329d6f4e3bd7123398e99e9192854540723cc60fd6`
  - `scene_028`:
    `e2491f5772cab6d89bd8f32af5d691e97dcde1df3a060aa831f9c7a2371d9aeb`
- Both source arrays were `(48, 2160, 3840, 3)` and the concatenated run was
  exactly 96 frames.
- The BP46 manifest records were snapshotted in the output identity:
  `selected_scene_records_sha256 =
  840c298776ededa1ff5786be3be299ea24968cf754e3aacbf747541ecb2cb2d6`.
- Measured implementation digest:
  `d1461a34f91befba45f371a477b4197e87f5719b6c7b776cef181fbd70285a79`.
- The implementation was linted, typed, tested, and committed before bounds,
  controls, or encodes were measured.

## Fixed configuration

Every PointStream point used:

- `panorama-stream`, canonical canvas, shared context, full-resolution
  delivery.
- Background `libaom-av1` through `/opt/local/bin/ffmpeg`,
  `ffmpeg version n7.1.1-56-gc2184b65d2`, with `-cpu-used 8`, `-usage realtime`,
  `-lag-in-frames 0`, `-bf 0`.
- Injected BP49 objects, appearance JPEG quality 40/downscale 2, motion
  `max_points=16`.
- Generation off and residual off.
- The only changed field was background stream CRF: 51, 63, 57, in that order.

Independent references used `/opt/local/bin/SvtAv1EncApp` v1.8.0 preset 0 for
AV1 and `/opt/local/bin/ffmpeg` `libvvenc` preset `slower` for VVC. The live
VVC probe rejected `placebo` and `veryslow`; `slower` was the verified
slowest supported preset.

## Pre-result bounds and controls

Bounds were written at `2026-09-03T14:46:05Z`, before controls and points, in
`outputs/bp52-background-crf/bounds-before-run.json`.

- CRF51 carried BP49's bands: coded bytes 80,000–50,000,000; VMAF 15–97;
  Y-PSNR 16–45 dB; SSIM 0.72–0.995; run time 30–10,800 s.
- CRF63 and CRF57 used the independent diagnostic bands: positive coded bytes
  below 50 MB; VMAF 0–98; Y-PSNR 8–45 dB; SSIM 0–1.
- Scene-local last-minus-first bands were VMAF [-25,+8] and Y-PSNR [-8,+3] dB.
- Exact frame count, dimensions, source hashes, and ledger equality were hard
  invariants. A joined-scene late-frame delta was diagnostic only.

Fresh controls used the existing
`experiments.tier.calibrate.anchors` fixture on the first two exact BP52
frames, at `(2,2160,3840,3)`. The unrelated natural-content fixture was
`sinner_alcaraz/scene_001`. All 12 control scores were produced:

- PSNR: identical `inf`, mild blur `41.3583`, severe blur `24.04078`,
  unrelated `12.30492`.
- SSIM: identical `1.0`, mild blur `0.99526`, severe blur `0.85710`,
  unrelated `0.67014`.
- VMAF: identical `97.54028`, mild blur `84.96034`, severe blur `0.0`,
  unrelated `0.0`.

The required checks held for every higher-is-better metric:
`identical > mild-blur > severe-blur` and `mild-blur > unrelated-clip`.
VMAF's absolute anchors also held: identical in [95,99], unrelated in [0,40].
No ordering between severe and unrelated was assumed. The complete control is
`outputs/bp52-background-crf/metric-controls.json`.

## PointStream results

All three points were submitted once, succeeded once, and failed zero times.
All had `usable=true`, `is_rate=true`, exact 96-frame delivery, no recovery
alarm, and no scene-local late-frame alarm.

### `bg-crf51`

- Size: total **474,313 B** =
  panorama **445,513 B** + actor reference **8,599 B** +
  metadata **20,201 B** + residual **0 B**.
- Background scene payloads: scene_000 **289,989 B**, scene_028 **155,524 B**.
- Full-frame quality: VMAF **77.417052**, Y-PSNR **33.003064 dB**,
  SSIM **0.96694254**.
- Scene-local late-frame deltas:
  - scene_000: VMAF **+0.962**, Y-PSNR **+1.028 dB**.
  - scene_028: VMAF **+6.135**, Y-PSNR **−0.303 dB**.
- Timing: attempt wall **4,054.732 s**; runner invocation
  **3,419.361 s**; preparation **0.857 s**; assembly/scoring
  **1,403.880 s**.
- Stage timing by scene, in seconds:
  - scene_000: background 82.799, codec 319.629, metrics 234.472,
    finish/scoring 329.537.
  - scene_028: background 83.754, codec 315.493, metrics 231.487,
    finish/scoring 324.708.
- Largest durable checkpoint gap: **1,403.880 s**. Hourly budget held.
- `encode_seconds` and `decode_seconds` are intentionally null for this
  PointStream semantic path; `run_seconds` includes reconstruction and scoring.
  The stage and phase times above must not be relabelled as a codec encode.

The fresh CRF51 output exactly matches BP49's recorded 474,313 B,
VMAF 77.417052, Y-PSNR 33.003064 dB, and SSIM 0.96694254. The fresh runner
time was 3,419.361 s versus BP49's 3,669.413 s; the main difference is
assembly/scoring (1,403.880 s versus 1,664.026 s), while all size, quality,
source, configuration, and checkpoint invariants match. This is recorded as
host timing variation, not hidden as a new quality result.

### `bg-crf63`

- Size: total **212,440 B** =
  panorama **183,640 B** + actor reference **8,599 B** +
  metadata **20,201 B** + residual **0 B**.
- Background scene payloads: scene_000 **123,492 B**, scene_028 **60,148 B**.
- Full-frame quality: VMAF **65.090965**, Y-PSNR **30.813146 dB**,
  SSIM **0.93956073**.
- Scene-local late-frame deltas:
  - scene_000: VMAF **+0.782**, Y-PSNR **+0.856 dB**.
  - scene_028: VMAF **+3.511**, Y-PSNR **−0.180 dB**.
- Timing: attempt wall **4,036.074 s**; runner invocation
  **3,408.278 s**; preparation **2.488 s**; assembly/scoring
  **1,433.953 s**.
- Stage timing by scene, in seconds:
  - scene_000: background 79.320, codec 312.939, metrics 231.207,
    finish/scoring 328.378.
  - scene_028: background 82.496, codec 311.018, metrics 230.409,
    finish/scoring 327.431.
- Largest durable checkpoint gap: **1,433.956 s**. Hourly budget held.

### `bg-crf57`

- Size: total **352,881 B** =
  panorama **324,081 B** + actor reference **8,599 B** +
  metadata **20,201 B** + residual **0 B**.
- Background scene payloads: scene_000 **218,891 B**, scene_028 **105,190 B**.
- Full-frame quality: VMAF **72.996910**, Y-PSNR **32.244682 dB**,
  SSIM **0.95679166**.
- Scene-local late-frame deltas:
  - scene_000: VMAF **+0.901**, Y-PSNR **+1.000 dB**.
  - scene_028: VMAF **+5.055**, Y-PSNR **−0.279 dB**.
- Timing: attempt wall **3,973.922 s**; runner invocation
  **3,345.301 s**; preparation **3.702 s**; assembly/scoring
  **1,352.685 s**.
- Stage timing by scene, in seconds:
  - scene_000: background 79.624, codec 312.383, metrics 232.969,
    finish/scoring 326.995.
  - scene_028: background 82.805, codec 314.266, metrics 231.949,
    finish/scoring 329.461.
- Largest durable checkpoint gap: **1,352.689 s**. Hourly budget held.

### CRF wiring check

The live background payload vectors were distinct:

- CRF51: `[289989, 155524]` B.
- CRF63: `[123492, 60148]` B.
- CRF57: `[218891, 105190]` B.

The decoded background plate SHA-256 vectors were also distinct for all three
CRFs. Each decoded plate had shape `(2276,4120,3)`. Therefore the requested
CRF reached the live `libaom-av1` command and changed both transmitted bytes
and decoded content; this is not an accepted-but-ignored flag.

## Independent reference bracketing

Four codec/QP settings were used, below the five-setting cap. Each setting was
run in both continuous and segmented access patterns, for eight successful
pattern results. Continuous is the comparison arm; segmented is retained as
an access-pattern diagnostic. No equal-QP/equal-quality assumption was made.

For each result below, the tuple is:
`bytes; VMAF; Y-PSNR dB; SSIM; codec encode s; decoder s; pattern wall s`.

- AV1 preset 0, QP63:
  - continuous: `109,198; 82.813572; 36.283943; 0.971227; 97.741; 8.924;
    699.6`.
  - segmented: `129,081; 85.644973; 37.158532; 0.975451; 102.992; 10.192;
    701.4`.
- VVC preset slower, QP63:
  - continuous: `16,270; 6.832282; 24.386269; 0.841844; 99.691; 13.121;
    704.4`.
  - segmented: `16,445; 8.296836; 24.339570; 0.840993; 93.633; 15.753;
    691.1`.
- VVC preset slower, QP51:
  - continuous: `63,801; 63.112704; 31.561492; 0.933538; 119.572; 14.156;
    722.5`.
  - segmented: `63,896; 62.261675; 31.412967; 0.932210; 119.196; 16.110;
    725.3`.
- VVC preset slower, QP39:
  - continuous: `223,734; 88.010253; 38.435819; 0.971969; 422.262; 13.604;
    1047.6`.
  - segmented: `224,676; 87.419004; 38.198345; 0.971576; 417.570; 15.939;
    1041.8`.

The VVC QP51 results remained below every PointStream candidate, so QP39 was
the prescribed adaptive move. QP39 exceeded every candidate, establishing a
bracket. No fourth VVC point was needed. The AV1 arm has one permitted point,
so no AV1 curve or BD-rate was fabricated.

## Timing, budget, failures and checkpoints

- PointStream stage: started `2026-09-03T14:46:05Z`, ended
  `2026-09-03T18:08:19Z`; watchdog exit 0.
- Reference stage: started `2026-09-03T18:08:43Z`, ended
  `2026-09-03T19:20:03Z`; exit 0.
- Adaptive QP39 stage: started `2026-09-03T19:20:18Z`, ended
  `2026-09-03T19:55:41Z`; exit 0.
- Total wall allocation from first start through final aggregate:
  approximately **18,576 s (5 h 10 min)** of the **28,800 s (8 h)** cap.
- All PointStream runner invocations had one attempt, durable scene checkpoints,
  ten-minute heartbeats, and maximum gaps below one hour.
- All reference points had usable decodes, 96 frames, and native dimensions.
- No codec timeout, TERM/KILL escalation, retry, OOM, or failed point occurred.
- Execution failure recorded: the first PointStream wrapper attempted to `tee`
  into `outputs/bp52-background-crf/run.log` before the driver had created the
  directory, so `tee` reported “No such file or directory” and did not retain
  that wrapper log. The Python measurement continued successfully; the durable
  bounds, controls, aggregate, per-point, and per-reference JSON files are
  complete, and the terminal output recorded the run progress. The reference
  wrapper log was created normally at
  `outputs/bp52-background-crf/references/reference-stage.log`.

## Outputs and commands

Primary outputs:

- `outputs/bp52-background-crf/bounds-before-run.json`
- `outputs/bp52-background-crf/metric-controls.json`
- `outputs/bp52-background-crf/background-search.json`
- `outputs/bp52-background-crf/points/`
- `outputs/bp52-background-crf/references/references-alcaraz_highlights-scene_000+scene_028-n48-av1.json`
- `outputs/bp52-background-crf/references/references-alcaraz_highlights-scene_000+scene_028-n48-vvc.json`
- `outputs/bp52-background-crf/references/*points/`
- `outputs/bp52-background-crf/references/reference-stage.log`

PointStream batch command:

```text
PYTHONPATH=/home/itec/emanuele/pointstream-bp52 \
PS_DATA_ROOT=/home/itec/emanuele/pointstream-data \
python -m experiments.tier.bp52_background_search
```

The measured watchdog wrapper was:

```text
timeout --signal=TERM --kill-after=120 28800s \
  conda run -n pointstream --no-capture-output \
  python -m experiments.tier.bp52_background_search
```

Initial reference command:

```text
python -m experiments.tier.low_rate_references \
  --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48 \
  --codec av1 --qp 63 \
  --out-dir /home/itec/emanuele/pointstream-data/outputs/bp52-background-crf/references
python -m experiments.tier.low_rate_references \
  --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48 \
  --codec vvc --qp 63 51 \
  --out-dir /home/itec/emanuele/pointstream-data/outputs/bp52-background-crf/references
```

Adaptive reference command and full aggregate rewrite:

```text
python -m experiments.tier.low_rate_references \
  --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48 \
  --codec vvc --qp 39 \
  --out-dir /home/itec/emanuele/pointstream-data/outputs/bp52-background-crf/references
python -m experiments.tier.low_rate_references \
  --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48 \
  --codec vvc --qp 63 51 39 \
  --out-dir /home/itec/emanuele/pointstream-data/outputs/bp52-background-crf/references
```

Historical BP49 output was read only and remains unchanged at
`outputs/bp49-native-recovery/`.

## Licensed conclusion and next decision

CRF63 reduces total PointStream bytes by **55.2%** relative to CRF51 but loses
**12.326 VMAF**. CRF57 reduces bytes by **25.6%** and loses **4.420 VMAF**.
The background remains the dominant measured category: 86.4% of total bytes at
CRF63, 91.8% at CRF57, and 93.9% at CRF51.

This batch licenses only the finding that background CRF is a live,
rate-bearing axis with a steep rate-quality tradeoff on this diagnostic pair.
It does not license an AV1 win, a VVC win, a BD-rate, an E1/Gate A claim, a
generalization claim, or a real-time claim. One sequence is one independent
experimental unit (`n=1`); no significance or generalization statistic is
reported.

The next plausible lever is background representation/resolution or encoder
effort, because quantization alone trades away quality while the background
still dominates the wire cost. Longer-context amortization is a separate
untested hypothesis. None of these alternatives is implemented here.

**Dependency:** stop this workstream and return to Codex for review of the
complete report, bounded reference bracket, negative result, and next lever.
