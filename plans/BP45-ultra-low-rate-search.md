# BP45 — Ultra-low-rate AV1/VVC search

**Current state (3 September):** M1 and the E1 harness are merged. Do not repeat
their implementation. BP49 and BP52 diagnostics are complete. Next is the
bounded BP53 background-scale test, then Codex review before broad E1; no
size–quality win has been established.

**Roadmap IDs:** M1 then E1  
**Preferred harness:** Cursor implements and validates; a detached batch runs the
sweep; Codex/Claude approves bounds and interprets results.  
**Outcome:** determine whether PointStream wins on rate--quality in the low-rate,
long-eligible-scene regime.

## Read

`AGENTS.md`, `plans/ROADMAP.md` §§2–3 and 8,
`plans/TERMINOLOGY.md`, `plans/done/BP31-findings.md`,
`plans/done/BP35-perceptual-bdrate.md`, the codec contracts, and the current paired
ladder. Do not read unrelated briefs.

## Owns

- metric direction/curve handling under `src/components/metrics/`
- focused metric tests
- `experiments/tier/low_rate_*.py`
- `outputs/bp45-low-rate/**`

Do not modify background geometry; BP44 owns it.

## Part 1 — instrument calibration

1. Add explicit metric semantics: name, units, higher/lower-is-better, valid
   range and transform used by curve integration.
2. Re-run identical, mild, severe and unrelated anchors for every headline
   metric.
3. Resolve and record AV1/VVC binary paths, versions and presets.
4. Select each installed encoder's slowest valid preset or full reference
   configuration for the primary comparison. Do not substitute a faster preset
   to make runtime convenient.
5. Probe each encoder's full legal quality range at fixed source resolution and
   frame rate.
6. Reject empty, undecodable, wrong-size, wrong-frame-count or non-monotone
   outputs.
7. Record the smallest valid bitstream and its measured quality.

Primary quality is full-frame VMAF. Y-PSNR and SSIM/MS-SSIM are secondary.
LPIPS remains diagnostic until its direction and scale are calibrated.

## Part 2 — coherent PointStream sweep

Start generation off. Sweep the following in a small staged search rather than
one Cartesian explosion:

1. background resolution × background quality;
2. correction off/coarse × background setting;
3. appearance resolution/quality/refresh;
4. motion density/precision;
5. add crossover-neighbour points to make valid curves.

Every operating point records the complete transmitted-byte ledger. Assert that
the intended rate-bearing categories move or explain why they do not.

Run AV1 and VVC on the exact same frames, dimensions, frame rate and colour
convention. They encode each whole sequence jointly. No frame dropping or
downscaling is allowed in the main comparison.

Run both access-pattern controls:

- **segmented:** each point scene is a separate independently decodable segment
  and each codec pays the required intra frame;
- **continuous:** the same ordered eligible scenes are encoded together and all
  codecs may reuse references across every boundary allowed to PointStream.

The headline control follows the product claim. Report the other as an
access/random-access tradeoff. Do not credit PointStream with avoiding repeated
intra frames while denying AV1/VVC equivalent cross-boundary prediction. If
PointStream is faster than the slowest-preset references, add faster reference
presets after Gate A and report the three-way rate--quality--time frontier.

## Bounds and controls

Write the bounds file before the first system encode. Use a deliberately broad
two-sided BD-rate interval because the experiment tests the regime:

- PointStream vs each anchor on VMAF: [−80%, +180%];
- no codec point may decode to zero/wrong frames;
- identical-input metric anchors must remain at their calibrated ceiling;
- late-frame VMAF and Y-PSNR changes must be bounded separately;
- the conventional fallback control must reproduce the reference codec result;
- an object-stream-off control must show whether any win comes from background
  reuse alone.

A negative result triggers an extra validation pass. A non-overlapping curve is
not assigned a BD-rate; test strict dominance at the lowest anchor point instead.

## Search disclosure

Persist every tried configuration, not just the winner. The report must identify
the axis that selected the final regime and the date the criterion was frozen.
E1 is diagnostic on two videos. It becomes evidence only after the frozen rule
passes E2 on at least six independent videos/matches.

## Completion report

Follow `plans/SESSION-REPORT.md`. Include calibration, codec floors, every
curve and overlap, size/quality/encode/decode time, byte ledgers, alarms, failed
points, exact commands and the Gate-A decision.

**Current integration status:** `done/BP47-integration.md` supersedes the "Next"
instructions and completion wording in the historical reports below.

## Delivered (M1, 2026-09-02)

**Outcome: complete for M1.** Quality-axis typing and the ultra-low AV1/VVC
probe are done. Gate A is not decided. E1's *encode wave* was not launched.

**PR:** [#51](https://github.com/emanuele-artioli/PointStream/pull/51) on
`cursor/m1-bp45` (`7602197` M1, `ecaf8de` E1 harness). CI green. Worktree:
`/home/itec/emanuele/pointstream/.claude/worktrees/bp45`, from `origin/main`
(`ecebd9b`). Named report: `plans/done/BP45-m1-report.md`.

**Owns (landed):** metric direction/transform/span on `MetricSpec`;
`compare_rd_curves` / `meets_or_beats_floor`; `experiments/tier/low_rate_*.py`;
outputs under `/home/itec/emanuele/pointstream-data/outputs/bp45-low-rate/`.
Background geometry was not touched.

**Calibration** (2 frames, 3840×2160; unrelated = `sinner_alcaraz/scene_001`):

| Metric | identical | mild | severe | unrelated |
|---|---:|---:|---:|---:|
| VMAF | 97.540 | 84.960 | 0.0 | 0.0 |
| PSNR (dB) | inf | 41.358 | 24.041 | 12.305 |
| SSIM | 1.0 | 0.995 | 0.857 | 0.670 |
| LPIPS | 0.0 | 0.017 | 0.298 | 0.549 |

**Codec floors** (native 4K, 24 fps, 2 frames `alcaraz_highlights/scene_000`,
10/10 usable, 0 alarms). Slowest *valid* presets: AV1
`/opt/local/bin/SvtAv1EncApp` v1.8.0 preset `0`; VVC ffmpeg `libvvenc` n7.1.1
preset `slower` (`placebo`/`veryslow` empty bitstream, exit 234). Smallest
valid: AV1 QP 63 **43,865 B** VMAF **86.53**; VVC QP 63 **2,698 B** VMAF
**10.17**. Finding, not an alarm: two 4K frames do not starve AV1.

**Bounds** written before the first encode:
`outputs/bp45-low-rate/bounds-before-run.json`. VMAF BD-rate vs AV1 and VVC:
[−80%, +180%]. No bound was revised.

**Commands:** `python -m experiments.tier.low_rate_bounds`;
`python -m experiments.tier.calibrate --metrics psnr ssim vmaf lpips`;
`python -m experiments.tier.low_rate_probe --frames 2`. Tests: 65 passed on
the metric + low-rate suite after the E1 integration.

## Delivered (E1 harness + integration, 2026-09-02)

**Outcome: partial.** The search harness is integrated. The five call-site
gaps (canvas, identity, checkpoints, timing label, fallback equivalence) are
closed in code. The 4K long-scene PointStream vs AV1/VVC wave was not run.
Gate A is not taken.

New modules: `low_rate_canvas.py`, `low_rate_identity.py`,
`low_rate_checkpoint.py`, `low_rate_fallback.py`, `low_rate_smoke.py`.

- **Slowest presets:** `primary_preset` reads `codec-floor.json`
  `selected_preset` and refuses `measure.PRESETS` (`av1=10`, `vvc=faster`).
- **Canonical canvas:** `apply_point` sets `background.canvas='canonical'` and
  `run(..., context_ids=)` gets each clip's `context_id`. Missing BP44 APIs
  SystemExit; merging BP44 is not enough by itself.
- **Long-scene loader:** `load_e1_sequence` calls
  `experiments.long_scenes.loader.load_long_scene_clip`. Default 48 frames,
  `alcaraz_highlights` `scene_000`/`scene_028`. Clips without `context_id` are
  refused.
- **Reference identity:** files are
  `references-{video}-{scenes}-n{frames}-{codec}.json`. Load checks video,
  scenes, duration, fps and codec; a mismatch is a SystemExit, not a silent
  BD-rate.
- **Checkpoints:** each reference QP/pattern and each sweep point is one JSON
  under `*.points/`. The aggregate report is rewritten after every point.
- **Metrics:** full-frame VMAF primary, Y-PSNR and SSIM secondary, plus
  last-minus-first VMAF/Y-PSNR for the rot bound.
- **Timing:** codec points keep `encode_seconds` / `decode_seconds`.
  PointStream wall is `run_seconds`; `encode_seconds` stays `None`.
- **Fallback control:** `FallbackConfig` is aligned to the reference QP/preset
  and encoded on the same frames. Bounds
  `fallback_reproduces_reference` (rate_rel 0.95–1.05, VMAF ±1).
- **Smoke:** `python -m experiments.tier.low_rate_smoke` (2×64×64, not 4K).
  Ran 2026-09-02: AV1 fallback 137 B vs reference 137 B, `rate_rel=1.0`,
  `held=True`.

Does not license a PointStream vs AV1/VVC BD-rate, an E1 evidence claim, or a
Gate-A decision. PointStream on 48-frame clips may still fail until B1
(canonical canvas) lands; that is recorded, not a result.

**Next:** merge D1 and B1; then the detached encode:

```
python -m experiments.tier.low_rate_references \
  --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48
python -m experiments.tier.low_rate_sweep \
  --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48
```

Codex: confirm the bounds file and the 2-frame AV1-floor reading before that
wave is treated as evidence.
