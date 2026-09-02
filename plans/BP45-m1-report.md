# BP45 M1 — session report

**Outcome:** complete for M1 (quality-axis typing and ultra-low anchor probe). E1 not run. Gate A is not decided.

**Roadmap:** M1. E1 remains blocked on B1 (canonical canvas) and D1 (long eligible scenes).

**Branch / worktree:** `cursor/m1-bp45` at `/home/itec/emanuele/pointstream/.claude/worktrees/bp45`, from `origin/main` (`ecebd9b`).

## What landed

Metric specs now carry name, units, direction, range, the BD-rate quality transform (`identity` / `negate`) and a per-metric span floor. `compare_rd_curves` reads those fields off the spec, fits on the higher-is-better axis, and reports overlap in native units. `compare_paired.describe()` names the axis. Calibration uses the contract direction instead of a hardcoded name set.

New harness (does not touch background geometry):

- `experiments/tier/low_rate_validate.py`
- `experiments/tier/low_rate_bounds.py`
- `experiments/tier/low_rate_plan.py` — 11 staged points, not a Cartesian product
- `experiments/tier/low_rate_probe.py`
- `experiments/tier/low_rate_sweep.py` — refuses E1 until B1/D1; `--allow-short-scenes` is diagnostic only

## Commands and tests

```
conda run -n pointstream --no-capture-output python -m pytest \
  tests/contracts/test_metrics.py tests/components/test_bd_rate.py \
  tests/components/test_comparison.py tests/experiments/test_low_rate.py -q
# 51 passed

ruff check <touched files>          # All checks passed
mypy --config-file pyproject.toml <touched src/experiments>  # no issues, 8 files
python -m src.contracts.layers      # Import direction: OK
```

Bounds were written **before** the first encode:

```
python -m experiments.tier.low_rate_bounds
# /home/itec/emanuele/pointstream-data/outputs/bp45-low-rate/bounds-before-run.json

python -m experiments.tier.calibrate --metrics psnr ssim vmaf lpips \
  --out .../bp45-low-rate/metric-calibration.json
# exit 0, 93 s

python -m experiments.tier.low_rate_probe --frames 2 \
  --out .../bp45-low-rate/codec-floor.json
# exit 0, 1499 s, 0 alarms
```

The PointStream sweep was not launched. `python -m experiments.tier.low_rate_sweep` exits until B1/D1, unless `--allow-short-scenes` is passed and labelled diagnostic.

## Encoder identity

| Codec | Binary | Version | Slowest valid preset |
|---|---|---|---|
| AV1 | `/opt/local/bin/SvtAv1EncApp` | SVT-AV1 v1.8.0 (release), `--roi-map-file` present | `0` |
| VVC | `/opt/local/bin/ffmpeg` `libvvenc` | ffmpeg n7.1.1-56-gc2184b65d2 | `slower` (`placebo` and `veryslow` rejected: empty bitstream, exit 234) |

QP ranges probed at native 3840×2160, 24 fps, 2 frames of `alcaraz_highlights/scene_000`. No downscale, no frame drop. Colour: encoder `yuv420p`, quality on the RGB round-trip from `coded_roundtrip`.

## Calibration (identical / mild / severe / unrelated)

2 frames, 3840×2160. Unrelated anchor: `sinner_alcaraz/scene_001`. Every ordering held. Direction came from the metric spec.

| Metric | identical | mild-blur | severe-blur | unrelated-clip |
|---|---:|---:|---:|---:|
| VMAF | 97.540 | 84.960 | 0.0 | 0.0 |
| PSNR (dB) | inf | 41.358 | 24.041 | 12.305 |
| SSIM | 1.0 | 0.995 | 0.857 | 0.670 |
| LPIPS | 0.0 | 0.017 | 0.298 | 0.549 |

VMAF identical 97.54 matches the BP23 4K ceiling (±0.01). Severe and unrelated still floor at 0. LPIPS 4K anchors match BP23. Quote unrelated beside any later score.

## Codec floors

Pre-run expectation (written here before reading the JSON scores, after seeing only “ok” byte counts): both encoders produce monotone rate and quality over their QP walk; AV1’s smallest 4K intra-ish bitstream is tens of kB, not a few hundred bytes; VVC at QP 63 can go much coarser. A VMAF below ~5 or above 99 at any point would be an instrument alarm.

All 10/10 points usable on both codecs. No empty, wrong-size, wrong-count or non-monotone alarms.

**AV1 preset 0** — smallest valid: QP 63, **43,865 B**, VMAF **86.53**, Y-PSNR 37.90 dB, SSIM 0.979, encode+decode 16.1 s.

**VVC preset slower** — smallest valid: QP 63, **2,698 B**, VMAF **10.17**, Y-PSNR 24.44 dB, SSIM 0.847, encode+decode 8.2 s.

VMAF span: AV1 86.53–97.37 (10.8 points, just above the 10-point BD-rate floor). VVC 10.17–97.43.

### Finding, not an alarm: AV1 does not enter the starved-VMAF regime on two 4K frames

Even at legal QP 63, AV1 on this 2-frame 4K clip stays at VMAF 86. That is still high-fidelity. VVC at the same QP reaches VMAF 10. The 2-frame AV1 “floor” is an intra-dominated clip, not the long-scene operating point E1 needs. Ultra-low-rate vs AV1 has to be measured on the long eligible scenes (D1), not on this probe’s two frames. The probe did its job: it established the legal range, the slowest presets, and that both encoders decode cleanly across that range.

Time: AV1 ~16–19 s per point at preset 0. VVC `slower` grows from 8 s at QP 63 to **263 s** at QP 0. Decode time is inside those encode+decode totals; it is not split out on this probe.

## Bounds and alarms

File: `outputs/bp45-low-rate/bounds-before-run.json`

- PointStream vs AV1/VVC VMAF BD-rate: **[−80%, +180%]** (percent, two-sided, search interval)
- Decode must match source frames/size
- Identical-anchor ceilings
- Late-frame VMAF [−25, +8] and Y-PSNR [−8, +3] dB (not exercised: only 2 frames)
- Fallback-reproduces-reference and object-stream-off controls are required on E1, not on this probe

**Alarms:** none on calibration or the codec-floor probe. No bound was revised.

## Gate A

Not taken. No PointStream curve was encoded. A non-overlapping later curve must use `meets_or_beats_floor` at the lowest valid anchor point (AV1: 43,865 B at VMAF 86.53; VVC: 2,698 B at VMAF 10.17 on this 2-frame clip). Those floors are not the E1 floors.

## Claims this does not license

- Any PointStream vs AV1/VVC BD-rate
- That AV1 cannot go below VMAF 86 on long scenes
- That `placebo`/`veryslow` VVC are missing from the standard rather than from this ffmpeg build
- Speed ranking of AV1 vs VVC (shared host, single sample per QP)

## Next

1. B1 canonical canvas (BP44) and D1 long eligible scenes (BP46).
2. Codex: confirm the bounds file and this 2-frame AV1-floor reading.
3. Then E1: `python -m experiments.tier.low_rate_sweep` on the long scenes, same frames/rate/colour to both anchors, continuous and segmented controls, generation off, persist every tried point.

Reproduction: worktree `cursor/m1-bp45`; outputs under `/home/itec/emanuele/pointstream-data/outputs/bp45-low-rate/`.
