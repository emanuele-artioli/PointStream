# BP53 — background transport scale report

## Outcome

**Complete** for the authorized three-point diagnostic. No further points were
run. Codex review is required before any broader curve.

Scale 1.0 CRF51 reproduced BP52 quality and background/actor bytes. The only
size change is **56 charged geometry-header bytes** (28 B × 2 scenes). Half-scale
coding reduced panorama bytes and full-frame quality. Neither half-scale point
beats the cited BP52 continuous AV1 QP63 or VVC QP51/QP39 bracket. This is one
diagnostic pair (n=1), not confirmation and not a BD-rate claim.

A first-pass control used `residual or default`, treated measured residual 0 as
missing, and stopped the batch. The predicate was fixed; the CRF51 encode was
not repeated. Half-scale points then ran.

## Isolation

- Base: `origin/main` `6b2f6c4` (PRs #57/#58/#59).
- Worktree: `/home/itec/emanuele/pointstream-bp53`
- Branch: `cursor/bp53-background-transport-scale`
- Data root: `/home/itec/emanuele/pointstream-data`
- PYTHONPATH: `/home/itec/emanuele/pointstream-bp53`
- BP49/BP52 output trees were not written.

Implementation commits: `2149a00` (encoder/config), `2872a5f` (log-dir guard),
`2ed5c40` (zero-residual control predicate).  
Measured implementation digest (after the control fix):
`9a7b2dbc8cfa2384e116296aca34ba7f2debf7de416545c9212c36df77124d39`.

## Tests

Required behaviour was listed in the brief (scale-1 compatibility, invalid
scales, geometry, encoder/client restore, even rounding, context reset,
snapshot/changed-scale, byte accounting). Tests:
`tests/components/background/test_transport_scale.py`.

Commands (worktree root, `conda run -n pointstream --no-capture-output`):

- `ruff check` on touched files: passed
- `mypy --config-file pyproject.toml` on touched files: passed
- `python -m src.contracts.layers`: Import direction OK
- unit: `pytest tests/components/background/test_transport_scale.py tests/contracts/test_config.py tests/components/test_background.py tests/pipeline/test_reconstruction_background.py`: 52 passed
- integration: `pytest …/test_transport_scale.py -m integration`: 6 passed
- related integration: stream stage, canonical canvas, background stream,
  recovery: 30 passed

## Inputs and tools

Same BP52 pair: `alcaraz_highlights` `scene_000` + `scene_028`, 48 frames, native
3840×2160, 24 fps, context `alcaraz_highlights_main_court`. RGB SHA-256
`38866577…` and `e2491f57…`. Manifest records SHA-256
`840c298776ededa1ff5786be3be299ea24968cf754e3aacbf747541ecb2cb2d6`.

ffmpeg `/opt/local/bin/ffmpeg` `n7.1.1-56-gc2184b65d2`, libaom-av1
`-cpu-used 8 -usage realtime -lag-in-frames 0 -bf 0`. Matches BP52, so
continuous AV1 QP63 and VVC QP51/QP39 from
`outputs/bp52-background-crf/background-search.json` are cited as immutable
prior diagnostics, not copied into this checkpoint identity. No fresh
reference encodes.

## Bounds and controls (before points)

Written `2026-09-03T21:00:40Z` at
`outputs/bp53-background-scale/bounds-before-run.json`.

- Scale 1.0 CRF51: BP52 bands plus exact quality / panorama / actor match;
  metadata = BP52 + 56 B.
- Half scale: VMAF 0–98, Y-PSNR 8–45 dB, SSIM 0–1, coded bytes in (0, 50 MB].
- Scene-local late-frame: VMAF [−25, +8], Y-PSNR [−8, +3] dB.

Metric controls on the first two native frames, fixture
`experiments.tier.calibrate.anchors`, unrelated `sinner_alcaraz/scene_001`:
identical > mild > severe and mild > unrelated held. VMAF identical 97.54028
in [95,99], unrelated 0.0 in [0,40]. File:
`outputs/bp53-background-scale/metric-controls.json`.

## Batch

Log directory existed before tee:
`outputs/bp53-background-scale/logs/job.log`.

| count | n |
|---|---|
| submitted | 3 |
| succeeded | 3 |
| failed | 0 |

All `usable=true`, `is_rate=true`, 96 frames, ledger balanced, recovery alarm
null, hourly checkpoint budget met. Max durable gaps: 1456 s, 3600 s, 1427 s.

False alarm (closed): first stop `residual is not zero` while residual was 0.
Predicate fixed in `2ed5c40`; CRF51 checkpoint reused.

## Size, quality, time

Geometry header 28 B/scene is in **metadata**, not the AV1 payload. Restored
plates are canonical `(2276, 4120, 3)`. Downsample/upsample seconds: **null**
(not a runner stage clock). `encode_seconds`/`decode_seconds`: **null**
(runner wall includes reconstruction and scoring).

### `bg-scale1-crf51`

- Bytes: **474,369** = panorama 445,513 + actor 8,599 + metadata 20,257 + residual 0
- Background payloads: 289,989 + 155,524 B (same as BP52)
- VMAF **77.417052**, Y-PSNR **33.003064 dB**, SSIM **0.96694254**
- Late-frame: scene_000 VMAF +0.962, Y-PSNR +1.028 dB; scene_028 VMAF +6.135,
  Y-PSNR −0.303 dB
- Attempt wall 4,099.4 s; runner 3,450.8 s; preparation 0.870 s;
  assembly/scoring 1,456.0 s
- Background stages 79.3 s / 85.8 s; codec 314.8 s / 314.2 s

### `bg-scale05-crf51`

- Bytes: **221,218** = panorama 192,362 + actor 8,599 + metadata 20,257 + residual 0
- Background payloads: 128,161 + 64,201 B
- VMAF **52.27058**, Y-PSNR **30.607 dB**, SSIM **0.94374**
- Late-frame: scene_000 VMAF +0.774, Y-PSNR +0.661 dB; scene_028 VMAF +3.902,
  Y-PSNR −0.215 dB
- Attempt wall 6,242.2 s; runner 5,603.8 s (assembly/scoring 3,599.7 s)

### `bg-scale05-crf63`

- Bytes: **104,543** = panorama 75,687 + actor 8,599 + metadata 20,257 + residual 0
- Background payloads: 53,464 + 22,223 B
- VMAF **37.906354**, Y-PSNR **28.295 dB**, SSIM **0.91232**
- Late-frame: scene_000 VMAF +0.554, Y-PSNR +0.499 dB; scene_028 VMAF +1.885,
  Y-PSNR +0.015 dB
- Attempt wall 4,039.7 s; runner 3,412.3 s

Half-scale payload reduction is not a quality-matched win. Cited BP52
continuous anchors: AV1 QP63 109,198 B / VMAF 82.81; VVC QP51 63,801 B /
VMAF 63.11; VVC QP39 223,734 B / VMAF 88.01. No overlap adequate for BD-rate.

## Paths and reproduction

- Aggregate: `outputs/bp53-background-scale/background-scale.json`
- Checkpoints: `outputs/bp53-background-scale/points/`
- Log: `outputs/bp53-background-scale/logs/job.log`

```
PYTHONPATH=/home/itec/emanuele/pointstream-bp53 \
PS_DATA_ROOT=/home/itec/emanuele/pointstream-data \
python -m experiments.tier.bp53_background_scale
```

## What this licenses

Geometry and accounting for `transport_scale` 1.0/0.5 under panorama-stream
work on this pair. Scale 1.0 is a BP52 regression control plus explicit
headers. Half-scale is a negative rate–quality diagnostic here.

It does **not** license a winning regime, BD-rate, real-time/speed ranking,
longer contexts, quarter-scale, or slower background presets. Next hypotheses
remain those in the brief: longer contexts and more background encoder effort,
tested separately after Codex review.
