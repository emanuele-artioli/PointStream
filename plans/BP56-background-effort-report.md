# BP56 — background encoder-effort report

## Outcome

**Complete** for the authorized three-point diagnostic. No fourth point was
run. Codex review is required before any broader curve or longer context.

Prefix stability and independent-client reconstruction held for libaom
`-usage good -cpu-used 4 -lag-in-frames 0 -bf 0` on synthetic 2/3/4-frame
textured, static and translated plates. The option changed both the ffmpeg
command and the emitted bytes.

The native realtime CRF51 control reproduced BP52/BP53 quality and
panorama/actor bytes exactly (metadata = BP52 + 56 geometry-header bytes).
At the same CRF51, good/cpu-used 4 **spent more background bits** (560,097 vs
445,513 panorama bytes) and raised full-frame VMAF (83.187 vs 77.417). CRF63
at the same effort is smaller than the control and slightly higher VMAF than
the control. That is a PointStream-internal diagnostic on one pair (n=1), not
a win against AV1/VVC, not BD-rate, and not confirmation.

BP52 continuous AV1 QP63 / VVC QP51 / VVC QP39 are **unranked** for this run:
source hashes, ffmpeg path/version, reference preset `0` and `yuv420p` matched,
but `experiments/tier/low_rate_measure.py` differs from `origin/main` (provenance
fields only). The brief forbids ranking when that check fails. No extra
reference encodes were run.

## Isolation

- Base: `origin/main` `a59934a` (PR #62 closeout).
- Worktree: `/home/itec/emanuele/pointstream-bp56`
- Branch: `cursor/bp56-background-encoder-effort`
- Implementation freeze: `6ff9936`
- Measured implementation digest:
  `243ec17937e1a1fbbb1587d2949e5e3d88b016e46a7434becd09ef6aa61c90f7`
- Checkpoint identity fingerprint:
  `ee186529f1d8732354ea1878bf6452a50d463f648d76899120801159def1fe86`
- Data root: `/home/itec/emanuele/pointstream-data`
- PYTHONPATH: `/home/itec/emanuele/pointstream-bp56`
- Output: `outputs/bp56-background-effort/` (never wrote BP49–BP53)

`points/identity.json` was not overwritten after the freeze.

## Tests

Required behaviour (brief Gate 1): default realtime flags unchanged and
`CODECS` not rewritten; candidate keeps lag/bf at 0; option changes command
and bytes; independent 2/3/4-frame prefixes; last-reference, same-size reset,
byte-only client vs encoder reconstruction; legacy resume without effort keys;
changed-option resume refused; non-av1 / non-stream effort refused.

Commands (worktree root, `conda run -n pointstream --no-capture-output`):

- `ruff check` on touched files: passed
- `mypy --config-file pyproject.toml` on touched files: passed
- `python -m src.contracts.layers`: Import direction OK
- unit: `pytest tests/components/background/test_encoder_effort.py tests/experiments/test_bp56_budget.py tests/contracts/test_config.py tests/components/background/test_transport_scale.py tests/experiments/test_low_rate.py tests/components/test_background.py`: 92 passed (integration deselected by default)
- integration: `pytest tests/components/background/test_encoder_effort.py tests/components/test_background_stream.py tests/components/background/test_transport_scale.py -m integration`: 25 passed

## Inputs and tools

Same BP52 pair: `alcaraz_highlights` `scene_000` + `scene_028`, 48 frames,
native 3840×2160, 24 fps, context `alcaraz_highlights_main_court`. RGB SHA-256
`388665774c91f980c3bf0e329d6f4e3bd7123398e99e9192854540723cc60fd6` and
`e2491f5772cab6d89bd8f32af5d691e97dcde1df3a060aa831f9c7a2371d9aeb`.
Manifest selected-record SHA-256
`840c298776ededa1ff5786be3be299ea24968cf754e3aacbf747541ecb2cb2d6`.

ffmpeg `/opt/local/bin/ffmpeg` `n7.1.1-56-gc2184b65d2`, libaom-av1.
Default: `-cpu-used 8 -usage realtime -lag-in-frames 0 -bf 0`.
Candidate: `-cpu-used 4 -usage good -lag-in-frames 0 -bf 0`.
AV1 residual/reference preset remains probe `0` (not SVT-AV1 for the
background stream). Generation off, residual off, transport scale 1.0,
reference mode last, canonical canvas.

## Prefix / client proof

Written `2026-09-04T12:59:21Z` at
`outputs/bp56-background-effort/prefix-proof.json` before native points.
Probe supported: command contains `-usage good` and `-cpu-used 4`.

Independent prefix payload sizes (bytes), candidate effort, CRF51:

| family | 2-frame | 3-frame | 4-frame |
|---|---|---|---|
| textured | 257, 81 | 257, 81, 66 | 257, 81, 66, 62 |
| static | 257, 22 | 257, 22, 22 | 257, 22, 22, 22 |
| translated | 257, 60 | 257, 60, 78 | 257, 60, 78, 57 |

Already-emitted packets are identical as the independent encode grows.
Integration tests also drove last-reference push, a same-size context reset
(new keyframe), and a receiver that holds only payload bytes; reconstructions
matched the encoder.

## Bounds and controls (before points)

Written `2026-09-04T12:59:21Z` at
`outputs/bp56-background-effort/bounds-before-run.json`.

- Control realtime CRF51: BP52 bands plus exact quality / panorama / actor
  match; metadata = BP52 + 2×28 B geometry headers.
- Candidates: VMAF 0–98, Y-PSNR 8–45 dB, SSIM 0–1, coded bytes in (0, 50 MB].
- Scene-local late-frame: VMAF [−25, +8], Y-PSNR [−8, +3] dB.

Metric controls on the first two native frames, fixture
`experiments.tier.calibrate.anchors`, unrelated `sinner_alcaraz/scene_001`:
identical > mild > severe and mild > unrelated held. VMAF identical 97.54028
in [95,99], unrelated 0.0 in [0,40]. File:
`outputs/bp56-background-effort/metric-controls.json`. Controls wall 50.7 s.

## Batch

Log directory existed before tee:
`outputs/bp56-background-effort/logs/job.log`.
Launched detached `2026-09-04T12:59:19Z`, exit `0` at `2026-09-04T16:24:39Z`.

| count | n |
|---|---|
| submitted | 3 |
| succeeded | 3 |
| failed | 0 |

All `usable=true`, `is_rate=true`, 96 frames, ledger balanced, recovery alarm
null, hourly checkpoint budget met. Max durable gaps: 1479.7 s, 1310.2 s,
1285.5 s (not cleared for longer runs: gap is inside the hourly limit by more
than 1 s, but the operational flag stays false as required).

Unknown crash interval: false. Consumed seconds 12,303.3 of 28,800 (~3.4 h),
not a lower bound. No restart.

**Attempt-wall vs 1-hour encode cap.** Each native attempt wall was 4329.8 s,
3975.7 s, 3947.2 s. Assembly/scoring was 1479.7 / 1310.2 / 1285.5 s of that.
libaom subprocess timeout was `min(3600, remaining−900)` s; no ffmpeg timeout
fired. Runner encode stages were minutes, not an hour. The one-hour cap was
applied to the encoder subprocess, not to scoring. Full attempt walls still
exceed 3600 s, as BP53 scoring already did. Longer runs are not cleared.

## Size, quality, time

Geometry header 28 B/scene is in **metadata**, not the AV1 payload.
`encode_seconds` / `decode_seconds`: **null** (runner wall includes
reconstruction and scoring; BP55 owns the split). Do not infer them by
subtracting metrics. Background stage walls below are the panorama-stream
stage, not a semantic encoder clock.

### `bg-realtime8-crf51` (control)

- Bytes: **474,369** = panorama 445,513 + actor 8,599 + metadata 20,257 + residual 0
- Background payloads: 289,989 + 155,524 B (same as BP52/BP53 scale-1 CRF51)
- VMAF **77.417052**, Y-PSNR **33.003064 dB**, SSIM **0.96694254**
- Late-frame: scene_000 VMAF +0.962, Y-PSNR +1.028 dB; scene_028 VMAF +6.135,
  Y-PSNR −0.303 dB
- Attempt wall 4,329.8 s; runner 3,683.8 s; preparation 1.0 s;
  assembly/scoring 1,479.7 s
- Background stages 166.5 s / 84.7 s; codec 363.9 s / 337.7 s

### `bg-good4-crf51`

- Bytes: **588,953** = panorama 560,097 + actor 8,599 + metadata 20,257 + residual 0
- Background payloads: 399,993 + 160,104 B (changed vs control; option used)
- VMAF **83.187126**, Y-PSNR **33.888881 dB**, SSIM **0.98140107**
- Late-frame: scene_000 VMAF +1.005, Y-PSNR +1.080 dB; scene_028 VMAF +7.798,
  Y-PSNR −0.272 dB
- Attempt wall 3,975.7 s; runner 3,326.2 s; preparation 2.5 s;
  assembly/scoring 1,310.2 s
- Background stages 91.4 s / 102.0 s; codec 316.6 s / 315.3 s

### `bg-good4-crf63`

- Bytes: **377,360** = panorama 348,504 + actor 8,599 + metadata 20,257 + residual 0
- Background payloads: 268,949 + 79,555 B
- VMAF **79.338914**, Y-PSNR **33.259160 dB**, SSIM **0.97396001**
- Late-frame: scene_000 VMAF +0.977, Y-PSNR +1.059 dB; scene_028 VMAF +6.293,
  Y-PSNR −0.369 dB
- Attempt wall 3,947.2 s; runner 3,318.1 s; preparation 3.7 s;
  assembly/scoring 1,285.5 s
- Background stages 92.2 s / 97.7 s; codec 321.3 s / 315.8 s

Decoded plates remain canonical `(2276, 4120, 3)`. Actor bytes are unchanged
across points (8,599). Payload SHA-256s differ across all three points.

## What this does and does not license

Licensed: prefix-stable higher-effort plumbing; control regression on this
pair; a same-CRF effort change that is not free (more bits, higher VMAF);
a coarser CRF at that effort that is smaller than the realtime CRF51 control
with a modest VMAF gain vs that control; all three on one diagnostic pair.

Not licensed: beating AV1 or VVC; BD-rate; statistical generalization; a
speed claim; slowest-background / SVT-AV1 comparison; longer-context
clearance; ranking the historical BP52 reference JSONs against this identity.

## Next

Return to Codex. Do not launch a preset grid, a fourth point, or longer
native contexts from this result. Open axes remain longer context and
foreground/background quality allocation. If Codex wants a ranked comparison,
re-freeze with metric-file identity matching the cited reference run, or cite
the provenance delta explicitly as a documentation-only change.

## Paths and reproduction

- Aggregate: `outputs/bp56-background-effort/background-effort.json`
- Checkpoints: `outputs/bp56-background-effort/points/`
- Log: `outputs/bp56-background-effort/logs/job.log`
- Bounds: `outputs/bp56-background-effort/bounds-before-run.json`
- Prefix proof: `outputs/bp56-background-effort/prefix-proof.json`

```
PYTHONPATH=/home/itec/emanuele/pointstream-bp56 \
PS_DATA_ROOT=/home/itec/emanuele/pointstream-data \
conda run -n pointstream --no-capture-output \
python -m experiments.tier.bp56_background_effort
```
