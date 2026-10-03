# Foreground smoke implementation status — 3 October 2026

## Current status

This work stopped before live crop review. The first all-host fleet inspection
found gpu3's RTX A6000 idle, but the launch-time check found no eligible GPU.
A subsequent gpu3 probe failed because `gpu3.itec.aau.at` did not resolve.
The dispatcher returned no job ID, so no GPU allocation started and no remote
dataset file or prior result was read or changed. The implementation and local
tests below are reviewable; they are not evidence about the saved crops,
generator quality, runtime, or rate-distortion.

The work is isolated on branch `codex/demo-foreground-smokes` in the Mac
worktree `/private/tmp/pointstream-foreground-smokes`. The original checkout's
dirty dispatcher and unrelated untracked files were left untouched. The six
untracked foreground seed files copied from the main checkout still match the
plan inventory's byte counts and SHA-256 values.

## Changes prepared

- Added `demo/pipeline/foreground_codec_v2.py`: a versioned experimental
  segment packet with fixed-endian metadata, 1/16-pixel full-frame uint16
  coordinates, independent segment decoding, CRC checks, and raw, zlib, and
  exact modular-delta-plus-zlib modes. It rejects malformed geometry, invalid
  lengths, trailing data, bad checksums, and more than two resolved tracks.
- Added causal bbox/center association with separate optional handedness,
  duplicate suppression, short disappearance recovery, and an explicit
  unsupported result for more than two unresolved detections.
- Added a bounded `foreground_smoke.py` audit path. It reads the three pose
  manifests, limits pixel reads to 24 candidates per recording, records
  heuristic geometry/mask diagnostics and file hashes, and writes contact
  sheets with all labels pending human review. It records the clip 3 second-0
  look override, excludes clip 3 seconds 210/240/420, keeps the final 300
  frames out of fit eligibility, and treats factory002 as audit-only. Blocking
  reads have a 90-second timeout and outputs are restricted to `PS_JOB_DIR`.
- Added a reviewed-label split freezer. It selects no more than 16 fit and
  eight validation crops, keeps each `(recording, source_second)` wholly on
  one side of the split, writes a lossless one-frame packet per crop, and
  stops unless it finds at least eight fit and four validation crops labeled
  `visible_hand` by a human.
- Added `smoke_hand_objective` for masked RGB MAE, balanced foreground /
  background alpha MAE, and full-crop black compositing MAE. The objective
  expects original unmatted RGB in `[-1,1]` and alpha in `[0,1]`.
- Added a four-channel pix2pix head with tanh RGB and sigmoid alpha. The
  existing three-channel `HandPix2PixUNet` module/state-dict path remains
  unchanged in source, but model loading and runtime behavior still need
  execution in the project's PyTorch environment.
- Added regression tests for packet bounds and corruption, same-label tracks,
  duplicate candidates, resets and missing hands, crop geometry, source-second
  split leakage, human-label gates, output confinement, and audit sample caps.

The runner exposes `audit`, `packet`, `fit`, `profile`, and `compare`. In this
implementation only audit, packet, and `fit --freeze-only` are enabled. Actual
training, provider profiling, and AV1 comparison remain gated and are not
implemented here; do not call this branch a completed foreground plan.

## Checks and evidence

The local focused command passed **42 tests**:

```bash
python -m pytest -q tests/demo/test_hand_packet_rate.py tests/demo/test_holdout_hand_rd.py tests/demo/test_sam_crop_pose.py tests/demo/test_foreground_contract.py tests/demo/test_foreground_runtime.py
```

`python -m py_compile` passed for the new runner, packet module, modified
objective and model, and all three new test modules. The objective test module
was not executed locally: the available Mac Python is `/Users/manu/miniconda3/bin/python`
and does not have PyTorch installed. This is an environment limitation, not a
passing objective test.

The read-only fleet inspection initially showed gpu3 as an NVIDIA RTX A6000
with no compute process, 1 MiB used, 48,675 MiB free, and 0% utilization.
`fleet launch` rechecked resources but returned “No eligible GPU is available”.
The next read-only gpu3 probe failed with a DNS-resolution error. There is no
job ID, output directory, contact sheet, audit manifest from the real dataset,
or GPU-time charge to report.

## Gate ledger

| Gate | Status | Evidence / reason |
|---|---|---|
| F1 — live crop audit and human labels | **Blocked** | No remote audit job started; no contact sheet is available for visual review. The local fixture test verifies the cap and pending-label behavior only. |
| F2 — shared packet / scorer contract | **Partial** | New codec and association unit tests pass. Existing AV1, generator, and metrics paths have not yet been wired to this common manifest, and no decoder-only data smoke ran. |
| F3 — alpha objective | **Implemented, unverified** | Code and tests are present; PyTorch is unavailable in the local test environment and the configured GPU environment could not be reached. |
| F4 — tiny fit | **Not run** | Correct human-reviewed crops are a hard prerequisite; do not train until F1 and the F2 integration gate pass. |
| F5 — runtime profile | **Not run** | Requires eligible GPU access and the actual installed ONNX/PyTorch providers. |
| F6 — final-cut packet / AV1 comparison | **Not run** | Requires the canonical decoder/scorer integration and matching saved hold-out inputs. |

## Next bounded action

When GPU-host DNS is available, inspect all six hosts again and dispatch only
the F1 audit to an eligible device, with a two-minute allocation cap. Review
the generated sheets manually and label only the sampled candidates. Continue
only if that produces at least eight fit and four validation `visible_hand`
crops from distinct source seconds. Then finish the F2 integration into the
existing AV1/generator/metric paths, run the objective tests in the configured
PyTorch environment, and follow the plan's F4–F6 ceilings. Keep the aggregate
30-minute cap and the eight-minute per-job maximum unchanged.
