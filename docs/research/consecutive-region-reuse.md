# Consecutive fixed-region reuse control

Registered 1 October 2026, before inspecting the new quality outcomes. This is
a bounded native AV1 control on real consecutive source frames, not a complete
PointStream stream or a semantically certified foreground-free background.

## Source search and scope

The external data root contains the original 4K MP4s and per-video scene metadata.
BP30's retained `plates` are sparse scene representatives and cannot establish
observed duration. BP21's 48-frame windows likewise do not qualify a longer
session. The inspected `alcaraz_perricard/segmentations/scene_001` cache supplied
no usable mask files; other scene directories contain track images, without a
newly certified all-frame mask contract. We therefore register a fixed image
region and make no semantic mask, foreground absence, or plate quality claim.

The two registered source intervals are `alcaraz_perricard` starting at nominal
second 1 and `federer_djokovic` starting at nominal second 20, each for 30 seconds
of observed consecutive frames. The latter crosses retained metadata boundaries
near 28.445 and 43.460 seconds. Rounded labels must uniquely join exact metadata
`t_end` values within 0.5 ms before encoding. These are prescribed metadata reset
events; the labels are not independently verified scene-cut truth.

Crop the source's upper quarter and bicubically scale it to 640x90 YUV420p.
Round the nominal start frame, convert it to an input seek timestamp, preserve
decoded frame cadence without a frame-rate filter, and reject any selected PTS
gap or duplication outside the nominal cadence tolerance of 50 microseconds.
Report the actual selected first/last PTS and every frame hash. The requested
nominal start frame is not an independently verified absolute source-frame ID.
Hash the complete source MP4, scene metadata, raw extracted pixels, every frame,
and extraction log. Persist this input audit before the first arm encoding.

## Locked measurement contract

Native libaom AV1 uses CRF 32 and 44, cpu-used 6, row-mt, four threads, IVF,
and keyframe interval 9999. Independently encoded one-frame references support
hold-only, periodic 1/5/10-second refresh, and five-second refresh plus prescribed
metadata resets. Matched geometry, source frames, quantizer settings and luma
metric are retained for continuous AV1, independent ten-second resets, and
metadata-reset anchors. Their access contracts are reported separately.

Charge the physical IVF initialization and every physical refresh/reset stream,
plus the exact persisted JSON placement/deployment manifest. File-container and
manifest overhead are included. A fresh receiver subprocess reads only that
manifest and its charged native streams, verifies hashes and complete placement,
then writes decoded luma. Its output must match the sender's independently
decoded prediction exactly. Source paths are not receiver arguments; the source
files remain on the host, so this is not an operating-system sandbox proof.

All frames score against the original fixed-region luma. Record mean per-frame
PSNR and pooled MSE/PSNR separately. There is no changed-target removal score.
Observed 1/5/10/30-second hold-policy prefixes get their own physical placement
manifests and startup/refresh byte ledgers. These shorter horizons describe hold
amortization only; continuous/reset anchor comparisons use the complete observed
interval. Two CRF points do not justify a BD-rate claim or extrapolated duration.

This experiment excludes foreground synthesis, semantic mask transmission,
correction, full-frame deployment costs, learned models and model adaptation.
It cannot establish a complete codec advantage, general scene persistence,
training independence, or a guaranteed benefit from future optimization.

## Execution and qualification

Use only selected frozen worker, monitor, claims and contracts files, with an
archive checksum and exact Git revision. Inputs and outputs remain external.
After a successful six-host fleet inspection, GPU1's CPU capacity had sufficient
headroom. Allocate an eight-thread CPU claim, use the last eight available cores
to avoid the receiver job's first two cores, native threads four, nice 19, idle
I/O priority, CUDA hidden and a 16 GiB address-space limit. Hash reads have a
20 MiB/s cumulative per-file budget. Poll load during native commands and stop
only this worker's child if host load exceeds 40 or a command reaches 600 seconds.
These low-priority times are not a speed benchmark.

The detached existing monitor supplies a one-hour runtime budget and durable
logs. Run the same entry point with `--smoke` on both eleven-second source
intervals and CRF 32. This exercises periodic refresh, ten-second resets and the
first metadata reset in the second source. A full study starts only after the
fresh-manifest smoke passes. Preserve failed pilots; do not silently replay or
migrate an interrupted job.

The first pilot at `f53c1e2` predates the strengthened persisted-manifest receiver;
it is retained as an instrument pilot and is not paper evidence. A repeat at
the strengthened frozen revision is required before the full campaign.

No manuscript file is edited by this experiment branch. Result identities,
compact measurements and scope-limited prose will be supplied to the coordinator.
