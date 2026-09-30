# CPU-only evidence strengthening

Authorized 30 September 2026 after the empirical manuscript rewrite. This
study reuses retained native BP21 AV1/VVC streams. It performs no encoding,
training, model inference, or GPU allocation and does not modify source data.

## Questions fixed before the run

1. Do independently decoded original and player-removed streams retain their
   saved bytes, geometry, frame counts and own-target mean-frame Y-PSNR?
2. How do player-removed streams score against the original source, including
   whole-frame, stored foreground-mask and background-mask measurements?
3. Do the original and unrestored removal curves have common original-target
   support? Report support or PCHIP BD-rate only inside that support.
4. How sensitive is the saved foreground-removal summary to grouping repeated
   scenes by source video? Source video is not verified match independence.

These are diagnostic questions. Removing players without restoring them is
not a complete semantic codec. A physical stream replay cannot qualify a
missing PointStream package, independent task truth, subjective quality,
training exposure, or long-duration background persistence.

## Immutable input and scoring contract

Input: external `outputs/bp21-headroom/report.json`, SHA-256
`b6f8b8463c73f7d3a54a918b677bdb5b1e18ae7cf789490c900e023fa1d21240`.
All eight registered 48-frame 3840x2160 windows, the stored masks and native
original/plate AV1/VVC three-point streams must exist. Each native curve's
physical byte sizes must equal the saved sizes. Each frame must have nonempty
foreground and background masks; no missing frames or regions are dropped.

The first full input audit found an extra older VVC QP39 file in the Djokovic--
Federer directory. That initial job was stopped and retained. The corrected
worker binds streams to the pinned report's three exact byte counts before
observing fresh quality, rejects ambiguous identities, and records excluded
older files. It suppresses BD-rate for reversed rate--quality segments.

The fresh decoder gets the bitstream alone. Original and plate luma and masks
are available only to the scorer. The reference is the stored 8-bit BT.601
luma Y4M, not a newly interpreted RGB/color target. Mean per-frame Y-PSNR
matches the historical metric. Pooled pixel MSE/PSNR is separately named.
Pixel hashes, native decoder path/binary hash/version, worker hash, exact code
revision, commands, runtime, observed GPU UUIDs and resource policy travel in
the report. Source and plate must be identical outside the removal masks.

Three points per curve restrict interpolation sensitivity and inference.
Source-cluster bootstrap (10,000 draws, fixed seed 20260930) reports both the
scene-weighted estimator and equal-source estimator, plus leave-one-source-out
sensitivity. It does not include pixel/encoder/model uncertainty and does not
claim a representative population or verified independent match sampling.

## Resource and execution contract

Choose GPU3 only after the six-host fleet inspection: it had no GPU compute
process and 63 available CPU threads. Recheck immediately before each launch.
Use a separate code snapshot and unique external `audits/cpu-evidence-*`
directory. Limit the process tree to eight CPU logical cores, native decoder
to four threads, BLAS/OpenMP to one, process priority to nice 19, I/O priority
to idle, and address space to 16 GiB. Hide all CUDA devices. File reads are
capped at 20 MiB/s. No timing from this low-priority run is a speed benchmark.

The worker pauses with retained partial output if host load exceeds 16 or any
GPU compute process appears on its host; it never signals outside processes.
The existing detached monitor supplies an eight-thread CPU claim and a two-hour
budget. A first-scene, first-QP original/plate full-48-frame smoke uses the same
worker/decoder/scoring path. A full run starts only after inspecting that smoke.
No automatic migration or replay of interrupted work.

Entry point:

```sh
python experiments/headroom/cpu_replay.py --data-root EXTERNAL_ROOT \
  --out UNIQUE_NEW_DIRECTORY --code-revision CLEAN_COMMIT --smoke
```

Remove `--smoke` for the full campaign after the pilot passes. Reports and media
stay outside Git; publish only a compact numerical/identity summary and the
implementation. Active Cursor work and historical outputs remain untouched.
