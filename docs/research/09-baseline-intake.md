# 9. Baseline intake and source-level replication audit

Inspected 28 September 2026. This extends E03 with concrete entry points and
newly identified integration gaps. No external model was installed or executed.
Revisions below were read with `git ls-remote`; checkpoints remain unverified.

| Candidate | Observed main revision | First experiment |
|---|---|---|
| GVC-RT | `d0e32bfa3e8e282f9a77437c223c605858eea637` | Supported-domain RGB sequence, 16 frames, two rates, actual stream and saved reconstruction |
| MTTF | `b5b0db16ed64841f03082b163e5abdf98cb4136d` | Square 384×384 human clip, 16 frames, native learned motion plus transmitted reference |
| GLC-video | `126db25f9c093508cd0c99cee32b53fd60074f9a` | Inference/estimated-rate reference first; do not classify as complete-wire codec yet |

GVC-RT and MTTF source observations used the browsed main files; commit-specific
raw retrieval failed. Reopen their pinned files at installation and record file
hashes before applying patches. GLC's pinned test file was retrieved successfully.
This limitation does not imply that either repository is unavailable.

## GVC-RT: first complete-stream candidate

The [official setup](https://github.com/semcomm/GVC-RT) specifies Python 3.12,
Torch 2.6/CUDA 12.4, I/P checkpoints, and a required C++ entropy extension.
Use a separate environment, not an upgrade of PointStream's pinned environment.

Invoke `test_video_gvcrt_1088.py` with explicit checkpoint and input-config paths,
two QP pairs (0/0 and 3/3), one worker, 16 frames, stream writing and reconstructed
frame saving enabled, cache reuse disabled, and external output paths. Preserve
the native intra/reset policy. The input is RGB PNG; do not begin with YUV.

The [test source](https://github.com/semcomm/GVC-RT/blob/main/test_video_gvcrt_1088.py)
needs these qualification checks:

- Missing P weights can fall through to random initialization: make this fatal.
- Processing uses at least 1088×1920 pixels. Recompute bpp from actual bytes and
  original display dimensions; use actual PTS duration rather than its 30 fps
  bitrate assumption.
- It feeds [0,1] values into LPIPS without normalization. The
  [LPIPS API](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py)
  expects [-1,1] by default. Independently rescore saved frames for every codec.
- Separate decompression from source-reading/scoring for the fresh receiver test.
- Preserve fleet's GPU UUID visibility: the script rewrites CUDA visibility.
  Qualify a minimal one-worker patch that retains the inherited mapping.

These are source-level findings, not a reassessment of the paper's reported gains.

## GLC-video: estimated-rate inference is a separate evidence tier

The [official video shell script](https://github.com/jzyustc/GLC/blob/main/test_video.sh)
uses `test_video.py`, separate image/video weights, a JSON sequence configuration,
one worker and saved reconstructions. Its advertised environment is Python 3.12
and Torch 2.5.1/CUDA 12.4 ([setup](https://github.com/jzyustc/GLC)).

The [pinned test entry point](https://github.com/jzyustc/GLC/blob/126db25f9c093508cd0c99cee32b53fd60074f9a/test_video.py)
constructs a `.bin` pathname but calls model inference and writes images/JSON,
not that stream. It also loads P weights non-strictly and rewrites GPU visibility.
Require a missing/unexpected-key audit and preserve the fleet mapping.
The [video model](https://github.com/jzyustc/GLC/blob/main/src/models/video_model.py)
combines Gaussian bit estimates with an index-storage calculation. These are not
measured complete-stream bytes.

Retain GLC as a literature-connected perceptual/estimated-rate comparator, clearly
separated from physical-rate curves. Promote only after a compatible released or
validated entropy encoder/decoder is found. Do not invent a stream flag or write
a new codec merely to satisfy the deadline. If unavailable, document that outcome
and prioritize GVC-RT for the physical-rate comparison.

## MTTF: charge the reference and isolate the decoder

The [encoder](https://github.com/xyzysz/Extreme-Human-Video-Compression-with-MTTF/blob/main/encode.py)
accepts config/checkpoint, source image, driving frames, output parameter directory,
frame count, image shape and quantization factor. Start with the native 384×384
TED configuration and 16 frames. Preserve its learned motion interface; DWPose is
not a drop-in replacement. Encode from the same decoded reference delivered to
the receiver, and count its compressed bytes alongside every parameter file.

The [decoder](https://github.com/xyzysz/Extreme-Human-Video-Compression-with-MTTF/blob/main/decode.py)
uses source image plus coded parameters for synthesis, but its demonstration
wrapper also opens driving frames to construct a comparison video. Extract a
receiver-only path around the existing synthesis function; supply FPS/dimensions
as charged metadata. Save reconstruction without the source/comparison panels.
The decoder source image is not cropped/resized like the encoder input, so prepare
one canonical square reference and verify both sides use identical pixels.

Checkpoint access, matting weights, legacy dependencies and config compatibility
remain intake gates. Native human-clip success precedes tennis transfer; center
cropping a tennis frame must never silently remove one player from the comparison.

## Common acceptance packet

Before launching, resolve each path and freeze a run manifest using E03's fields:
code/patch and checkpoint digests, independent environment, input frames/PTS,
reference policy, command, output location, resource estimate, budget and controls.
Inspect all fleet hosts, then launch a representative bounded smoke through the
actual patched entry point. Verify that its process uses the claimed GPU UUID.

Retain the official/native metric report for reproduction diagnostics and a
separate common-metric report for cross-codec comparisons. Include all frames,
initial references and side information. A fresh receiver must reproduce the
saved output without source inputs. A failed checkpoint/import/stream test is an
integration failure, not poor model quality. End the card with replicated,
inference-only, unavailable, or failed, plus the next bounded action.

This audit makes the prerequisites concrete; it does not certify that they pass.
See [the experiment plan](07-experiment-plan.md) for ordering and
[the evaluation contract](06-evaluation.md) for claim eligibility.
