# Proposed native whole-frame base recovery encode — NOT AUTHORIZED OR RUN

Goal: produce verifiable actual whole-source SVT streams and decoded pixels for
Al031@129 and Med014@194, as inputs to a later saved-80 feasibility test. This
is a proposed CPU job, not a result or a delivered codec. No model inference,
quality scores, ball reconstruction or new generation. Independently search
only the exact paths identified by the inventory first; if matching streams
and pixel manifests turn up, verify them instead of encoding duplicates.

Source clips and native intervals: Al SHA256
054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76,
frames 9–248 at 25/1 fps; Med SHA256
a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be,
frames 74–313 at 50/1 fps. Convert using recorded ffmpeg 8.1.2 SHA256
7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2,
select inclusive frame indices, area scale 1920×1080, TV-range yuv420p,
passthrough timing, 240 frames. Compare full converted YUV SHA256 against
the two values in the inventory; halt on a mismatch.

Use pinned packed environment SHA256
32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661,
SVT-AV1 4.2.0 executable SHA256
4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4
and dav1d 1.5.4 SHA256
e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6.
Native per-plane downscaling in `experiments.tennis.k6.downscaled_source`
and its pinned dependencies, not an invented resize, at scales 8, 4, 2
yield 240×136, 480×272, 960×544 respectively. Encode each at CRF63,
preset4, 8-bit, range0, rc0, keyint -1, lp6, 240 frames and native fps.
Decode each via dav1d. Preserve six complete IVF streams, decoded yuv420p
outputs, source-conversion hash, per-frame decoded hashes, exact dimensions,
frame indices, FPS/timebase and each IVF timestamp, tool/environment hashes,
commands, payload bytes, 32-byte IVF and 12-byte/frame header bytes plus any
external framing. Save only to unique new job output, intermediates in scratch.
Do not overwrite data. File integrity checks on a fresh independent read and
decode are required. The old stream hashes are a reproducibility comparison,
not a requirement to call a newly encoded stream the old stream.

Validator: verify input hashes, exact converted YUV hash and expected sizes;
IVF signature, 240 complete frames/timestamps, no truncation, resolution,
correct total bytes and headers; independently decoded 240 frame sizes and
per-frame hashes; verify selected 80 ticks 0,3,...,237 against manifest, no
silent frame reorder, each output nonempty and finite, plus a simple source
luma sanity check that does not invoke a quality judge. Re-decode at least one
complete point independently; validate all six file hashes and metadata. A
mismatch blocks promotion. This is a whole-source base containing player and
ball; no target matte/Telea is transmitted to receiver.

Proposed budget: <=1800 aggregate CPU-s and <=45 min wall, 0 GPU, separately
authorized; one-thread/small-thread native smoke <=300 wall seconds with measured
child CPU, first 24 frames on both sources, scale2 and exact commands, then
project the six full 240-frame encodes, all decoding/validation and overhead
under both caps before promotion. A short stream's encode hash need not match
a full stream. Original receipts show scale8/4/2 encode wall 0.77/1.65/5.90 s
Al and 0.77/1.59/4.15 s Med under other parallel load; these do not measure
CPU-seconds, conversion, output, decoding or current headroom, so are not a
validated quote. Reserve and revise only from measured smoke. Archive each
completed setting. Stop on invalid smoke or projection/cap failure.

This does not supply ball metadata or full-frame quality and makes no claim
about the old missing streams. A later separate quality/DCVC quote needs real
inputs, native decoder availability, scorer smoke, controls and measured cost;
there is no measured GPU-hour quote yet. Ball needs exact-window encoder-side
position/visibility/shape/color/appearance and packet/timing provenance,
including unknown frames and cost. Any new detection/capture needs its own
scope and authority. Prior player pass used granted Telea and is development.
