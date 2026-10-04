# Retained receiver-package replay

On 1 October 2026, twelve retained E06 packages were reconstructed in fresh
processes from copied payloads, with CUDA hidden. Every output contains all
48 registered 640×360 RGB frames. Python file-open guards reject external
source/cache access; recorded native decoder commands read only temporary
files derived from payloads. This is an observed access boundary, not an OS
sandbox or proof against arbitrary native code. The receiver receives no
content-specific learned weights. JPEG/WebP, native VVC/AV1 and the identified
software libraries are preinstalled decoding resources.

The four original envelopes exactly match historical SHA-256 and physical byte
counts. Each compact envelope produces exactly the same RGB pixel hash as its
original. All four floor packages match the first-reference, residual-off parent
pixel hash. The complete physical packages include background, geometry,
appearance, masks, placements and envelope information. Compact bytes are
30,297 and 108,210 for first-reference residual off/on; 92,023 and 164,088 for
per-frame crops off/on. At fixed first-reference-off pixels, RLE masks reduce
the physical package to 17,581 bytes, keyed-XOR masks to 22,055, and the thin
placement arm to 28,125. These are lossless packaging comparisons on fixed
pixels, not a new rate–quality ladder.

The scorer alone reads the retained prepared RGB source. Its pixel hash
`1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8`
matches the historical recipe. Fresh pooled BT.601 uint8 Y-PSNR is 20.680368 dB
for first-reference/off and 20.800022 dB for per-frame/off; correction raises
these to 24.133447 and 24.151142 dB. Mean per-frame dB is separately recorded.
Fresh pooled scores differ from historical values by at most 0.000277 dB;
this is not claimed as exact historical score reproduction. No missing frames
are discarded, and no mask-derived metric is interpreted as annotation truth.

Qualification is restricted to one selected development window and this
cached prepared target. The original source recipe records native-PTS/PNG
agreement for only one of 48 frames; no MP4 extraction was repeated. No new
matched native-anchor decode, held-out exposure, subjective quality,
long-session reuse, causal operating assumption or broad codec advantage is
certified. Other historical candidates remain separately unqualified: a
bounded depth-three search of JE10 found its two color IVF components and
result JSON, while measured-tennis retained its result and comparison images;
no complete receiver package was found within those named directories.
This absence is scoped to that search, not all storage.

The representative first-package smoke passed before the twelve-package run.
The existing detached monitor completed both runs and released their CPU claims.
CPU use was restricted to two logical cores, BLAS/OpenMP to one, nice 19,
idle I/O priority and an 8 GiB address-space limit; no speed claim follows.
Frozen decoder revision is `0b56ceb`; the later code adds mandatory inventory
count and a tested cache-access rejection without replaying the measurement.
The narrow snapshot contains only selected source/configuration paths.

[Machine-readable measurements](receiver-evidence.json) retain original package
paths/hashes, complete bytes, decoded pixel hashes, all-frame denominators,
source identity, native binary path/hash/version, linked libraries, Python,
NumPy/OpenCV/Pillow versions, observed GPU UUIDs and report identity.
External full scores, decoded frames and supervisor receipts remain under
`gpu1:/home/itec/emanuele/pointstream-data/audits/receiver-20261001/`.

Receiver HOLE status: persisted, complete-cost, source-hidden E06 reconstruction
is now qualified for this prepared window under the observed guard. The common
source/anchor, exposure, original extraction and comprehensive quality HOLEs
remain open. The JE10 and two-window measured-tennis receiver HOLEs remain open.
