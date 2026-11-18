# Lossless reference-image compression pilot

## Decision registered before encoding

For the two exact reference PNGs inside the selected qpng.pscp carrier, compare complete original PNG payload and REF1 transport bytes against WebP lossless using the installed cwebp 1.6.0 (`-lossless -m 6 -metadata none`) and AVIF lossless using installed avifenc 1.4.2 (`--lossless --speed 8 --jobs 1`). Decode the candidates with the matching installed decoders and require identical dimensions and every RGB byte compared with the original PNG. If a codec cannot produce and decode identical pixels it is rejected. Report per-player payload sizes, zlib-compressed metadata plus image in a REF1-shaped container, and their sums. Preserve actual output hashes, command options, versions and the exact selected carrier SHA-256. Encode each candidate once; no rate sweep. No JPEG XL encoder is present in the inspected local toolchain. Do not alter or dispatch the current receiver packet.

Hypothesis: a modern lossless image codec lowers transmitted reference bytes while keeping every conditioning pixel identical. Competing explanation: PNG plus the existing REF1 transport is already competitive for one or both images, or the codec's nominal lossless mode changes RGB. Promote a candidate for receiver implementation only if both images decode exactly and combined hypothetical REF1 bytes decrease. The payload-only saving is an upper estimate of packet savings until the alternate codec, resource identity, metadata and packet framing are implemented and priced. A CPU-only local pass has a budget of 120 seconds per image and codec, 10 minutes total, zero GPU. Do not infer generated-video quality gains from this byte pilot.

## Measured results

The frozen packet SHA is `3dd4aa753dfa3d15a6fdbd1b368d6b48929b4eb8c933e2217a33bed87f735bfd`. Both 512 × 784 RGB references round-tripped exactly with each candidate. These are one development rally's two existing appearance references. Each hypothetical transport uses its original reference metadata with the codec field updated, the same zlib level 6, and the existing REF1 header. Recompression of each original PNG reference reproduced its exact paid section length. The receipt includes SHA-256 of the encoded payloads and decoded RGB arrays, tool binaries and versions. The actual PNG REF1 sections total 365,939 bytes. This differs from the earlier 366,161-byte pilot reference total because it is the actual Linux-selected carrier with its independently rebound transport.

| Reference | PNG image | Paid PNG REF1 | WebP image | Hypothetical WebP REF1 | AVIF image | Hypothetical AVIF REF1 |
|---|---:|---:|---:|---:|---:|---:|
| Player 1 | 277,428 | 277,732 | 180,142 | 180,503 | 425,111 | 425,495 |
| Player 2 | 90,091 | 88,207 | 61,074 | 61,386 | 260,238 | 260,575 |
| Both | 367,519 | 365,939 | 241,216 | 241,889 | 685,349 | 686,070 |

Lossless WebP reduces the sum of modeled REF1 bytes by 124,050, **33.90%**, corresponding to 206.75 kbps if charged once over this 4.8-second window. The new full carrier, manifest, Linux decoder and resource metadata have not been implemented or priced. WebP is worth qualifying on Linux because it reproduces the exact conditioning pixels and reduces reference transport cost for both players. The selected AVIF setting is larger than PNG for both images. JPEG XL was unavailable locally. The checked application code did not expose a current WebP player-reference profile; the project history records earlier WebP appearance work in PR 75, which should be examined for reusable pieces before adding one. Cross-platform WebP decode parity and a bounded malformed-input check are needed before using a new profile in the receiver.

The two identical pilot passes used about 3.1 aggregate local CPU-seconds and no GPU; the second pass added explicit decoded-RGB hashes to the receipt and produced the same payloads. Reproduce from the project root with `PYTHONPATH=. python docs/experiments/lossless-reference-codecs-20261118/pilot.py`. The source snapshot used for the measurement is `91ea28df882ec44eb1dcb53aae8ec36e96b0df6c`, and the report and script are the attached working-tree additions. [Machine-readable receipt](result.json) · [Exact pilot](pilot.py)

## Lossy comparison registered before encoding

The author also asked about alternatives to JPEG. For each reference compare the exact existing Q40 JPEG payload and decoded RGB with lossy WebP and AVIF. Require each alternative payload to be no larger than that exact JPEG, and search quality using at most seven bisection steps over 0–100 plus quality zero to find the highest feasible encoder quality at that target. The search examines payload bytes alone; quality scores do not guide it. Use WebP `-m 6 -metadata none`, AVIF `--speed 8 --jobs 1`, and compute RGB PSNR against the selected PNG over all 512 × 784 pixels. Report the actual payloads and PSNR per player. This is a conditioning-image diagnostic, not a player-generation or preference result. If no candidate fits a target, report it. Budget 2 CPU minutes, 4 wall minutes, no GPU. JPEG XL is unavailable in the inspected local toolchain.

### Q40-matched result

The exact Q40 carrier and section hashes match the earlier published pilot. The decoded JPEG RGB arrays match both review-pilot hashes. The quality search used only encoded image length and then measured reconstruction PSNR. The PNG is the full prepared appearance crop, so this metric includes the entire crop as well as the player.

| Player | Q40 JPEG bytes / PSNR | WebP quality, bytes / PSNR | AVIF quality, bytes / PSNR |
|---|---|---|---|
| Player 1 | 20,913 / 38.02 dB | 80, 20,400 / 40.46 dB | 58, 20,300 / 41.32 dB |
| Player 2 | 20,206 / 38.37 dB | 85, 19,762 / 41.50 dB | 65, 20,007 / 42.80 dB |

These image PSNR results support testing compressed alternatives. The author accepted Q40 generated outputs for Player 1 and rejected them for Player 2. The new WebP and AVIF images have not been used for player generation or visually assessed. Image PSNR cannot establish whether the resulting players preserve identity, clothing and pose. Complete packet cost, cross-platform decoder parity and actual rendered output would need testing before substituting a codec in the receiver. The search used 1.99 local CPU-seconds and 1.80 wall seconds, with no GPU. Reproduce with `PYTHONPATH=. python docs/experiments/lossless-reference-codecs-20261118/lossy_pilot.py`. [Results and all size-search trials](lossy-result.json) · [Exact pilot](lossy_pilot.py)

## Campus decoder preflight registered before execution

Feed the two frozen PNG references and their exact lossless WebP candidates to the already staged Python/OpenCV runtime used by the earlier receiver checks on gpu5. In one CPU-only worker decode each pair with the installed OpenCV and require byte-identical RGB arrays and shapes. Verify the local WebP payload SHA-256 against this pilot's receipt before sending. Record the actual runtime paths, binary and build hashes, and CPU time. The worker uses an input pipe and no experiment files or GPU. CPU-only limit is 30 seconds, 2 minutes elapsed. This is a decoder-library parity preflight and does not establish a paid WebP packet, source-denied receiver or sandbox qualification; those require an explicit profile and subsequent isolated check.

### Campus preflight result

On gpu5, the exact two WebP bitstreams decoded to the same 784 × 512 RGB bytes as the paid PNGs. Their decoded hashes also match the earlier local PNG hashes. The worker used the same staged Python executable and OpenCV build fingerprint as the earlier receiver JPEG preflight, and `CUDA_VISIBLE_DEVICES` was empty. It consumed 0.32 campus CPU-seconds and took 59.62 seconds end to end, including local encoding and remote startup. This was an ordinary CPU worker rather than the receiver's isolated packet sandbox. The WebP recipe, full packet and isolated source-denied decoder remain to be implemented and verified. [Campus parity receipt](linux-preflight-result.json) · [Reproducible preflight](linux_preflight.py)

The current `common_player_packet` supports the registered PNG recipe and JPEG recipes. A source-denied WebP packet requires a new authenticated profile and receiver parser, bounded RIFF/VP8L dimensions and payload checks before image allocation, decoder identity and decoded-RGB bindings, compatibility tests, and a separately staged isolated source capsule. The WebP result above therefore remains a component-size projection and installed-library pixel preflight. The existing packet profiles and the active PNG replay were left intact.

The subsequent [complete carrier and isolated conditioning test](carrier/README.md) implements the lossless profile and measures its whole-packet cost on both replay bases.
