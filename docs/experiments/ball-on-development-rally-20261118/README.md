# Automatic ball placement on the combined-development rally

Use the same authenticated Nadal–Kyrgios 240-frame source and pinned RacketVision BallTrack weights as the player and ball experiments respectively. Produce 240 automatic center predictions at native cadence. The feasibility question is whether a receiver can place real optical detections from this rally using a single paid native base. Missing detections, coordinate errors or target geometry could limit its value. No visual acceptance or correctness rate for the tracker follows from the number of placements.

Freeze the existing CPU-preprocessed tracker recipe with float32 CPU model execution, batch size two, two Torch threads, no training. Use OpenCV FFMPEG and area resize to the project’s 1280×720 RGB observation size, then scale predicted centers by the exact 1.5 canvas ratio. Preserve nulls. Authenticate checkpoint, code archive and the 1920×1080 50 Hz input. Test the first eight frames for finite output and a bit-exact repeat, then run the whole shot only if the measured forecast fits 15 wall minutes and 30 local CPU-minutes. The median uses the entire respective input sequence, so prefix and full predictions can differ. Stop at any contract failure. Apply the validated canvas ball transport to the Q70 carrier and record total bytes, placements and receiver pixel agreement. Independent source labels and an actual composed player reconstruction are needed for foreground quality.

## Observed result

On the complete 4.8-second rally BallTrack sent a location on **195 of 240 frames**. These are encoder predictions; independent ball labels are needed to count true detections and misses. The duplicate eight-frame smoke was deterministic with zero predicted locations. Full inference used 186.6 s wall and 368.2 CPU-s. Both panels used the pinned model with no fitting or GPU.

The real tile stream cost **17,620 bytes**, positions 960 bytes, and sidecar framing 534 bytes. The Q70 player input packet and this complete sidecar together cost **110,397 bytes (184.00 kbps)**. The receiver exercised the native fallback layer and placed the tracked source patches on it, matching an independent assembly on all 240 frames. The paid player geometry and references are included in this byte count; the decoder in this check did not synthesize the players. This is a transport and operating-rate measurement. Matched-quality joint reconstruction remains open.

[Track and model identities](track.json) · [Native receiver and byte accounting](../joint-readiness-20261118/sender-sidecar.json)

## Three-rate ball-only ablation, registered before decoding

Use the exact three plain SVT-AV1 IVF streams from the 960×544 development control. Reuse the authenticated 240-tick sidecar's native tile and model locations. Bind each tile stream to a fresh **native-only** carrier, so these measurements charge no player reference. Render the 1920×1080 canvas at all 240 timestamps. Require every absent frame to equal its decoded base, every pasted region to equal an independent tile decode, and all other pixels to equal their source base. Report complete bytes and pooled RGB PSNR over the 195 predicted 32×32 regions against the true source pixels, separately from any independently labelled ball metric. This tests whether a paid local appearance stream improves fidelity at the locations the sender actually chose. The native anchor uses fewer bytes at the same base CRF. Compare regional quality at matched bitrate only inside the measured overlap. Allow four local CPU wall minutes per full rate with zero GPU. Existing 240-frame real-track replay proves input/decoder feasibility.

| Base CRF | Native-only kbps | Native + ball kbps | Native crop PSNR (dB) | Native + ball crop PSNR (dB) |
|---|---:|---:|---:|---:|
| 36 | 638.34 | 670.20 | 26.90 | 30.98 |
| 48 | 304.32 | 336.18 | 24.35 | 30.98 |
| 60 | 126.88 | 158.74 | 22.00 | 30.98 |

All three complete receivers matched the independent reconstruction on all 240 frames. Adding the real tracked tiles costs 31.86 kbps on this 4.8-second shot. Even the lowest-rate ball output preserved the pixels at the predicted locations better than the highest-rate plain output by this diagnostic. The ball PSNR stays constant because the sender pays the same independently coded tile at every base rate. Equal-quality interpolation is unsupported between these measured PSNR ranges. Occluded balls, missed positions and false detections require source-only reference labels. The result measures source-pixel fidelity at the sender’s positions. [Full ablation](three-rate.json)

### Matched ablation byte budget

Rebind each archived Mac player packet to the same three verified native IVF base streams used by the ball-only ablation. Hold both references and all keypoints fixed at each quality. Rebuild the complete wire encoding and bind the identical paid tile/placement stream to each carrier. This yields actual rate targets for the native, player-only, ball-only and joint arms before rendering their player pixels. The selected Linux carrier, its runtime manifest and receiver decoding must be used for the eventual quality comparison.

| Base CRF | Q40 joint kbps | Q70 joint kbps |
|---|---:|---:|
| 60 | 233.30 | 273.38 |
| 48 | 410.74 | 450.82 |
| 36 | 744.76 | 784.84 |

The real serializations use each base once. Changes in the tile stream are unnecessary when the tracker, source and placement policy are identical. The original same-window neural anchor points (36.83 and 49.78 kbps) cannot bracket these cold-start rates. The new native and ROI curves cover the first two combined operating points; the [additional upper control](../native-control-20261118/full-upper.json) subsequently bracketed the third. [All four byte counts, per rate and setting](joint-bytes.json)
