# Shared player-reference image codec calibration

The codec uses one globally fixed image-format and quality configuration for every player in an evaluation. Image sizes can differ by image content. Calibrate the setting on development footage, freeze it before opening held-out matches, and report each player's results and transmitted bytes. Do not pick a quality based on an individual player's outcome.

## Registered CPU pilot

Use the exact two 512 × 784 PNG references from the selected authenticated development-rally packet. Compare three fixed configurations per format, applied unchanged to both images. JPEG uses OpenCV RGB8/BGR8 conversion with quality 40, 70 and 90. WebP uses cwebp 1.6.0 quality 65, 80 and 90, method 6 and no metadata. AVIF uses avifenc 1.4.2 quality 45, 65 and 75, speed 8 and one job. Quality numbers are codec-specific and are not treated as matched perceptual quality. Report image bytes, hash, decoded RGB hash and full-crop RGB PSNR per image. Count the actual JPEG-Q40 carrier bytes and compare modeled REF1 framing for alternatives, without implying these models are deployed packet bytes. Bound the CPU experiment to two CPU minutes and four elapsed minutes, no GPU. The exact tool identities, source and decoded images must be recorded. These two subjects are a development check, not a general claim about a codec.

Hypothesis: one lossy codec setting may improve image rate distortion over the installed JPEG at comparable payloads. Competing explanation: apparent image PSNR improvement does not carry through generation or to other players. Image PSNR alone cannot promote a setting.

## Registered downstream render

Use common AVIF quality 65 for both players as the next 24-frame development render. This is a single setting chosen before running the common-codec pilot, based on the earlier candidate intended for Player 2 and a higher setting than previously prepared for Player 1. Test it against the already reviewed JPEG Q40 and lossless references with fixed poses, boxes, checkpoint, frame times and seed. Show source-aligned clips under blinded, randomized labels. Rate body, clothing, identity and pose separately for both players. One development setting can establish only feasibility. Freeze candidate format and setting using aggregate development evidence, then evaluate the same setting on previously unopened players and matches, including their actual reference bytes and initialization/reuse costs. Include representative appearance complexity, motion, scale and occlusions. Include the lossless reference as a quality control, plus a rate-matched other codec where available, and retain any failures in the result. The previously prepared AVIF inputs used player-specific qualities 58 and 65 and were never rendered. The shared Q65 inputs replace them for this question.

The previously approved budget of 900 GPU-seconds and 1,350 CPU-seconds covered one AVIF setting. Input preparation and parity should be checked for this fixed shared setting before dispatch, and admission must use the normal fleet smoke and latest actual costs. A conditioning-only render does not establish complete AVIF packet support or cross-platform receiver parity. Select a global operating setting from downstream results at comparable total reference rate across subjects. Never select a winner solely from this RGB PSNR diagnostic.

## CPU result and prepared input

All candidates decoded to the expected shape. The Q40 JPEG payloads are the actual prior review payloads. Other JPEGs use the OpenCV encoder and decoded RGB in the current environment. Both Q40 modeled REF1 sections reproduced the paid sizes exactly. [Source hashes, encoder versions, all 18 configurations and timing](receipt.json)

| Common format and quality | Player 1 bytes / RGB PSNR | Player 2 bytes / RGB PSNR | Both payloads |
|---|---:|---:|---:|
| JPEG Q40 | 20,913 / 38.02 dB | 20,206 / 38.37 dB | 41,119 |
| WebP Q65 | 14,390 / 38.70 dB | 11,512 / 38.55 dB | 25,902 |
| WebP Q80 | 20,400 / 40.46 dB | 15,632 / 40.27 dB | 36,032 |
| AVIF Q45 | 12,877 / 39.15 dB | 11,586 / 39.81 dB | 24,463 |
| AVIF Q65 | 24,172 / 42.01 dB | 20,007 / 42.80 dB | 44,179 |

At these sampled settings, AVIF Q45 has higher full-crop RGB PSNR and fewer image bytes than JPEG Q40 for both references. This is a preliminary two-image result, and crop-level PSNR does not determine how well the renderer preserves a player. One common AVIF Q65 conditioning archive has been created. Every changed reference image hash matches the registered image test, and each non-reference conditioning array matches the authenticated JPEG-Q40 input. Its two image payloads total 44,179 bytes before transport and manifest overhead. [Prepared conditioning manifest](prepared.json) · [Fleet template](spec-template.json)

Local inputs are in `/private/tmp/ps-common-avif-q65-20261122`. The archive and config have been staged on campus in a fresh directory and independently SHA-256 verified. No render job has been submitted. Before dispatch, check the latest job records and eligible hosts, validate the frozen source snapshot and current smoke forecast, then replace the template deadline. The prior approved AVIF pilot covers one setting. The 900 GPU-second cap includes the 24-frame Player 2 smoke and both full renders. Its 1,350 CPU-second budget includes the registered external CPU reservation. Preserve the existing lossless, JPEG and previously staged AVIF data.

After the fleet results are available, run `docs/experiments/avif-appearance-20261118/make_clips.py` with the verified prepared config, conditioning and base directories, smoke and full outputs, prepared config SHA-256, and `--output /private/tmp/ps-common-avif-q65-review-clips-20261122`. It validates the actual outputs and frame counts. Then run `PYTHONPATH=. python docs/experiments/common-reference-20261122/review.py`. The builder checks the clip receipt against the common Q65 config, authenticates the previous source and control assets, and creates the blinded review and a separate reveal file. The builder has no new clips until the render completes.

A local synthetic fixture exercised the builder with copies of existing clips and authenticated control assets. The builder produced both player grids, and rejected a mismatched config hash before creating its output directory. This checks preparation only and supplies no AVIF generation evidence.

[Independent campus staging receipt](staging-receipt.json)
