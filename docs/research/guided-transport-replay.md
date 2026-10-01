# Guided-development transport replay

The three retained September 27 run-05 packages reconstruct all 16 registered
3840×2160 RGB frames in fresh CPU processes, using the copied physical package
and preinstalled decoder software. All 48 source PNG file hashes and decoded
source RGB frame hashes match the original pilot audit. All three fresh output
pixel hashes exactly match their saved decoder outputs. Observed receiver
external opens contain only its copied package and emitted decoded array; no
SAM checkpoint, pose detector, source frame or content-specific learned weight
is supplied. Python file-open guards and recorded subprocess arguments establish
an observed boundary, not an OS sandbox.

| Registered source | Complete package bytes | Fresh pooled Y-PSNR (dB) |
|---|---:|---:|
| Alcaraz highlights scene 000, frame IDs 38–53 | 649,798 | 6.492212 |
| Alcaraz highlights scene 010, frame IDs 1–16 | 1,175,424 | 7.954044 |
| Alcaraz/Perricard scene 007, frame IDs 361–376 | 3,202,610 | 6.140564 |

Scores compare every delivered frame to its full retained RGB source, converted
to uint8 BT.601 luma. Mean frame dB and pooled pixel-MSE dB remain separate in
the record. This is a fresh measurement of foreground-only semantic transports:
the original configuration disables background, generation, pose, motion and
correction. The client renders an absent background as black. The poor
whole-frame scores show why foreground package size alone cannot establish
complete-video compression. They do not close the background or generative
codec families.

Physical package bytes exactly equal the saved runner ledger and its component
sum. The delivered information is coded appearance plus PSM1 masks, placements
and envelope data; background and correction charges are zero because those
roles are absent. Offline bidirectional SAM observations and guided prompt
retries are encoder-side development conditions, not zero-information causal
prediction. The three selected windows are not an eligible held-out split.
Run-03 and run-05 duplicate content, so only run-05 is counted. No matched
16-frame native streams were identified in the bounded package directories;
rate advantage, segmentation accuracy and task quality remain unqualified.
Native MP4 extraction and independent annotations were not recertified.

The original audit omitted a usable Git head in its dispatch summary. Its exact
fleet dispatch receipt recovers base revision
`768945c55e6a5a33d817be1e150824344d909db2`, selected tracked-patch SHA-256
`d7dd6782a75adddc3b70704038c979682acf85c6b9824824cb9d4e09a8253373`,
and snapshot SHA-256
`83eed424b2d7d512019793b9cc067278e28c96c9fc4c3f50ba98534c6028d7c8`.
Selected new-file hashes remain in the compact record. Receiver replay used
clean decoder revision `4748cbd91e693f9d9b19b4f207819beb26ab38fc` and a narrow
source/configuration snapshot. Historical encoder GPU identity is preserved;
the new smoke and full replay allocate no GPU and invoke no native video codec.

A representative first-scene full-16-frame smoke passed through the same
source-hidden decoder and scorer before all three windows were run. The
existing detached monitors completed and released their two-thread CPU claims.
Affinity uses the first two permitted logical cores, BLAS/OpenMP one thread,
nice 19, idle I/O priority and an 8 GiB address-space limit. No timing headline
is derived from these low-priority runs.

[Compact measurements and provenance](guided-transport-replay.json) retain all
original package and frame identities, exact costs, decoded hashes, per-frame
MSE denominators and original dispatch checksums. Full reports, media and
monitor receipts remain outside Git under
`gpu1:/home/itec/emanuele/pointstream-data/audits/receiver-20261001/`.

Receiver HOLE status: payload-only reconstruction and complete physical costs
are now qualified for these three selected foreground transports under the
observed guard. Whole-video background recovery, matched native anchors,
held-out exposure, independent perception/task labels and causal operation
remain open. This transport result does not fill those comparison HOLEs.
