# Frozen ball localization on three match windows

The paper figure asks whether the transmitted ball location is usable directly by a machine while the same complete stream is rendered for viewers. Test the same 209/269/235 packets as the three-match ball-quality figure, with source RGB, ordinary AV1, fitted-ROI AV1 and actual tracked-tile reconstructions. Report every official labeled visible and labeled-not-visible frame, direct packet coordinates, and the detector output at a fixed 10 original-source-pixel tolerance. Not-visible does not establish physical absence. Same native cadence and full preceding temporal context are preserved; no tracker locations come from labels. Outcomes including losses and source failures are reported.

The frozen comparator is the earlier WASB `wasb-step1` checkpoint `9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb`, with three-frame input and output, stride 1, postprocessor threshold 0.5 and online tracker. The [author-reported split](../k5c-inputs/wasb-reported-split-provenance.json) identifies TrackNet training games 1–7. The original dataset event table lists those seven physical matches, all predating the RacketVision events tested here. The three broadcasts and the Basel confirmation are outside the documented training split. The training logs for the downloaded checkpoint and any undisclosed pretraining remain unverified. [Matched training-event table](reported-training-events.json) · [Original TrackNet paper](https://arxiv.org/abs/1907.03698)

CPU preflight is already completed by the preceding three racket-diagnostic jobs, which independently decoded every one of these source-aligned packets, verified packet and decoded hashes, and reproduced a previous DISTS ball score: [methods and per-source receipts](../ball-confirmation-starved-20261122/README.md). The GPU smoke additionally verifies those checksums, exact checkpoint and CUDA, predicts the first 24 real ticks of all four inputs and requires a 1.5× full-run forecast to fit. Pilot 269 first, then other matches only on success. Reserve at most 1,800 GPU-seconds and 3,600 CPU-seconds in total; stop for integrity, device, lineage contradiction, or forecast failure. Outputs and stage events remain in the fleet and numerical receipts here.

The first job is `20261121T225917Z-9ca4594a`, initially pending. It reserves at most 500 GPU-seconds for match269; [specification](269-spec.json). Inspect the existing ID after interruption.

269 completed. At the 12 official visible frames, WASB found 11/12 on source, 0/12 on plain AV1, 0/12 on ROI, and 12/12 on tiled video; the directly transmitted tile centers also hit 12/12. There are no labeled-not-visible rows in this match window. [Full predictions](269-result.json). Source lineage remains unknown. The smoke and full result passed packet, decoded hash, fixed CUDA and model checks; per-arm full inference took at most 1.2 seconds, so the predeclared 209 and 235 confirmations are justified.

The 209 and 235 confirmations are jobs `20261121T230240Z-e1fc4b49` and `20261121T230240Z-21bf3891`. Their caps are 500 and 800 GPU-seconds, respectively; together with the pilot's 500-second cap, the maximum reservation is 1,800 seconds. Inspect each ID on interruption.

Both confirmations completed. Hits within 10 native source pixels on the official visible rows are:

| Source | WASB on source | WASB on AV1 | WASB on ROI | WASB on tiled video | Direct tile center |
|---|---:|---:|---:|---:|---:|
| 209 | 1/9 | 0/9 | 0/9 | 3/9 | 8/9 |
| 269 | 11/12 | 0/12 | 0/12 | 12/12 | 12/12 |
| 235 | 7/11 | 0/11 | 0/11 | 8/11 | 10/11 |
| Pooled labels | 19/32 | 0/32 | 0/32 | 23/32 | 30/32 |

On five labeled-not-visible rows (three on 209, two on 235), WASB returned a position in 3 on source, 0 on plain, 0 on ROI, and 1 on tiled video; the direct packet contained a position in 1. These labels are not proof that the ball was physically absent. This is a descriptive result on three short, previously opened windows; 209 shows a poor source-image ceiling, and the three events fall outside the reported training games. The direct interface is the sender's fitted tile-center coordinates, including clipping effects, rather than the true center. [209 predictions](209-result.json) · [269 predictions](269-result.json) · [235 predictions](235-result.json).

All three jobs reached `complete`, validated their frozen settings and input reconstructions, and their terminal events were acknowledged. The [vector figure](../../paper/figures/ball-machine-three/ball-machine-three.pdf) verifies the numerical receipt hashes when generated.
