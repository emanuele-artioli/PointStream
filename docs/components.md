# Components

PointStream splits a video into **foreground** (people, the objects they handle,
the ball) and **background**, and encodes each with the best tool for it. The
components below are high level beyond segmentation. Background and foreground
encoding get their own design sessions.

| # | Component | State |
|---|---|---|
| 1 | Data | To build |
| 2 | Segmentation | `src/segmentation/` |
| 3 | Background | To design |
| 4 | Foreground | To design |
| 5 | Reconstruction | To design |
| 6 | Bitstream and rate accounting | To design |
| 7 | Evaluation | Segmentation metrics in `src/segmentation/evaluate.py` |
| 8 | Demo | Second priority |
| 9 | Infrastructure | Fleet in `experiments/jobs/`; storage in `src/segmentation/storage.py` |

## 1. Data

Labelled datasets ([resources](resources.md#datasets)) behind one adapter
interface. Each adapter yields frames and masks with the dataset's native classes,
a per-frame "labelled" flag (sparse labels are scored only where they exist),
point labels for the ball, and a provenance tier per mask.

## 2. Segmentation

`src/segmentation/` turns a clip into lossless per-instance masks (`ClipMasks`,
COCO-RLE, zstd-compressed). A domain in `domains.yaml` names the foreground
classes; everything unlabelled is background.

- Backends: SAM 3.1 Object Multiplex (offline reference, run as a worker in its
  own environment from a pinned checkout, windowed for long clips) and YOLOE-26
  candidates with mask-IoU tracking.
- To add: label-prompted SAM 3.1 for classes a dataset lacks (players and
  rackets from keypoints and boxes, arms where labels stop at the hand), the
  ball segmented by SAM 3.1 from point prompts at labelled positions, and
  proposers for the handled object.
- CLI: `python -m src.segmentation {run,dataset,bench,suite,validate,export,preview,sheet}`.
  `suite` and `validate` form a fleet smoke gate; every run records its provenance.
- Evaluation: J, F, recall, precision and flicker against a reference root.

## 3. Background

After a warm-up, the client holds one representation of everything the camera
has seen (a panorama for a camera that rotates in place, or a neural model
fitted to the warm-up), and each later background frame is rebuilt from it
with only the camera pose. The foreground is removed first; a panorama fills
what one frame hides from other frames, a neural model may need generative
inpainting. Steps G1–G5 in [PLAN](../PLAN.md#g-background-user-2026-10-08).

## 4. Foreground

Per-object representation (appearance reference, pose or keypoints, mask),
decoded generatively or as coded crops. A parametric trajectory is a candidate
for the ball.

## 5. Reconstruction

The decoder composites the foreground, with transparency, over the background.

## 6. Bitstream and rate accounting

Every bit the receiver needs is counted, including model updates or adapters
sent per content.

## 7. Evaluation

SVT-AV1 and a neural-codec baseline; weighted PSNR (0.7 foreground + 0.3
background) on dataset masks, perceptual metrics, encode and decode timing, and
the ablations a journal paper needs. Results are reported per provenance tier.

## 8. Demo

Built from the same components, after the paper's needs.

## 9. Infrastructure

The fleet ([fleet](fleet.md)), provenance records, dataset manifests and
environments ([resources](resources.md)).
