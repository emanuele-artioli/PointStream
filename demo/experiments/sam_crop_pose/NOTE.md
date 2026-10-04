# SAM-crop hand pose comparison

Judgment after looking at `gallery.jpg`, 1 October 2026. This is the note to carry into the paper, not the mask-containment ranking by itself.

DWPose-l and RTMPose-l are terrible on the sheet. The other three trade blows. RTMPose-m gives up on many frames and draws nothing, and it has more than one hallucination where the keypoints are all over the place. Those are easy to reject from edge proportions: the landmark span is not a plausible hand. ViTPose-L is quite good, and still makes obvious mistakes, especially on clip 1 at 451 s. RTMW-l looks best on the sheet, including a realistic pose for a thumb that is completely hidden.

Use RTMW-l for training. RTMPose-m is faster at runtime, but the GPU spread is narrow enough that RTMW-l can be the runtime model too.

## What was measured

The box is a SAM hand/arm component from the sampled training seconds (`/home/itec/emanuele/Datasets/pointstream-demo/`), area at least 1500 px, expanded by 20%. Clip 3 at 210 s and 240 s (the aisle) is left out of the ranking. Each pose head receives that box. Whole-body models get it as their only person box. The hand detector is not in this test.

The automatic score is the fraction of the 21 joints that land inside the mask. There is no labeled joint ground truth, so that score is not a claim that the skeleton is correct. The gallery is what the model choice is based on.

| Model | Crops with at least half the joints inside the mask | Mean fraction inside |
|---|---:|---:|
| ViTPose-L | 4174 / 4681 (89.2%) | 0.854 |
| RTMW-l | 3880 / 4681 (82.9%) | 0.803 |
| DWPose-l | 3164 / 4681 (67.6%) | 0.649 |
| RTMPose-l | 3164 / 4681 (67.6%) | 0.649 |
| RTMPose-m | 2457 / 4681 (52.5%) | 0.485 |

DWPose-l and RTMPose-l match because they are the same ucoco-dw whole-body weights.

## Speed

CPU milliseconds per crop, 40 crops, each model alone, after warmup. ONNX Runtime fell back to CPU on gpu1 because that build needs cuDNN 9.

| Model | CPU ms | GPU ms |
|---|---:|---:|
| RTMPose-m | 32 | 4.4 |
| DWPose-l | 62 | 6.2 |
| RTMPose-l | 63 | 6.4 |
| RTMW-l | 95 | 7.3 |
| ViTPose-L | 224 | 7.6 |

GPU smoke: gpu6, RTX 6000 Ada, CUDA execution provider confirmed, eight crops, three warmup calls. ViTPose-L is about 1.7× RTMPose-m on that GPU, versus about 7× on CPU. Times include preparing the crop on the host.

## How to rerun

`demo/experiments/compare_pose_on_sam_crops.py` scores every crop. `pose_crop_gallery.py` redraws the sheet and times CPU. `pose_gpu_timing.py` times the GPU and stops if CUDA did not load. The full per-crop report stays at `/home/itec/emanuele/pointstream-data/jobs/sam-crop-pose/report.json`.

## Hand generator smoke

RTMW-l crops from these SAM boxes were used to train a SPADE generator and a pix2pix UNet (`demo/experiments/train_rtmw_hands.py`). FoundHand (CVPR 2025) was then scored zero-shot from the Drive checkpoint, using the demo sampler, against those two checkpoints (`demo/experiments/compare_foundhand.py`). The score is masked appearance L1 on one held-out crop from the last sampled second of each folder. Lower is better. Numbers are in `foundhand_compare.json`.

| Scene | FoundHand | SPADE | pix2pix |
|---|---:|---:|---:|
| clip 1, frame 35106 | 1.68 | 0.84 | 1.50 |
| clip 3, frame 11702 | 2.39 | 1.68 | 1.45 |
| factory 2, frame 35100 | 2.60 | 1.20 | 2.54 |
| Mean | 2.22 | 1.24 | 1.83 |

SPADE wins the smoke. FoundHand is worst on every frame. FoundHand-10M stays undownloaded: the checkpoint is already that pretraining, and the public hands are a different domain from these factory gloves.
