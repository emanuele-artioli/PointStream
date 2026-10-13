# Coding for machines: how MPEG judges a codec, and PointStream's machine judge

Literature study for PLAN K2 (2026-10-13). No runs. Question (user,
2026-10-13): is H3c2's judge (pose re-estimated on the decoded video, against
rate) how machine codecs are evaluated, and what should PointStream adopt so
that its keypoint modes (hand pose, body pose, ball trajectory) are judged on
a recognized axis beside the human-viewing metrics (LPIPS, DISTS, weighted
PSNR)? Added the same day (user, through the foreground-audit session): which
datasets VCM and FCM use, and whether PointStream should use any of them.

Every claim below was checked against the cited primary source (MPEG output
documents downloaded from mpeg.org's public meeting pages, papers from
arXiv). What could not be verified is listed at the end.

## Summary

- **Yes, in kind.** MPEG's Video Coding for Machines (VCM) and Feature Coding
  for Machines (FCM) judge a codec by running a fixed, pretrained task network
  on the decoded content and plotting the task metric (mAP, MOTA, mIoU)
  against bitrate; the comparison is a BD-rate on the task metric against an
  anchor. H3c2 is this, with pose error as the metric.
- **Four differences matter for PointStream.**
  1. *Labels.* VCM and FCM score against the datasets' human labels, not
     against the network's output on the uncompressed video. H3c2 scored
     against WiLoR on the source (a pseudo-label). Pseudo-labels have
     precedent outside MPEG (Hu et al. 2020's landmark NME, FOMM's AKD), but
     they hide the estimator's own error, and here they favour the keypoint
     stream, which *is* that estimator's output.
  2. *Independent judge.* The task network is fixed by the test conditions
     and is not part of the codec; VCM's requirements make the codec agnostic
     to the network. H3c2 judged WiLoR's coded joints with WiLoR.
  3. *Operating range.* FCM computes BD-rate only where the task metric is
     within about 10% of what the network achieves on the uncompressed data
     (its performance points). A win at an accuracy far below the source's
     does not count.
  4. *Anchor.* VVC: VTM inside VCM's reference software (VCM), HM-18.0 since
     the August 2026 FCM conditions (FCM, previously VTM-23.3). PointStream's
     anchors are SVT-AV1 and DCVC-UF.
- **Pose is not in MPEG's current test conditions.** The January 2021 VCM
  conditions listed pose estimation and action recognition (HiEve); by July
  2022 only detection, instance segmentation and tracking remained, and so it
  stays in the September 2026 conditions. No recognized pose protocol exists
  to copy; PointStream defines one from the VCM/FCM method and the pose
  benchmarks' own metrics (PCK, OKS-AP).
- **Proposal** (below): per keypoint mode, a task metric on decoded content
  against human labels (HInt for hands, TrackNet and RacketVision for the
  ball and rackets; pseudo-labels from an independent estimator only where no
  labels exist, i.e. tennis bodies), BD-rate on that metric over FCM-style
  performance points, against the pixel envelope and a VTM anchor, reported
  beside the human-viewing curves as a scalable (machine base layer + human
  enhancement) bitstream.
- **Datasets:** none of VCM's or FCM's fits PointStream's domains (no
  egocentric video, no racket sports, no hand or body keypoints in the
  current conditions). Use them only if a reviewer asks for comparability with
  MPEG numbers; the closest is SFU-HW's three basketball sequences.

## MPEG-AI status

| Part | Status | Source |
|---|---|---|
| ISO/IEC 23888-2, Video coding for machines | Draft International Standard at MPEG 153 (January 2026); FDIS planned late 2026 | [MPEG 153 page and press release](https://www.mpeg.org/meetings/mpeg-153/) |
| ISO/IEC 23888-4, Feature coding for machines | Committee Draft at MPEG 154 (April 2026), after the FCVCM call for proposals evaluated in October 2023 | [MPEG 154](https://www.mpeg.org/meetings/mpeg-154/); [ISO CD 23888-4](https://www.iso.org/standard/92539.html) |
| JPEG AI (ISO/IEC 6048, ITU-T T.840) | Version 1 published 2025 (T.840.1 / ISO/IEC 6048-1:2025), for human viewing; machine tasks in the latent domain are named as the next version's scope | [jpeg.org/jpegai](https://jpeg.org/jpegai/); [Esenlik et al. 2025, arXiv:2510.13867](https://arxiv.org/abs/2510.13867), sec. II and conclusion |

## How VCM judges a codec

Source: **Common test conditions for video coding for machines**, ISO/IEC JTC
1/SC 29/WG 04 N0870 (w26718), 2026-09-07, MPEG 155
([zip](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/155_Geneva/w26718.zip));
its predecessor N0789 (w26007, MPEG 153) has the same evaluation rules.

- **Pipeline.** Encode the source (YUV 4:2:0) with the codec, decode, convert
  to PNG with FFmpeg, run the task network, score "using the task network
  output and the ground truth". Y/U/V/RGB-PSNR against the original are
  reported beside (SSIM and MS-SSIM in the template too).
- **Task networks (fixed).** Detection: Faster R-CNN X101-FPN (Detectron2,
  `model_final_68b088.pkl`; FLIR uses a fine-tuned copy). Tracking:
  JDE-1088×608. Semantic segmentation: Panoptic FPN R-101-FPN 3x.
- **Metrics.** mAP (detection, instance segmentation), MOTA (tracking), mIoU
  (semantic segmentation). Rate in kbps for video, bpp for images.
- **Anchor.** VCM-RS, the VCM reference software, with VTM 20.0 as its inner
  codec (the text names VCM-RS v1.2 and, in the anchor command, v1.3; the
  attached templates are named `VCMRS_2.2`). The anchor is not plain VVC: it
  runs VCM-RS's own pre/post tools, including RoI retargeting (up to 3 or 11
  RoIs, background filter and down-scaling on PandaSet), temporal and spatial
  resampling, bit-depth truncation and luma enhancement. Six configurations:
  random access, low delay and all intra, each "inner" and "end-to-end".
  Random-access intra period 32 at ≤30 fps, 64 at 50–60 fps.
- **Rate points.** Six QPs spaced by 4 from a per-sequence starting QP
  (SFU-HW, TVD), spaced by 5 (22–47) for PandaSet and images. A codec without
  QPs gives six points each within ±10% of the anchor's rate; a codec whose
  curve does not overlap the anchor's must add points until it does.
- **BD-rate.** A monotonic cubic curve fit (Wang et al., PCS 2024) to the
  task metric, then BD-rate over the middle four of the six points (lowest
  and highest QP dropped). Per-dataset averages.
- **Cross-check.** An independent expert must reproduce the bitrates exactly
  and the task metric within 1% (relative).
- **Requirements** (WG 2 N0190, "Use cases and requirements for VCM",
  2022-04-29,
  [zip](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/138_OnLine/w21545.zip)):
  at least 30% BD-rate over VVC on machine tasks; agnostic to network models
  and to task types; one bitstream serving any number of task instances;
  hybrid machine-and-human consumption is in scope. Pose estimation is
  defined as a technique and appears in the use-case tables (unmanned store;
  AR/VR goggles).
- **History.** The first CTC (WG 2 N54, 2021-01-15,
  [zip](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/133_OnLine/w20161.zip))
  listed five tasks: detection (OpenImages V6, FLIR), instance segmentation
  (OpenImages), and tracking, **pose estimation** and action recognition on
  HiEve-10, judged by BD-rate on mAP, fmAP or MOTA. The July 2022 CTC used
  for the call for proposals (WG 2 N231,
  [zip](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/139_OnLine/w21834.zip))
  keeps detection, instance segmentation and tracking only (OpenImages, FLIR,
  TVD, SFU-HW-Objects-v1), with VTM 12.0 as anchor. Why pose was dropped is
  not stated in the documents read.
- **Results so far.** The call for proposals drew 17 responses; 10 used
  region-of-interest coding at the encoder; the best proposal per task
  reached −57% (tracking), −45% (instance segmentation) and −39% (detection)
  BD-rate against the anchor (WG 2 N248, CfP response report, 2022-10-28,
  [zip](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/140_Mainz/w22071.zip)).

Relevance to PointStream: VCM's own reference software is already
object-centric in a crude form (RoI retargeting, background filtering), and
RoI coding was the majority approach in the call. A paper claiming an
object-centric codec judged by machines must cite this and, ideally, compare
against it.

## How FCM judges a codec

Source: **Common test and training conditions for FCM**, WG 04 N0848 (w26666),
2026-08-21, MPEG 155
([zip](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/155_Geneva/w26666.zip));
predecessor N0777 (w25972, 2026-02-13).

FCM codes intermediate features of a split network (backbone on the device,
head in the cloud), so its codec is not comparable to PointStream's. Its
*evaluation* is the more careful of the two, and three of its rules carry
over directly:

- **Performance points (PPs).** A proposal must hit four task-metric targets
  per class. PP1 is "the task performance achieved when the respective
  network is run without introducing splitting", i.e. on the uncompressed
  data; PP2–PP4 step down. BD-rate is computed only over this range, "so that
  BD-rate computations are performed on overlapping curves and on a relevant
  range of task performance, i.e., not substantially below the performance
  obtained when inferencing with an unsplit network directly on the dataset".
  In N0848, PP4 is 90–95% of PP1: OpenImages detection mAP@0.5 79.3 → 75.3;
  TVD MOTA 50.3 → 46.2; HiEve MOTA 34.0 → 31.3 (1080p); SFU-HW class A/B
  mAP@0.5:0.95 62.5 → 56.3; PandaSet mIoU 46.0 → 41.4.
- **No number without overlap.** BD-rate follows JVET-H0030 with no
  extrapolation; where curves do not overlap, the template outputs 1e308
  (shown as `####`) instead of ±100%, and that poisons every average it
  enters. An optional "clipped" BD-rate integrates only up to the unsplit
  performance.
- **Two anchors.** The *feature anchor* (the FCM test model, FCTM) and the
  *remote-inference anchor*: compress the video with a conventional codec,
  decode, run the whole network. In N0848 both inner codec and remote
  anchor are HM-18.0 (low delay; all intra for images), with a rescaled
  variant; in N0777 both were VTM-23.3. FCTM v12.0 against the remote anchor:
  −93% (OpenImages detection and segmentation), −88% and −85% (SFU-HW C, D),
  −97% (TVD).
- **Robustness of the judge.** A supplementary "shifted-confidence threshold"
  result re-runs the tracker's head on the same bitstreams with a different
  detection threshold, to check that a gain is not an artefact of the
  detector's yes/no boundary.
- **Tasks and data.** OpenImages V6 (5,000 images each for detection and
  instance segmentation, mAP@0.5); SFU-HW-Objects-v3.2 (detection,
  mAP@0.5:0.95; also with YOLOX since N0848); TVD (three videos, MOTA); HiEve
  (five videos, MOTA); PandaSet (36 videos, mIoU). All against the datasets'
  labels.

## Related evaluation practice outside MPEG

| Work | What it judges | Labels | Anchor and statistic |
|---|---|---|---|
| Hu, Yang, Yang, Duan, Liu, *Towards Coding for Human and Machine Vision: A Scalable Image Coding Approach*, ICME 2020 ([arXiv:2001.02915](https://arxiv.org/abs/2001.02915)) | Facial landmarks (Bulat & Tzimiropoulos detector) on decoded VGGFace2 faces; an edge-map base layer for machines, pixels for humans | **Pseudo**: "Detection results on the original data are served as ground truth"; normalized point-to-point error (NME) | JPEG; NME against bpp, no BD statistic |
| Choi & Bajić, *Scalable Image Coding for Humans and Machines*, TIP 2022 ([arXiv:2107.08373](https://arxiv.org/abs/2107.08373)) | Detection (YOLOv3, base layer), segmentation (Mask R-CNN, enhancement), reconstruction | Dataset labels | VVC, HEVC and learned codecs; BD-rate and BD-mAP |
| Choi & Bajić, *Scalable Video Coding for Humans and Machines*, MMSP 2022 ([arXiv:2208.02512](https://arxiv.org/abs/2208.02512)) | Detection on the base layer of video | Dataset labels | HEVC, VVC; notes mAP–rate curves "are not always concave, or even monotonic", so BD-rate-mAP can be undefined |
| Siarohin et al., *First Order Motion Model*, NeurIPS 2019 ([arXiv:2003.00196](https://arxiv.org/abs/2003.00196)) | Motion transfer: Average Keypoint Distance (AKD) between keypoints detected on the source and on the generated video; Missing Keypoint Rate (MKR); identity distance (AED). For bodies (Tai-Chi-HD) the detector is OpenPose (Cao et al.) | **Pseudo** (same detector on both) | Not a codec evaluation; AKD later became a generative face video coding metric |
| Chen, Chen, Wang, Ye, *Generative Face Video Coding Techniques and Standardization Efforts: A Review*, DCC 2024 ([arXiv:2311.02649](https://arxiv.org/abs/2311.02649)) | JVET's GFVC test conditions: 50 VoxCeleb sequences at 256², 250 frames; VTM LDB at QP 45–52 as anchor; quality by DISTS and LPIPS | — (human-viewing metrics) | VTM; rate–DISTS/LPIPS curves |
| Chen, Yin, Konuko, Valenzise, Zhang, Wang, Ye, *Generative Models at the Frontier of Compression: A Survey on Generative Face Video Coding*, 2025 ([arXiv:2506.07369](https://arxiv.org/abs/2506.07369)) | Correlation of quality metrics with subjective scores on GFVC output; AKD and AED included | — | AKD correlates poorly with viewers (PLCC 0.44 against DISTS 0.86 and LPIPS 0.85, its table 5): it measures motion fidelity, not quality |
| Jin et al., *Semantically Video Coding* ([arXiv:2201.10162](https://arxiv.org/abs/2201.10162)) | Detection, pose estimation (stacked hourglass, PCK), action recognition on a structured bitstream | Dataset labels | VVC |

GFVC's test conditions use human-viewing metrics only; the keypoint metrics
(AKD) come from face reenactment, and are not part of a standard protocol.
H3c2's judge is FOMM's AKD and Hu et al.'s NME applied to hands.

## What this means for PointStream's judge

1. **H3c2's pseudo-labels flatter the keypoint stream.** Against WiLoR on the
   source, the keypoint stream's only error is H2's coding (smoothing, 15 Hz,
   quantization); WiLoR's own error is invisible, and the pixel coders are
   charged for every way they move WiLoR's output, including moves toward the
   truth. Against human labels both pay WiLoR's error, so the margin can only
   shrink (or grow, if blur moves WiLoR's output toward its prior less than
   coding does; it cannot be predicted). H3c2's 69% stands as what it is, a
   pseudo-label result, and needs a labelled counterpart before the paper
   calls it a machine-task result.
2. **The judge must not be the encoder's estimator.** VCM fixes the task
   networks, independent of the codec. If the stream carries WiLoR's joints,
   the judge is HaMeR (or the reverse), on the decoded pixels for the pixel
   coders and on the rendered or projected joints for the keypoint stream.
   A judge that is the encoder's own estimator measures self-agreement.
3. **Report only the relevant range.** FCM's rule: BD-rate where the task
   metric is within about 10% of the source's. For hands that is where PCK on
   the decoded video is at least 90% of PCK on the source; a keypoint win at
   an accuracy the source never had is meaningless, and one far below the
   source's is not the regime anyone would deploy.
4. **Keep the human axis beside it.** VCM reports PSNR with every machine
   result, and its requirements include hybrid human-and-machine use. The
   scalable-coding papers give PointStream its framing: the keypoints are a
   machine base layer (judged by the task metric at the base rate), and the
   pixels are the enhancement for human viewing (judged by LPIPS, DISTS and
   weighted PSNR at the total rate).

## Proposal: PointStream's machine-judge protocol

Applies to every keypoint mode from K4 on, and as a re-scoring of H3c2. Each
use still gets its own entry in [experiments](experiments.md) with its rule
written before its run.

**Statistic.** Per item and task metric: BD-rate (PCHIP of log rate over the
overlapping metric range, project convention) of the mode against the anchor
envelope (upper concave hull of the anchor's encodes), restricted to the
**machine range**: from the metric on the uncompressed source (as FCM's PP1)
down to 90% of it (FCM's PP4 is 90–95%). Where curves do not overlap in that
range: "better/worse at every rate" (as FCM's `####`), never a number. Mean
over items with a bootstrap interval by item, as H3c2.

**Rate.** Everything the decoder needs for the judged content, billed as in
H3c2 (keypoints, shapes, side information). For a scalable claim, two rates:
the base layer alone (machine metric) and base plus enhancement (human
metrics).

**Anchors.**
- The pixel envelope already used (SVT-AV1 and DCVC-UF, their concave hull):
  the strongest anchor PointStream has and the one its decisions use.
- **VTM**, random access, with VCM's QP spacing (six QPs, 4 apart): the
  recognized anchor for a coding-for-machines claim (VCM's requirement is
  stated against VVC). Plain VTM, not VCM-RS: VCM-RS is in MPEG's
  member-only repository (`git.mpeg.expert`), so whether it can be run
  depends on access through AAU's MPEG participation (the user's call). Run
  only for the machine-judge items, once a mode reaches the paper. This is
  the "correct invocation" AGENTS.md asks for before VVC joins.

**Tasks, metrics and labels.**

| Mode | Task metric (on decoded content) | Labels (primary) | Judge (independent of the encoder's estimator) | Secondary |
|---|---|---|---|---|
| Hand pose, egocentric | PCK@0.05, 0.1, 0.15 of the hand box, all and visible joints (HaMeR's HInt protocol, already used in H2); mean 2D joint error over the box (H3c2) | **HInt** VISOR test keypoints (human, 21 joints with visibility), on the HInt frames that fall inside an item's coded window | HaMeR if the stream carries WiLoR (H2's choice), WiLoR if HaMeR | H3c2's pseudo-labels (judge on source), reported beside, as FOMM's AKD and Hu et al.'s NME |
| Hand pose, 3D (if claimed) | MPJPE after root alignment | **HOT3D** motion-capture hands | as above | — |
| Ball (tennis, table tennis) | Detection precision, recall and F1 with a true positive at positioning error ≤ 5 px at 1280×720 (TrackNet's definition), scaled to the item's resolution; for the keypoint stream, its decoded positions | **TrackNet** (dense, every frame); RacketVision, OpenTTGames (sparse) | a ball detector other than the one that fed the encoder (TrackNet family or another) | trajectory error on frames with a ball |
| Racket pose | PCK on RacketVision's 5 racket keypoints | **RacketVision** (sparse) | an estimator not used by the encoder | — |
| Body pose, tennis players | COCO OKS-AP (the pose benchmark's own mAP) and PCK@0.1 of the box | **None in our datasets**: pseudo-labels from a strong estimator on the source, as Hu et al. and FOMM | a different estimator family from the one that fed the encoder (e.g. ViTPose against an RTMPose or YOLO-pose stream), so a pseudo-label is never also the stream | if a reviewer needs human labels: HiEve's 14-keypoint manual poses (licence below), as a general check |
| Masks (PLAN K2's "mask on original against decoded") | mIoU (VCM's and FCM's segmentation metric) of a fixed segmenter on decoded frames | dataset masks (VISOR human tier), never PointStream's segmenter (project rule) | a segmenter not used by the encoder | — |
| Player tracking (optional) | MOTA (VCM, FCM); HOTA beside | as body pose | an off-the-shelf tracker | — |

**Robustness.** As FCM's shifted-confidence anchor: re-score each result with
a second judge setting (detection threshold, or the other estimator) and
check that the sign of the BD-rate holds.

**First uses (no runs here; each needs its own entry).**
1. *H3c2 relabelled* (CPU first, then GPU only if items qualify): count the
   HInt VISOR test hands inside the four pilot items' windows and the 34
   items of evaluation set v2. HInt frames are VISOR's sparse keyframes
   (B1: all 1,786 HInt frames in the dense-mask videos it surveyed are keyframes), so few may fall in a
   240-frame window; if fewer than about 30 hands, widen the windows or
   choose items by HInt coverage. Then re-judge H3c2's saved decodes and
   joints with HaMeR against HInt (PCK, BD-rate in the machine range).
2. *K5's ball oracle* scored by TrackNet's F1 on decoded frames against
   TrackNet's labels, at equal rate and as a BD-rate in the machine range.
3. *K6's player oracle* scored by OKS-AP with independent pseudo-labels.

## Datasets VCM and FCM evaluate on

| Dataset | Labels used by MPEG | Content, resolution, frame rate | Licence and access | Fit for PointStream |
|---|---|---|---|---|
| **SFU-HW-Objects-v3.2** (VCM, FCM) | Boxes, COCO classes; detection (VCM mAP; FCM mAP@0.5:0.95). v1 labels: YOLOv3 proposals corrected by hand frame by frame in Yolo_mark ([Choi et al., Data in Brief 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7797526/)); FCM notes the ground truth was revised between v1 and v3.2 | 14 HEVC test sequences (classes A–D, O), clips of 33–129 frames, 24–60 fps; class sizes from HEVC's CTC | Labels CC BY 4.0 ([Mendeley](https://data.mendeley.com/datasets/367kty6nw7/3)); videos from the JCT-VC FTP, licence per sequence | General only. BasketballDrive, BasketballDrill and BasketballPass are sports with people, but indoor basketball, boxes only |
| **TVD** (VCM, FCM) | Tracking (MOTA, boxes with IDs) on 3 sequences cut into 7 clips (VCM) or 3 videos (FCM); TVD-I: 166 images with boxes and instance masks | 86 sequences of 65 frames at 3840×2160; tracking clips 1920×1080, 50 fps, 8/10-bit | Tencent / Shenzhen Boyan: personal, non-commercial use incl. research and standards work, no redistribution ([TVD page](https://multimedia.tencent.com/resources/tvd)) | No: street and indoor scenes, tracking boxes |
| **HiEve** (FCM; VCM 2021) | Tracking (MOTA) on 5 videos in FCM. The dataset also has manual 14-keypoint poses (>1M) and actions ([Lin et al., arXiv:2005.04490](https://arxiv.org/abs/2005.04490)); VCM's 2021 CTC used it for pose and actions | 32 videos, crowded events (airport, mall, stadium, …); FCM's five are 1280×720 and 1920×1080, coded at 30 fps | Signed licence agreement with SJTU required (FCM CTTC) | Only as a general human-pose judge; crowded surveillance, not sport or egocentric |
| **OpenImages V6** (VCM, FCM) | Detection, instance segmentation (5,000 validation images each; mAP@0.5 in FCM) | Still images, already JPEG-compressed | Annotations CC BY 4.0, images listed CC BY 2.0 | No (images) |
| **PandaSet** (VCM, FCM, optional) | Semantic segmentation from LiDAR labels projected to the front camera, 10 classes (mIoU) | Driving, 80 frames per sequence, 1920×1080, 10 fps | Not verified (see below) | No |
| **FLIR** (VCM) | Detection on 300 infrared images, with a fine-tuned detector | Thermal images | From MPEG's content server; FLIR's own licence not verified | No |

**Recommendation.** Do not adopt any of them as a PointStream benchmark. Their
tasks (boxes, tracks, semantic masks) are not what keypoints carry, their
content is outside both domains, and their labels for the one relevant task
(HiEve's poses) sit behind a signed licence and are no longer in MPEG's
conditions. PointStream's own labelled sets serve the machine judge better:
HInt and HOT3D (hands), TrackNet, RacketVision and OpenTTGames (ball,
racket). If a reviewer asks for MPEG comparability, the cheapest step is
VTM as an extra anchor on PointStream's own items; running PointStream on
SFU-HW's basketball sequences would need a player-and-ball pipeline on
indoor basketball and gives detection boxes, not poses.

## Searches and what could not be verified

Read in full or in the relevant sections: VCM CTC N0870 and N0789, FCM CTTC
N0848 and N0777, VCM CTC 2021 (N54) and 2022 (N231), VCM requirements N0190,
VCM CfP N220 and its response report N248 (all from mpeg.org's public meeting
pages); the arXiv papers in the tables; SFU-HW-Objects-v1's data paper; the
TVD page; HaMeR (HInt PCK protocol, [arXiv:2312.05251](https://arxiv.org/abs/2312.05251));
TrackNet ([arXiv:1907.03698](https://arxiv.org/abs/1907.03698)); the
Bjøntegaard review by Herglotz et al. ([arXiv:2304.12852](https://arxiv.org/abs/2304.12852)),
which records JVET's move from polynomial fits to PCHIP.

Not verified:
- **JPEG AI's test conditions for computer-vision tasks** (the CTTC's
  "Part B", e.g. WG 1 N100421): ds.jpeg.org failed TLS verification through
  this session's proxy, and WebFetch got HTTP 503. Only JPEG AI's stated scope
  (version 1 is for human viewing; latent-domain tasks are planned) is
  reported.
- **JVET's GFVC common test conditions** (JVET-AG2035): not retrieved from
  jvet-experts.org; the conditions are taken from Chen et al.'s review, which
  describes them. No keypoint metric was found in them.
- **Why VCM dropped pose and action recognition** between N54 (2021) and
  N231 (2022), and VCM's "Evaluation framework and methodologies" N00277
  (2022), which is not on the public page.
- **VCM-RS and the VCM CTC scripts** (`git.mpeg.expert`): member-only, not
  inspected; the exact VCM-RS version of the current anchor is inconsistent
  in N0870's text (v1.2, v1.3) and its template names (2.2).
- JVET-H0030's interpolation method (FCM's BD-rate reference) was not read.
- Licences of PandaSet and FLIR; whether HiEve's test labels are public
  (MPEG distributes them for its five videos).
- No published study of hand-pose accuracy on compressed video was found;
  searches for pose-under-compression codec evaluations found only PCK on a
  structured bitstream (Jin et al.) and keypoint-detection BD-rates in
  [arXiv:2503.21469](https://arxiv.org/abs/2503.21469) (not read in full).
