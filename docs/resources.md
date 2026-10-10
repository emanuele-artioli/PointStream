# Resources

## Storage links

| Link (repository root) | Target | Holds |
|---|---|---|
| `Models` | `/home/itec/emanuele/Models` | Weights, by family (`YOLO/`, `SAM/`, …) |
| `Datasets` | `/home/itec/emanuele/Datasets` | Datasets, their manifests, experiment outputs |

`scripts/link-storage` creates both links. `src/segmentation/storage.py`
resolves them: `PS_MODELS_ROOT` / `PS_DATASETS_ROOT`, else the link, else the
target directory itself (fleet snapshots carry no links). `model_path(family,
name)` returns an existing weight or raises an error naming the expected path.

The fleet's own data root, holding `jobs/` and packed environments, is
`PS_DATA_ROOT`. The fleet sets it in every job; on the hosts it is
`/home/itec/emanuele/pointstream-data`, a link to `Datasets/pointstream-data`.

## Datasets

Two domains with real labels: **egocentric hand-object video** (hands, arms,
the object in hand), worked on first, and **racket sports** (players, rackets,
ball), worked on second ([PLAN.md](../PLAN.md)). Every dataset lives in
`Datasets/<name>` as downloaded archives, never extracted onto the NFS home.
Every dataset has an immutable manifest in `Datasets/manifests/` with each
file's sha256, the source, licence, label format, counts, resolution, fps,
labelled classes, and a smoke read of one labelled sample (overlays in
`Datasets/manifests/smoke/`). The tooling is in `tools/datasets/`.

| Dataset | What is there | Labels | Limits |
|---|---|---|---|
| **EPIC-KITCHENS VISOR** (`EPIC-KITCHENS-VISOR`, 476 GB, CC BY-NC) | VISOR zip plus the 179 EPIC-KITCHENS videos it annotates; 1920×1080, 59.94 or 50 fps | Polygon masks of left/right hand and active objects (open vocabulary, EPIC-100 noun classes), hand-object contact. Sparse: 40.6k frames, 216k masks, about one frame every 78. Dense interpolations: 41% of frames in P01_01, in runs with a median of 449 frames | The 21 test videos have no annotations. Dense masks are interpolations filtered by J&F, not human-drawn. Only active objects are masked. |
| **EgoHOS** (`EgoHOS`, 4.5 GB, no licence stated) | 11,747 still images (Ego4D, EPIC, THU-READ, Escape Room, YouTube), mixed resolution | Left/right hand; first- and second-order interacting objects per hand | Not video: frames sampled sparsely from source videos. The contact boundaries described in the paper are not in the release. |
| **HOT3D-Clips** (`HOT3D`, 208 GB, licence accepted 2026-10-05) | 1,516 train_aria and 467 test_aria clips, 150 frames each; RGB stream `214-1` is 1408×1408 fisheye, rotated 90°, 30 fps | Object masks (amodal, rendered from GT pose; modal, from SAM2), 6DoF object poses, MANO/UmeTrack hand poses and boxes | No hand masks: they must be rendered from MANO through the fisheye model, and MANO stops at the wrist. Hands wear mocap markers. test_aria has no labels. Lab scenes, JPEG frames. Quest3 clips not downloaded. |
| **OpenTTGames** (`OpenTTGames`, 35 GB, CC BY-NC-SA) | 12 videos, 1920×1080, 120 fps | Events; ball coordinates and masks (channels R=table, G=human, B=scoreboard) on 4 frames before and 12 after each event, about 9% of frames | Masks are 320×128 and model-aided: good for checking that the right objects were found, not for grading borders. Rackets are not in the human mask; the umpire is. |
| **RacketVision** (`RacketVision`, 7.6 GB, MIT) | 1,672 clips, 435k frames, 1920×1080, 24–60 fps (badminton, table tennis, tennis) | Ball points on 20.1 / 11.5 / 14.3% of frames; racket box plus 5 keypoints on 9.2 / 3.9 / 4.9%, usually one racket per labelled frame | No masks. The interpolated ball tracks and merged racket predictions are not ground truth. |
| **TrackNet** (`TrackNet`, 2.6 GB, no licence stated) | 95 tennis clips, 19,835 frames, 1280×720, 30 fps | Ball position and visibility on every frame | JPEG frames only, no video. Only a third-party mirror is reachable. |
| **HInt** (`HInt`, 3.8 GB, MIT) | `HInt_annotation_partial.zip`: hand keypoints on EPIC-KITCHENS VISOR frames (1920×1080; 2,780 / 625 / 1,906 hands in train / val / test, from 109 / 90 / 40 of our VISOR videos) and New Days; Ego4D annotations without frames | 21 2D keypoints per hand with existence and occlusion flags, handedness in the file name | Keypoints only, no 3D. Its splits are its own; HaMeR trains on HInt train. 377 EPIC files (`EK_frame_…`) name no video. Downloaded with TLS verification off (the host's certificate expired 2025-05-08; no published checksum): sha256 `ac42d9f8…c7fe` recorded while streaming, size equal to the server's Content-Length. Ego4D frames need an Ego4D licence. |
| **EPIC-KITCHENS-100 hand-object detections** (`EPIC-KITCHENS-hand-objects`, 0.17 GB, Non-Commercial Government Licence; EPIC-KITCHENS CC BY-NC 4.0) | `hand-objects/<P>/<video>.pkl` for the 34 videos of VISOR evaluation set v2 (doi:10.5523/bris.3l8eci2oqgst92n14w2yqi5ytu), plus the release's `EPIC_100_frame_counts.csv` | Detector output (Shan et al. 2020), not labels: per EPIC rgb frame, hand boxes with side, contact state and score (kept down to 0.1), object boxes (down to 0.01); normalised to [0, 1], detector input 456×256 | Used only as an independent check of filled hands (B1b). Hand boxes exclude the forearm VISOR's hands include. No published checksum: sha256 recorded while streaming, size equal to Content-Length (`tools/datasets/download/hand_objects.sh`); every file has the release's frame count. Read by `src/segmentation/hand_objects.py` (no `protobuf` dependency; equal on every field to the official reader on two videos, 18k frames). Manifest `EPIC-KITCHENS-hand-objects.json` (sha256 `711870f2…1ee5`). |

Candidate if needed: SA-Co/VEval SmartGlasses. Not used: DeepSportradar
(images, not video), ENIGMA-51 (no arms; SAM-HQ masks), DAVIS (too small for the
domains).

### What each dataset is used for

| Use | Datasets |
|---|---|
| End-to-end codec evaluation (video plus dense foreground masks) | VISOR (phase 1); OpenTTGames players stage (phase 2) |
| Segmentation training and evaluation | VISOR, EgoHOS; OpenTTGames as an object-level check |
| Handled-object proposer | VISOR (contact), EgoHOS (object orders); HOT3D as cross-test |
| Hand keypoints and foreground encoding | HOT3D (motion-capture oracle); VISOR with an estimated hand pose; HInt (VISOR frames, 2D keypoints) to compare the estimators |
| Racket pose and player prompts | RacketVision |
| Ball tracking and ball encoding | TrackNet (dense), RacketVision, OpenTTGames |

**Provenance tiers.** Every mask records one: `human` (VISOR sparse, EgoHOS);
`interpolated` (VISOR dense); `model_aided` (OpenTTGames); `rendered` (HOT3D,
from motion-capture poses); `sam_from_label_prompt` (SAM 3.1 prompted by a
labelled point, box or keypoint); `sam_text` (SAM 3.1 from a text prompt only).
Results are reported per tier. No data is labelled by hand.

**Modelling decisions.**
- One model per dataset with its native labels (for example VISOR's left and
  right hand), expressed as one domain per dataset in
  `src/segmentation/domains.yaml`: shared code, per-dataset classes and models.
- VISOR and EgoHOS hand masks include the visible forearm and sleeve up to the
  image border (checked visually on the smoke overlays), so no SAM arm fill is
  needed there.
- The ball mask comes from SAM 3.1 pointed at the labelled position, so it keeps
  the ball's real appearance (blur, colour). A rendered disk or streak is the
  fallback where SAM fails, and its rate is measured.
- A TrackNet-style ball tracker is trained or fine-tuned on RacketVision and
  tested on OpenTTGames and TrackNet before it is relied on.
- The handled object is proposed by a hand-object interaction model (VISOR- or
  EgoHOS-trained) whose boxes prompt SAM 3.1. A vision-language model naming the
  object for a SAM text prompt is the fallback.
- Every trained proposer passes a cross-dataset check before it is adopted.

**Open questions.**
- Is the demo rebuilt on VISOR, or kept on the archived copy of its data?

**Hand pose for VISOR** (decided 2026-10-06, [Models](#models)): a
MANO regressor, HaMeR or WiLoR, not DWPose. VISOR has no hand keypoints, and the
codec does not need them: it is scored on pixels inside the dataset masks. They
matter only for choosing the estimator. For that, HInt ([ddshan/hint](https://github.com/ddshan/hint),
MIT) labels 21 2D hand keypoints with occlusion flags on EPIC-KITCHENS VISOR
frames (5.3K hands; train, val and test), New Days and Ego4D. HaMeR trains on
HInt train, so only HInt test is fair to it. Adopted for H and downloaded to `Datasets/HInt` (manifest
`Datasets/manifests/HInt.json`, table above).

**VISOR frame mapping** (B1, fleet job `20261006T193326Z-64057023`, gpu3,
commit `efc516a`, environment `pointstream-20261006T113321Z`; 325 released
sparse JPEGs from 10 videos, inputs `pointstream-data/visor/b1-2026-10-06/mapping-v2/`).
VISOR uses three numberings of one video: decoded frames (0-based, what
PointStream encodes), EPIC-KITCHENS rgb frames, and VISOR's own frames (VISOR
re-extracted with a newer ffmpeg and names every mask by them).
`frame_mapping.json` links VISOR to EPIC frames only on *sparse* frames.

| Video type (videos checked) | EPIC rgb frame k is decoded frame | VISOR frame n is decoded frame n − 1 |
|---|---|---|
| HEVC 1080p 50 fps, EK-100 (4) | k − 1 (139/139) | yes (139/139) |
| H.264 1080p 59.94 fps (3) | nearest to (k − 1)/60 s (86/86) | no (18/86; off by up to 13 on P02_02) |
| H.264 1080p 29.97 fps (2) | (k − 1)/60 s rounded up (72/72) | no (0/72; about two VISOR frames per decoded frame) |
| H.264 1080p 47.95 fps (1, train) | unresolved (26/28 nearest, 23/28 rounded up) | no |

A rule holds on a frame when its decoded frame is within 0.5 grey levels (mean
absolute RGB difference) of the best match to the JPEG; best matches were
0.8–2.6. The EK-55 videos repeat frames in pairs (ties). `visor.epic_frame_to_video_index`
implements the verified rules and raises for any other rate.

Dense masks have no mapping between keyframes: `visor.frame_alignment` places
them by counting from their run's keyframes and records the *drift* (decoded
frames gained or lost against the extraction rate between keyframes). EK-100
has drift 0 on all 2,383 keyframe stretches of the 16 val videos; EK-55
drifts by 1–3 frames on about a quarter. Evaluation scores only exactly placed
frames; training admits drift ≤ 1 ([decision](experiments.md#2026-10-06-visor-frame-drift-exact-frames-for-evaluation-small-drift-for-training)).
Dense polygons are drawn on an 854×480 canvas and scaled to 1080p; on P32_07's
keyframes they match the human 1080p masks at IoU 0.97–0.995.

**VISOR evaluation set v2** (`experiments/visor/eval_set.json`; source
`pointstream-data/visor/b1-2026-10-06/evalset-v2/eval_set.json`, sha256
`0d25301a…ec2`; dense archive `visor-val-dense.tar`, `a2d9e0cb…c3c`). 34
items, one per validation video: 15 EK-100 (50 fps) and 19 EK-55 (59.94 fps),
240 frames each. Every frame is labelled and exactly placed on the video, and
no hand a human labelled at an end of its dense run is missing
(`visor.hand_gaps`); the window is picked content-blind by
sha256("pointstream-b1:<video>"). Masks carry provenance `interpolated`;
objects other than hands can still be missing (108 at the 78 human-labelled
frames inside the items). Rule and outcome in
[experiments](experiments.md#2026-10-07-visor-evaluation-set-v2-and-a-sam-fill-for-missing-hands).
Set v1 (16 EK-100 items, `b994c531…d9cb`) is superseded.

**Archived.** `tennis_games`, `Egocentric-10K` and the derived
`pointstream-demo` are in `Datasets/archive/pre-reset-2026-10-05/`: moved,
read-only, not deleted, with `MANIFEST.json` (104,884 files, 72.5 GB, sha256
each). `src/segmentation/domains.yaml` still lists clips from them as defaults
until the per-dataset domains replace it.

### Using VISOR

The rules every session that reads, scores on, or trains from VISOR follows.
Decided in B1 (2026-10-06/07); evidence and reasons in
[experiments](experiments.md#2026-10-06--b1-visor-frame-mapping-and-evaluation-set),
[drift decision](experiments.md#2026-10-06-visor-frame-drift-exact-frames-for-evaluation-small-drift-for-training)
and [set v2 decision](experiments.md#2026-10-07-visor-evaluation-set-v2-and-a-sam-fill-for-missing-hands);
illustrated in the B1 report ("VISOR frame drift" artifact).

**Reading.** Use `src/segmentation/visor.py`; never index frames by hand.
- `load_annotations` reads a sparse JSON or a dense `_interpolations.zip`;
  `clip_masks` returns `ClipMasks` with VISOR's classes (`left hand`,
  `right hand`, `active object`; the open-vocabulary name is each instance's
  `label`; a glove on a hand is that hand), per-frame `labelled` flags and a
  provenance per mask: `human` (sparse file) or `interpolated` (everything in
  the dense file, its keyframes too).
- Dense polygons are drawn at 854×480 and scaled to 1080p: their edges are
  about two pixels coarse (IoU 0.97–0.995 against the human 1080p masks).
- An unlabelled frame is unknown, not empty; `evaluate.compare` scores only
  labelled frames.

**Frames.** VISOR names masks by its own frame extraction. Only sparse frames
link to the video (`frame_mapping.json` → EPIC rgb frame →
`epic_frame_to_video_index`, verified per rate in the table above). Dense
frames are placed between their run's keyframes by `frame_alignment`, which
records each frame's *drift*: how far VISOR's numbering slipped against the
video between the two keyframes, which bounds the placement error.
- EK-100 (50 fps): drift 0 everywhere; VISOR n is decoded frame n − 1.
- EK-55 59.94 fps: exact on about three quarters of stretches; elsewhere off
  by 1–3 frames.
- EK-55 29.97 fps: two VISOR frames per decoded frame; 47.95 fps (P17_01) has
  no verified rule and the reader raises; P12_04 is 720p video with 1080p masks.

**Missing hands and objects.** VISOR drops or cuts short each object's dense
track when its interpolation scored poorly, so 22–30% of dense val frames lack
a hand a human labelled at an end of the run, and most frames lack some
labelled object. `hand_gaps(dense, sparse)` lists the frames where a labelled
hand is missing. A hand that comes and goes between keyframes is invisible to
every label. HInt's VISOR frames are all sparse keyframes, so HInt cannot fill
these gaps.

**Evaluation (B2, D1, any codec or segmenter score).**
- Use evaluation set v2 (`experiments/visor/eval_set.json`, below) and nothing
  else: 34 windows of 240 frames, one per validation video, every frame exactly
  placed and hand-complete. B2 decodes each window from
  `first_video_index` and may check its sparse JPEGs (in the archive) against
  the decoded frames.
- Score only exactly placed frames. Adding other footage (more EK-55
  stretches) requires `clip_masks(..., exact_only=True)` and a recorded
  decision.
- Objects other than hands can still be missing: report it with every result.
- Report per provenance tier, and every number with and without B1b's fill:
  mask set `visor_dense_sam_fill` (tier `sam_from_label_prompt`, objects only;
  hands were not adopted), `pointstream-data/jobs/fleet/inbox/20261007T222646Z-3a7ac6ef/full/published.tar`
  (sha256 `049977f0…799a`, record `merge.json` `1c705e4a…4bea`). It cuts the
  labelled objects missing from 33% to 1.6% of each frame's; B2 with both sets:
  `pointstream-data/visor/b1b-2026-10-07/b2-rescore-339c918/`. The "without"
  number is the fair one for D1, where SAM 3.1 is a contestant.
- Codec baseline (B2, `experiments/visor/b2.py`): SVT-AV1 preset 4 and
  DCVC-UF HT-L curves on all 34 items, keyed by mask set; the streams are
  published in the full jobs' `published.tar`, so a new mask set (B1b's
  fill) is scored with `b2 run --streams` without re-encoding. PointStream is
  compared at equal rate with the better of the two per metric
  ([outcome](experiments.md#2026-10-07--b2-baseline-rate-distortion-on-visor)).
- Floor for segmenters: "hold the first frame" (the item's first-frame masks
  copied to every frame) scores, mean over the 34 items, foreground J 0.459,
  F 0.370 (EK-100 0.403, EK-55 0.504; per item 0.03–0.85; left hand J 0.374,
  right hand 0.346, active object 0.467; job `20261007T081136Z-71468647`).
  A segmenter must clearly beat it per scope to count; report it as the first
  row of the D1 table.

**Training (E1, F1, generators).**
- Train on the train split; keep the evaluation videos out.
- Admit dense frames with drift ≤ 1; every exported mask records its drift so
  the cutoff can change without re-exporting.
- Instance crops of labelled objects are safe. Background frames ("foreground
  removed") and segmenter targets are not, where `hand_gaps` reports a missing
  hand: exclude those frames, or a model learns hands as background. B1b's
  hand fill was not adopted (it let departed hands go in 4 of 6 cases and
  some hand tracks moved onto the other hand), so these frames stay excluded
  until a hand fill passes.

## Models

Each family directory in `Models` has a `MANIFEST.json` (sha256, size, source
path and URL) written by `tools/models/place.py`, which never overwrites.

**SAM 3.1 Object Multiplex**
- Meta's `sam3` pinned to `2345a4ad109ac29c569da749c91d84f10dc08c40`, installed
  from git into the environment. `Sam31SequenceSegmenter` checks the installed
  commit (`direct_url.json`) and every file against its `RECORD` hash;
  `SAM31_SOURCE_ROOT` selects a git checkout instead.
- `Models/SAM/sam3.1_multiplex.pt`, sha256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` (copied
  from the Hugging Face snapshot `daa63191`, with its `sam3.1_config.json`).
- **GPUs: Ada and A6000 only** (`gpu_models: ["RTX 6000 Ada", "RTX A6000"]`).
  On the RTX 8000 it falls back to math attention (39.5 GiB, 2.2× slower); on
  the GV100 it runs out of memory ([model–GPU table](fleet.md#modelgpu-table)).
- **Mask prompts** (B1b, `src/segmentation/sam31_tracker.py`). The multiplex
  predictor's public API takes text, boxes and points only (its `add_mask`
  raises for the multiplex model). `Sam31MaskTracker` builds SAM 3.1's own
  tracker with sam3's `build_sam3_multiplex_video_model` and loads the same
  checkpoint's `tracker.model.*` and `detector.backbone.vision_backbone.*`
  weights (931 of 931 keys). Two limits of the pinned code: the tracker must
  ask its backbone for its own two heads only (asking for the detector's head
  as well breaks `forward_image`), and a multiplex state takes mask prompts on
  its first frame only, so each prompted frame is its own session. About 5
  frames/s with the backbone at 1080p on an RTX A6000, 5–6 GiB.

**YOLOE-26**
- Only the `n` and `x` segmentation weights are in `Models/YOLO`
  (`yoloe-26n-seg.pt`, `yoloe-26x-seg.pt`), with the `mobileclip2_b.ts` text
  encoder, which must be bound locally (`bind_local_text_encoder`).
- ByteTrack loses fast hands; mask-IoU association works.

**Neural video codec: DCVC-UF** (decided 2026-10-06)
- Microsoft's [DCVC](https://github.com/microsoft/DCVC) at
  `cbdae87a5445114cdc7f48816da63ea80bdeac40`: DCVC-UF (CVPR 2026), the newest
  model there, after DCVC-RT (CVPR 2025) and DCVC-FM (CVPR 2024). It writes a
  real bitstream (rANS entropy coder), so rate is counted from bytes. Three
  structures: LD (low delay), HT-S and HT-L (chunks of 8 frames).
- Weights in `Models/DCVC`: `cvpr2026_image.pth.tar`, `cvpr2026_video_{ld,hts,htl}.pth.tar`,
  copied from the pre-reset download (2026-09-29; DCVC publishes no checksum,
  hashes in the manifest).
- **GPUs: Ada, A6000 or RTX 8000, never GV100** (upstream has no Volta path).
  Decode on the same GPU class as the encode: the two extension variants use
  different kernels, and bit-exactness across classes is not established.
- `src/codecs/dcvc_uf_worker.py` encodes and decodes one stream. It is ported
  from the pre-reset adapter, which was correct but never passed its provenance
  gate. It runs as its own process because DCVC's package is also called `src`.
  The decoder sees only the container bytes and the checkpoints.
- Training (audited 2026-10-10 for G5b). `train_video.py`, `train_image.py`
  and `training.md` at the same revision train in plain PyTorch: in training
  mode the models never call the CUDA inference extensions (the reference
  frame enters by `pixel_unshuffle`), and the extra imports are numpy,
  Pillow and scipy, all in the environment. A fine-tuned state dict loads
  strictly into the inference models, which rebuild their entropy tables
  with `update(0)`, so the worker codes with it unchanged. G5b's fine-tuning
  (`experiments/background/g5b_train.py`) runs, like the worker, as a script
  in DCVC's tree. No new environment and no new dependency.
- Not chosen. HNeRV fits one network per video with no temporal model: the
  pre-reset latents cost 76–79 kbps at 18–21 dB at 240p, no better than AV1.
  GLC-video reports estimated rather than coded rates. GVC-RT and MTTF were
  never installed.

**Quality metrics** (B2, `src/codecs/quality.py`)
- LPIPS uses torchmetrics' AlexNet heads (in the environment) and the
  torchvision AlexNet backbone `Models/LPIPS/alexnet-owt-7be5be79.pth`
  (sha256 `7be5be79…ee02`, its prefix matches the published file name; copied
  from the shared torch hub cache on 2026-10-07), loaded from that path, never
  downloaded.
- DISTS (G5c) uses torchmetrics 1.9's `DISTSNetwork` and its alpha/beta
  (`dists_models/weights.pt` inside torchmetrics, sha256 `f5e65c96…8218`)
  on the torchvision VGG16 backbone `Models/DISTS/vgg16-397923af.pth`
  (sha256 `397923af…5bf0`, copied from the shared torch hub cache on
  2026-10-10, manifest beside it), loaded from that path, never downloaded.
  On a region, every layer's statistics are weighted by the region
  (`src.codecs.quality.masked_dists`). No new dependency.
- VMAF: the environment's ffmpeg has no libvmaf; the hosts' `/opt/local/bin/ffmpeg`
  (7.1.1, libvmaf 3, built-in `vmaf_v0.6.1`) has it on gpu1–gpu6. Results
  record its path, sha256 and the libvmaf it links.

**Hand pose estimators**
- HaMeR ([geopavlakos/hamer](https://github.com/geopavlakos/hamer) at
  `3a01849f4148352e9260b69bf28b65d1671a4905`), CVPR 2024, ViT-H. Weights in
  `Models/HaMeR` (`hamer.ckpt`, `model_config.yaml`, `dataset_config.yaml`,
  `mano_mean_params.npz`), copied from the official demo archive already in
  `Datasets/HaMeR/Hand-Texture-Module/`.
- WiLoR ([rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) at
  `fcb911312a38fa8badd30d9656a167485d61b8f9`), CVPR 2025, newer and faster, with
  its own YOLO hand detector. Weights in `Models/WiLoR` (`wilor_final.ckpt`,
  `detector.pt`) from the authors' Hugging Face space, LFS sha256 verified.
- Both regress MANO pose and shape, the representation HOT3D's motion-capture
  oracle uses, so the two hand evaluations in H compare like with like. Both
  take a hand box and handedness, which VISOR masks (and later PointStream's
  segmenter) supply, so their demo detectors (ViTDet, ViTPose) are not needed.
- DWPose is not used: it is 2D only, its person detector needs a visible body,
  and egocentric frames show hands and forearms.
- Both run on all four GPU classes.
- Which of the two is not settled by the literature. WiLoR's paper beats HaMeR
  only on lab benchmarks, by small margins (FreiHAND PA-MPJPE 5.5 vs 6.0 mm,
  HO3Dv2 7.5 vs 7.7 mm); it does not evaluate on HInt, and no paper found
  compares the two on egocentric data. HaMeR trains on HInt train (VISOR
  frames), WiLoR does not. Establishing which is better on egocentric video is
  a small contribution of its own: H compares them and the result is a paper
  table ([PLAN](../PLAN.md#h-foreground-encoding)).

**MANO**
- `Models/MANO/MANO_{LEFT,RIGHT}.pkl`: chumpy-free copies of the official v1.2
  models (`Datasets/MANO/mano_v1_2/models`), written by
  `tools/models/mano_dechumpy.py`. chumpy does not import on Python 3.12;
  `shapedirs` was a chumpy `Select` and is evaluated exactly. `MANIFEST.json`
  holds the source and output sha256 and each array's hash. The HOT3D smoke
  validates the copies: rendered MANO hands match the labelled amodal boxes.

## Environments

**One environment, `pointstream`**, for every phase-1 component: Python 3.12,
torch 2.10.0+cu128. The audit found conflicts, but none that needed a second
environment; each is resolved below.

| File | Holds |
|---|---|
| `environment.yaml` | conda-forge packages (Python, ffmpeg 8.1 with libsvtav1 and libdav1d, SVT-AV1, dav1d) and `env/requirements.txt` |
| `env/requirements.txt`, `env/no-deps.txt` | Top-level pins; the packages installed without their declared dependencies, and why |
| `env/build-environment.yaml` | Build-only prefix (CUDA 12.8 toolkit, gcc 13) for the DCVC-UF extensions; nothing runs from it |
| `env/build.sh` | Builds both prefixes, the extensions and the vendored trees on host-local disk, then writes the locks (resumable by step) |
| `env/patches/dcvc-build-targets.patch` | DCVC's extension build takes its GPU targets from the build instead of probing the build host |
| `env/locks/` | `conda list --explicit --md5` and `pip freeze` of both prefixes, `pip check`, and `pointstream.opt.json` (DCVC, CUTLASS and WiLoR revisions, wheel hashes) |

Packed for fleet staging: `pointstream-data/environments/pointstream-20261006T113321Z.tar.gz`,
sha256 `44835688156ec6dd5a96ae068e636bb65174597a8ca5f34a6efc33b12bd751b9`
(4.9 GB; 8.5 GB unpacked). Built from the recipe alone on gpu6, 2026-10-06, at commit `fa68777`.

Current pack, with G5's NVRC: `pointstream-data/environments/pointstream-20261009T214002Z.tar.gz`,
sha256 `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661` (4.9 GB). It
is the 2026-10-06 pack extended on gpu6, 2026-10-09: a copy of its prefix, plus the
six packages at the end of `env/no-deps.txt` installed without dependencies, plus
`env/nvrc.sh`. The recipe now does both (`env/build.sh`), and the locks differ from
the 2026-10-06 locks by exactly those six packages and `opt/NVRC`. Earlier jobs keep
naming the pack they ran with.

**Components and what they pin**

| Component | Source | Stated requirements | In the environment |
|---|---|---|---|
| SAM 3.1 | `sam3` @ `2345a4a` | Python 3.12, torch 2.10.0 cu128 (README), numpy<2, timm>=1.0.17 | as stated |
| YOLOE-26 | `ultralytics==8.4.6` | torch>=1.8, opencv-python | `--no-deps` |
| VISOR reader | PyAV 19.0.1 | — | decodes the EPIC-KITCHENS videos frame-accurately; `src/segmentation/visor.py` |
| HOT3D-Clips | `hand_tracking_toolkit` @ `bc628e9` | numpy, scipy, torch, opencv-python, webdataset | `--no-deps`; FISHEYE624 cameras, MANO via smplx, numpy rasterizer. The hot3d repo (`146b34a`) documents the clip format; its pixi environment (Python 3.10, torch 2.1, projectaria_tools) serves the VRS release, not the clips |
| HaMeR | `hamer` @ `3a01849` | smplx==0.1.28, chumpy, mmcv==1.3.9, detectron2, pyrender | `--no-deps` |
| WiLoR | `opt/WiLoR` @ `fcb9113` | Python 3.10, torch cu117, ultralytics==8.1.34, chumpy | vendored tree on `sys.path` (no packaging) |
| DCVC-UF | `opt/DCVC` @ `cbdae87`, CUTLASS v4.4.1 | Python>=3.12, torch 2.9.1 and CUDA 13.0 tested, extensions built for the build host's GPU | two extension builds in `opt/dcvc-extensions/` |
| SVT-AV1 | conda-forge `svt-av1` 4.2.0, `ffmpeg` 8.1, `dav1d` 1.5 | — | `SvtAv1EncApp`, `dav1d`, `ffmpeg` on the prefix's `bin` |
| NVRC (G5 oracle) | `opt/NVRC` @ `ccc432d`, `env/patches/nvrc-pointstream.patch` | Python 3.13, torch 2.6.0 cu124, compressai==1.2.6, accelerate==1.3.0, pytorch-msssim==1.0.0, timm==0.9.16, deepspeed==0.16.2 | torch 2.10 and timm 1.0.30 as installed; compressai 1.2.8 (conflict 9); deepspeed left out (patched optional); run as a script from its directory |

**Conflicts and their resolution**
1. *CUDA.* DCVC-UF is tested on CUDA 13.0; every host runs driver 535 (CUDA
   12.2). torch 2.10.0+cu128 runs on it through CUDA minor-version
   compatibility and carries SASS for sm_70 to sm_120. The extensions are
   compiled with CUDA 12.8 to SASS only, because the driver cannot JIT newer PTX.
2. *DCVC extension targets.* Its build uses `-arch=native` and a compile-time
   SM, so it fits only the GPU it was built on. The patch names the targets.
   Two variants are built: `sm80` (SASS for sm_70, 75, 80 and 86, sm_80 hint
   tables) and `sm89` (Ada hint tables). The worker loads the one for the
   device. DCVC then dispatches at run time: plain PyTorch below sm_75, CUTLASS
   Sm75 kernels on Turing, Sm80 above.
3. *Two cv2 builds.* ultralytics, hand_tracking_toolkit and hamer declare
   `opencv-python`; PointStream uses `opencv-python-headless`. Both install
   `cv2/` over each other, so those three are installed `--no-deps` with their
   other dependencies listed explicitly. `pip check` reports exactly these
   omissions (`env/locks/pointstream.pip-check.txt`).
4. *chumpy.* MANO pickles need chumpy, which fails on Python 3.12. They are
   converted once (Models above); nothing imports chumpy.
5. *HaMeR's demo stack.* mmcv==1.3.9 (ViTPose) and detectron2 (ViTDet) are demo
   detectors. They are left out: hand boxes and handedness come from masks.
6. *WiLoR's ultralytics==8.1.34.* Its detector checkpoint must load under
   8.4.6; the WiLoR smoke checks it.
7. *pkg_resources.* sam3 imports it, and setuptools removed it after 80.x:
   `setuptools==80.9.0`.
8. *Two packages named `src`.* DCVC's top-level package collides with
   PointStream's. The worker interface (one process per encode or decode,
   JSON plan in, JSON report out) keeps them apart in the same environment.
9. *NVRC's pins* (G5, 2026-10-09). compressai 1.2.6 has only a source archive,
   whose build fails here; 1.2.8 has a cp312 wheel (manylinux_2_34; the hosts
   run glibc 2.35) with the same entropy-model, rANS and transform API. Its
   resolver would replace torch, so compressai and its import-time
   `torch-geometric` (with `pandas` and `xxhash`) are installed without
   dependencies; its declared `pybind11` and `tomli` are build-time only and
   appear in `pip check`. NVRC's torch 2.6 and timm 0.9.16 pins are its
   authors' setup; it trains and writes and decodes its bitstream under
   torch 2.10 and timm 1.0.30 (dev check on gpu6, Ada). deepspeed serves only
   a FLOPs profiler; the patch makes it optional, and makes a fourth PNG
   channel mask the training loss.

**Earlier environments.** `~/.conda/envs/pointstream` (Python 3.10, torch
2.2.2) stays as the interpreter of the fleet workers and the dispatcher only;
`experiments/jobs` must keep running on it. Workloads run from the packed
environment. `pointstream-sam31`, `pointstream-dcvc` and `pointstream-neural`
were deleted on 2026-10-06; their exact specs are in
`Datasets/archive/conda-envs-2026-10-06/`, as are those of
`pointstream-diffueraser` (DiffuEraser video inpainting from the pre-reset demo),
deleted the same day because nothing in phase 1 used it. If inpainting becomes a
background candidate, it is audited into the environment like any component.

## Archive

The code and documentation before this layout are tagged
`archive/pre-reset-2026-10-05` in this repository (`cb7d1d8`). The manuscript
at that point is Overleaf `main` `e3d52d3`, tagged
`archive/pre-reset-2026-10-05` in the server's paper clone only, because
Overleaf rejects tags.
