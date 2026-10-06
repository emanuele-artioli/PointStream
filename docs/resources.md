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

Candidate if needed: SA-Co/VEval SmartGlasses. Not used: DeepSportradar
(images, not video), ENIGMA-51 (no arms; SAM-HQ masks), DAVIS (too small for the
domains).

### What each dataset is used for

| Use | Datasets |
|---|---|
| End-to-end codec evaluation (video plus dense foreground masks) | VISOR (phase 1); OpenTTGames players stage (phase 2) |
| Segmentation training and evaluation | VISOR, EgoHOS; OpenTTGames as an object-level check |
| Handled-object proposer | VISOR (contact), EgoHOS (object orders); HOT3D as cross-test |
| Hand keypoints and foreground encoding | HOT3D (motion-capture oracle); VISOR with an estimated hand pose |
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
- Which hand-pose estimator gives VISOR keypoints (HaMeR is in `Datasets/HaMeR`)?

**Archived.** `tennis_games`, `Egocentric-10K` and the derived
`pointstream-demo` are in `Datasets/archive/pre-reset-2026-10-05/`: moved,
read-only, not deleted, with `MANIFEST.json` (104,884 files, 72.5 GB, sha256
each). `src/segmentation/domains.yaml` still lists clips from them as defaults
until the per-dataset domains replace it.

## Models

**SAM 3.1 Object Multiplex**
- Meta's source pinned to `2345a4ad109ac29c569da749c91d84f10dc08c40`, checked
  out at `~/.cache/sam3-meta`, run by `~/.conda/envs/pointstream-sam31`.
- Checkpoint `sam3.1_multiplex.pt`, sha256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`. It
  belongs in `Models/SAM`. Until it is moved there, the code falls back to the
  Hugging Face cache snapshot (`SAM31_CHECKPOINT` overrides both).

**YOLOE-26**
- Only the `n` and `x` segmentation weights are in `Models/YOLO`
  (`yoloe-26n-seg.pt`, `yoloe-26x-seg.pt`), with the `mobileclip2_b.ts` text
  encoder, which must be bound locally (`bind_local_text_encoder`).
- ByteTrack loses fast hands; mask-IoU association works.

## Environments

The environment audit ([PLAN.md](../PLAN.md)) decides the environment set and
replaces `environment.yaml`. Each environment gets a lock file and the reason it
exists here.

| Environment | Purpose |
|---|---|
| `~/.conda/envs/pointstream` | Fleet hosts' main environment |
| `~/.conda/envs/pointstream-sam31` | SAM 3.1 worker |

Packed environments for fleet staging go under
`/home/itec/emanuele/pointstream-data/environments`, which does not exist until
the first one is packed ([fleet](fleet.md#host-local-staging)).

## Archive

The code and documentation before this layout are tagged
`archive/pre-reset-2026-10-05` in this repository (`cb7d1d8`). The manuscript
at that point is Overleaf `main` `e3d52d3`, tagged
`archive/pre-reset-2026-10-05` in the server's paper clone only, because
Overleaf rejects tags.
