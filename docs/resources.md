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

Two domains with real labels: **racket sports** (table tennis, tennis,
badminton: players, rackets, ball) and **egocentric hand-object video** (hands,
arms, the object in hand). All are for free scientific use. Each gets an
immutable manifest (paths, sha256, licence, label format, frame counts,
resolution, fps).

| Dataset | Domain | Labels used | Gaps filled by SAM 3.1 |
|---|---|---|---|
| **OpenTTGames** (primary sports) | Table tennis, full HD, 120 fps | Person, table and scoreboard masks (model-aided), ball coordinates, events | Ball mask from a point prompt at the labelled coordinate |
| **RacketVision** | Tennis, badminton, table tennis; 1080p; 435k frames | Ball position, racket box and 5 keypoints | Player and racket masks prompted from keypoints and boxes; ball mask from a point prompt |
| **TrackNet** (secondary) | Tennis, 720p, ~20k frames | Ball position and visibility | Players, racket, ball masks |
| **EPIC-KITCHENS VISOR** (primary egocentric) | Kitchens, 1080p video | Manual and dense interpolated masks of hands (left/right) and active objects; hand-object relations | Arms, if not covered |
| **EgoHOS** | Egocentric video, released as frames | Hands, interacting objects (first and second order), contact boundaries | Arms, if not covered |
| **HOT3D** | Egocentric (Aria), lab | Hand and object masks rendered from motion-capture 3D | Arms |

Candidate if needed: SA-Co/VEval SmartGlasses. Not used: DeepSportradar
(images, not video), ENIGMA-51 (no arms; SAM-HQ masks), DAVIS (too small for the
domains).

**Provenance tiers.** Every mask records one: `human`; `model_aided`
(OpenTTGames); `sam_from_label_prompt` (SAM 3.1 prompted by a labelled point,
box or keypoint); `sam_text` (SAM 3.1 from a text prompt only). Results are
reported per tier. No data is labelled by hand.

**Modelling decisions.**
- One model per dataset with its native labels (for example VISOR's left and
  right hand), expressed as one domain per dataset in
  `src/segmentation/domains.yaml`: shared code, per-dataset classes and models.
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
- How dense are OpenTTGames masks per clip (they exist only on labelled frames)?
- Do VISOR and EgoHOS hand masks include the forearm? If not, label-prompted
  SAM supplies arms.
- HOT3D masks are rendered through `projectaria_tools`; which streams are RGB?
- Is the demo rebuilt on VISOR/EgoHOS, or kept on a read-only archived copy of
  its current data?

**Archived.** `tennis_games`, `Egocentric-10K` and the derived
`pointstream-demo` move to an archive directory under `Datasets` with a
manifest: moved, read-only, not deleted. `src/segmentation/domains.yaml` still
lists clips from them as defaults until the per-dataset domains replace it.

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
