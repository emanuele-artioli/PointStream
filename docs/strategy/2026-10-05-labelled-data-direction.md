# PointStream direction after the labelled-data decision (5 October 2026)

Status: agreed direction from the segmentation session on branch
`claude/pointstream-segmentation-module-3e59bf`. It replaces the unlabelled
evaluation basis in `PLAN.md`; the thesis in
[semantic-codec-thesis.md](semantic-codec-thesis.md) is unchanged. Each
workstream below is scoped to be one session.

## 1. The paper in one paragraph

Conventional codecs spend bits on pixels without knowing what they encode.
PointStream splits a video into **foreground** (what the viewer watches: people,
the objects they handle, the ball) and **background** (encoded only well enough
not to distract), and encodes each with the best tool for it. The paper targets
two domains with real labels: **racket sports** (table tennis, tennis,
badminton: players, rackets, ball) and **egocentric hand-object video** (hands,
arms, the object in hand). Non-racket sports are future work. Evaluation uses
human or model-aided labels wherever a dataset has them; SAM 3.1 fills only the
gaps, and every mask records which of those it came from.

## 2. What changed, and why

| Before | After | Reason |
|---|---|---|
| Unlabelled clips (`tennis_games`, Egocentric-10K); SAM 3.1 treated as ground truth | Labelled datasets (below); SAM 3.1 only where labels are missing, prompted *from* the labels where possible | Real truth for evaluation, and for the training targets |
| Foreground = player + racket / hand + arm by text prompt | Foreground includes the **ball** and the **handled object** | The viewer watches them; both are hard to segment without labels |
| Foreground masks for scoring came from our own segmenter | Weighted PSNR (0.7 FG + 0.3 BG) uses dataset masks | Removes the codec grading itself |
| "Sports" broadly | Racket sports | Every adopted sports dataset is a racket sport |
| No hand labelling ruled in or out | **No hand labelling** | Use what exists; SAM 3.1 for the rest |

Labels matter for training too, not only for evaluation: foreground encoders
train on segmented objects, and background encoders need the foreground removed
cleanly to see the background underneath. The handled object is the hardest part
to obtain without labels, which is why these datasets were chosen.

## 3. Datasets

Adopted (download all; storage on the GPU servers is not a constraint; free
scientific use):

| Dataset | Domain | Labels we use | Gaps filled by SAM 3.1 |
|---|---|---|---|
| **OpenTTGames** (primary sports) | Table tennis, full HD, 120 fps | Person/table/scoreboard masks (model-aided), ball coordinates, events | Ball mask from a point prompt at the labelled coordinate |
| **RacketVision** | Tennis, badminton, table tennis; 1080p; 435k frames | Ball position, racket box + 5 keypoints | Player and racket masks prompted from the keypoints/boxes; ball mask from a point prompt |
| **TrackNet** (secondary) | Tennis, 720p, ~20k frames | Ball position + visibility | Players, racket, ball masks |
| **EPIC-KITCHENS VISOR** (primary egocentric) | Kitchens, 1080p video | Manual + dense interpolated masks of hands (left/right) and active objects; hand-object relations | Arms, if not covered |
| **EgoHOS** | Egocentric video (released as frames) | Hands, interacting objects (1st/2nd order), contact boundaries | Arms, if not covered |
| **HOT3D** | Egocentric (Aria), lab | Hand and object masks rendered from motion-capture 3D | Arms |

Later, if needed: SA-Co/VEval SmartGlasses. Not used: DeepSportradar (images,
not video), ENIGMA-51 (no arms; SAM-HQ masks), DAVIS (too small for the
domains).

Every mask carries a provenance tier: `human`, `model_aided` (OpenTTGames),
`sam_from_label_prompt` (SAM 3.1 prompted by a labelled point/box/keypoint),
`sam_text` (SAM 3.1 text prompt only). Results are reported per tier.

Archive: `Datasets/tennis_games` and `Datasets/Egocentric-10K` (plus the
derived `Datasets/pointstream-demo`) are archived, not deleted, until the
labelled datasets show where we stand. The demo currently depends on them.

## 4. Decisions on modelling

- **One model per dataset, native labels.** Keep each dataset's own taxonomy
  (e.g. VISOR left/right hand) instead of collapsing to shared classes. The
  segmentation module expresses this as one domain per dataset in
  `src/segmentation/domains.yaml`. Shared code, per-dataset classes and models.
- **Ball from positions.** Point SAM 3.1 at the labelled position to get a mask
  that captures the ball's actual appearance (blur, colour). Fall back to a
  rendered disk/streak only where SAM fails; measure how often.
- **Ball tracker.** Train or fine-tune a TrackNet-style tracker on RacketVision
  and test it on OpenTTGames and TrackNet before relying on it; overfitting to
  one dataset is the risk to check.
- **Handled object.** Hand-object interaction models (VISOR- or EgoHOS-trained)
  propose the object; their boxes prompt SAM 3.1. A vision-language model that
  names the object for a SAM text prompt is the fallback.
- **Cross-dataset checks before committing** to any trained proposer.

## 5. Where we stand (evidence, 5 October 2026)

- `src/segmentation` replaces the scattered segmentation code (SAM 3.1 reference
  worker with pinned checkout, YOLOE-26 backends, lossless `ClipMasks`,
  benchmark/suite/dataset CLI, windowed SAM for long clips, provenance per run).
  The runner, demo and SAM 3.1 audit script use it; legacy code is deleted.
- Fleet benchmark on the unlabelled clips (job `20261005T154444Z-b8fcc04e`):
  agreement with SAM 3.1 and throughput only; not citable and now superseded as
  evaluation by the labelled datasets. See §8 for its numbers.
- The `01_segmentation` scorecard verdict (synthetic ellipses) is withdrawn.
- Earlier codec campaigns: no claimable weighted-PSNR point (see `PLAN.md`).
- YOLOE-26 s/m/l weights are not on the servers (only n and x).
- The 30 September 2026 TOMM date in `AGENTS.md` has passed; a new target is
  needed.

## 6. Workstreams (one session each)

Dependencies: A → B → (C, D, F); E after A, in parallel with C/D; G and H
after F; I throughout.

**A. Data acquisition and archive.** Download OpenTTGames, RacketVision,
TrackNet, VISOR, EgoHOS and HOT3D to the fleet datasets root; write immutable
manifests (paths, sha256, licence, label format, frame counts, resolution, fps)
under `pointstream-data/manifests/`. Archive `tennis_games`, Egocentric-10K
and `pointstream-demo` to an archive location with a manifest (move, read-only,
no deletion). Done when every dataset has a manifest and a smoke read of one
clip.

**B. Ground-truth adapters and label-based evaluation** (`src/segmentation`).
Per-dataset reader → `ClipMasks` with native classes, per-frame "labelled"
flags (sparse labels are scored only where they exist), provenance tiers, and
point labels for the ball. One domain per dataset in `domains.yaml`. Extend
`evaluate.py` with point metrics for the ball and per-tier reporting;
`bench --reference` accepts a ground-truth set. Done when each dataset converts
and a trivial candidate scores against it.

**C. SAM 3.1 from label prompts (pseudo-ground truth).** Ball masks from point
prompts; players/rackets from RacketVision keypoints and boxes; arms where
egocentric labels stop at the hand. Validate on OpenTTGames, which has both
masks and ball coordinates: how close is prompted SAM to the labels? Done when
the gap is measured and the pseudo-labels are written with provenance.

**D. Segmentation benchmark on labels (paper table).** SAM 3.1 (text and
prompted), YOLOE-26 n–x (obtain s/m/l weights), ball tracker, HOI proposer;
accuracy vs speed per dataset and per class, on the fleet. Done when the table
is reproducible from a recorded job.

**E. Proposers.** Ball tracker trained/fine-tuned on RacketVision, cross-tested
on OpenTTGames and TrackNet; HOI proposer from VISOR/EgoHOS, cross-tested on
HOT3D; VLM auto-prompt fallback. Done when cross-dataset numbers decide which
to commit to.

**F. Training data export.** Per dataset: foreground crops + masks per
instance, and background frames with the foreground removed, written by
`python -m src.segmentation dataset` with provenance. Feeds G and H.

**G. Background encoding.** Next session per the original plan.

**H. Foreground encoding.** Players, handled objects, ball (parametric ball
trajectory is a candidate).

**I. Paper.** Rescope to racket sports + egocentric hand-object video; update
`AGENTS.md` target and `PLAN.md`; dataset and evaluation-protocol sections
(weighted PSNR on dataset masks, provenance tiers).

## 7. Open questions

- OpenTTGames masks exist only on labelled frames: how dense are they per clip?
- Do VISOR/EgoHOS hand masks include the forearm? If not, C supplies arms.
- HOT3D masks require rendering through `projectaria_tools`; which streams are
  RGB?
- The demo is built on Egocentric-10K: rebuild it on VISOR/EgoHOS, or keep a
  read-only archived copy for the demo only?
- New submission target and venue.

## 8. Last benchmark on the unlabelled clips (superseded, for reference)

Filled in from job `20261005T154444Z-b8fcc04e` (4 clips per domain, ≤300
frames, agreement with SAM 3.1 text prompts):

Tennis (804 frames; SAM 3.1 reference at 348–834 ms/frame, 22–37 GiB peak,
24–100 s model load):

| Backend | J | F | recall | precision | flicker | ms/frame |
|---|---|---|---|---|---|---|
| YOLOE-26n | 0.31 | 0.42 | 0.31 | 0.79 | 0.50 | 75 |
| YOLOE-26x | 0.22 | 0.41 | 0.22 | 0.85 | 0.69 | 126 |

YOLOE covers roughly a quarter to a third of the pixels SAM 3.1 labels, with
high precision; per clip it varies from J 0.02 to 0.69 (YOLOE-26x nearly misses
the players on `djokovic_zverev_004`, the slowest and largest clip). Not
inspected visually, because this basis is superseded; do not cite.

Egocentric: in the 16-frame smoke, YOLOE with "arm"/"hand" prompts produced no
masks and SAM 3.1 found hands but no arms. Full run (1,200 frames; SAM 3.1 at
345–364 ms/frame, ~28 GiB peak; no backend failures):

| Backend | J | F | recall | precision | flicker | ms/frame |
|---|---|---|---|---|---|---|
| YOLOE-26n | 0.15 | 0.18 | 0.17 | 0.28 | 0.20 | 35 |
| YOLOE-26x | 0.18 | 0.23 | 0.22 | 0.31 | 0.17 | 90 |
