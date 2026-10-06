# Plan

Each workstream is scoped to one session. Dependencies: A → B → (C, D, F);
E after A, in parallel with C and D; G and H after F; I throughout. The
environment audit (step 0) comes before the first component that needs a new
dependency.

## 0. Environment audit, first wave

Audit SAM 3.1, YOLOE/Ultralytics, the dataset tools (for example the VISOR and
HOT3D loaders), SVT-AV1 and the chosen neural codec, with their pinned
dependency versions. The audit decides the environment set and replaces
`environment.yaml` ([AGENTS.md](AGENTS.md#environments)). Each later component
starts with its own audit increment.

## A. Data acquisition and archive

Download OpenTTGames, RacketVision, TrackNet, VISOR, EgoHOS and HOT3D under
`Datasets`, with immutable manifests ([resources](docs/resources.md#datasets)).
Archive `tennis_games`, `Egocentric-10K` and `pointstream-demo` (move,
read-only, no deletion, with a manifest).
**Done when** every dataset has a manifest and a smoke read of one clip.

## B. Ground-truth adapters and label-based evaluation

In `src/segmentation`: a reader per dataset producing `ClipMasks` with native
classes, per-frame labelled flags, provenance tiers and ball point labels; one
domain per dataset in `domains.yaml`. Extend `evaluate.py` with point metrics
for the ball and per-tier reporting; `bench --reference` accepts a ground-truth
set.
**Done when** each dataset converts and a trivial candidate scores against it.

## C. SAM 3.1 from label prompts (pseudo-ground truth)

Ball masks from point prompts; players and rackets from RacketVision keypoints
and boxes; arms where egocentric labels stop at the hand. Validate on
OpenTTGames, which has both masks and ball coordinates.
**Done when** the gap between prompted SAM and the labels is measured and the
pseudo-labels are written with provenance.

## D. Segmentation benchmark on labels (paper table)

SAM 3.1 (text and prompted), YOLOE-26 n–x (obtain the s, m and l weights), ball
tracker, hand-object proposer: accuracy against speed per dataset and per class,
on the fleet.
**Done when** the table is reproducible from a recorded job.

## E. Proposers

A ball tracker trained or fine-tuned on RacketVision, cross-tested on
OpenTTGames and TrackNet; a hand-object proposer from VISOR/EgoHOS, cross-tested
on HOT3D; the vision-language auto-prompt fallback.
**Done when** cross-dataset numbers decide which to commit to.

## F. Training-data export

Per dataset: foreground crops and masks per instance, and background frames with
the foreground removed, written by `python -m src.segmentation dataset` with
provenance. Feeds G and H.

## G. Background encoding

Design session ([components](docs/components.md#3-background)).

## H. Foreground encoding

Players, handled objects and the ball (a parametric ball trajectory is a
candidate) ([components](docs/components.md#4-foreground)).

## I. Paper

Rescope to racket sports and egocentric hand-object video, with a fresh paper
repository state: dataset and evaluation-protocol sections (weighted PSNR on
dataset masks, provenance tiers). Set the venue and submission date.
