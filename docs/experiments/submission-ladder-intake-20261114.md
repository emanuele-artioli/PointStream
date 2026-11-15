# Experiments: submission ladder footage intake

### 2026-11-14 — existing datasets retained; concrete source joins recovered

The existing datasets remain useful. Their limitation is the combination of
annotations, rather than poor footage: VISOR provides aligned hand/object masks
but its dense masks are interpolated and can omit objects; TrackNet provides
dense ball position/visibility but only JPEG frames; RacketVision provides native
video and sparse independent ball/racket labels but no dense player/racket masks.
A new dataset acquisition is not needed to start development. No new external
search is claimed in this intake. Strong confirmation still needs explicit
source-group/exposure checks and source-only human mask review where pixel truth
is absent.

The [machine receipt](submission-ladder-intake-20261114.json) binds source paths,
actual member hashes, known source hashes, selections and unresolved gates.
Read-only metadata was inspected on gpu2 under the user's 12-hour autonomous
budget authority (expires November15 11:07:34UTC); no source archive was modified,
no GPU/inference/native encoding was performed, and no actual source preparation
job is asserted by this receipt. ROOT expanded the metadata ceiling to3600CPU
seconds; complete administrative CPU accounting is not measured. Source manifests
and selected members were hashed where stated, not whole video collections.

## Development inputs ready for CPU preparation

- **Tennis:** familiar `alcaraz_hurkacz/031@129`, first64 frames of its existing
  240-frame window, corresponding to source clip indices9–72 at25fps. Exact source
  video is in `tennis/k6b-confirm-20261024-execution/sources/`; its accepted SHA is
  `054e4535…cef76`. Actual source-region transport is `exact-rle.matte` from accepted
  job `20261025T082944Z-29f021a5`, newly hashed `5aa4c788…9a7ea5` (57189B). Union
  its source-model person regions and upsample nearest to1920×1080. This is
  fixed-source SAM foreground, not independent human truth. Reuse tests a new
  mechanism on known development footage without suggesting held-out evidence.
- **Egocentric:** filter VISOR34 to the verified50fps source class (avoiding
  untagged EK55 colour assumptions), rank `SHA256("pointstream-ladder-dev-v1:" +
  item_id)`, select first2. These are P09_103 and P07_110, both already exposed
  through B2/G1. Choose64 frames centered on the first human keyframe, clamped
  inside the240-frame window. The primary P09_103 selection is native
  indices17224–17287, mask-window indices101–164, with one sparse human frame
  at selected tick32. Its source SHA is `c94c6cf8…bb4338`. Precomputed dense masks
  are in the26.6MB B1 mask archive, newly hashed `0cf83b8c…e1937b5`; exact RLE
  member `publish/masks/P09_103_0000017124/masks.rle` newly hashes
  `7bdd07f0…8efac1`. P07_110 is a reserve development selection, not a result.

The new `experiments.codec_ladder_prepare` module accepts the video and mask
identities, selected indices and frame count. It writes raw native YUV420,
boolean all-frame foreground NPZ and a preparation manifest into a new
`published.tar`. Tennis uses the original K4 limited-range source-conversion
command; VISOR preserves native full-range BT.709 planes. It checks source and
mask hashes, native geometry/cadence, VISOR exact frame alignment, label flags
and zero drift, all-frame sizes and publication readback. It does not label
objects or score codec quality.

Run only through ROOT's separately frozen CPU fleet job. A16-frame smoke must
forecast64-frame preparation plus publication with1.5 margin and30s overhead
under600s before promotion. The full64 stage must itself finish inside600s.
`--validate` reconstructs this cost rule and physically checks published payload
hashes, byte counts and full boolean-mask coverage. Six focused tests cover
alignment, empty-frame retention, nearest union reconstruction and a corrupted
cost forecast. This implementation is not source-denied codec qualification.

## RacketVision native confirmation candidates

An actual native-video/label join was recovered directly inside the existing
`RacketVision/tennis.tar` (manifest SHA `9b1ea8c4…d6a3820`). Its43-entry test split
newly hashes `120612c7…7678781`. A content-blind rank of split-pair identifiers
selected `match11_000`, `match126_000`, `match12_000` before any video pixels or
codec outcomes were examined. Each source MP4 and corresponding50-row native
ball CSV was actually read, hashed, and matched to the original per-member
checksum sidecar. Each has25fps1920×1080 BT.709 limited-range native video;
frame counts are249,240,249. Racket box/keypoint annotations occur on45,16,4
frames respectively. Sparse missing rows mean unknown, never absent.

These3 are absent from the eight tennis clips selected by the previous G1
background audit (`match134,132,102,106,119,123,110,136`). That eliminates a
specific recorded exposure route, not all research/model training exposure.
Original broadcast identity, duplicate-match overlap with TrackNet/other
broadcasts, label overlay/frame alignment, and judging-model training exclusions
remain gates before calling this independent confirmation. These clips are
candidates reserved from codec outcome inspection; no confirmation cohort or
success claim is frozen here. Human-reviewed SAM proposals can supply a small
source-only pixel-reference set, with model-assisted provenance and separate
blind annotations, without discarding this useful native footage.

### 2026-11-14 — VISOR fleet data-root source repaired by independent copy

ROOT's initial VISOR publication rejected the original dataset video path because
fleet data-file inputs must live inside the external `pointstream-data` root.
The historical G1 source path already existed there and its SHA matched, but it
was a0644 same-inode hardlink to the original. It was preserved unchanged.

Under ROOT's explicit1800CPU/GPU0 storage-preparation allowance, the source was
copied first to gpu2's separate local SSD, hash-verified and read back, then
independently copied into a fresh data-root intake directory and read back.
The eligible immutable source is now
`pointstream-data/ladder-autonomy-20261114/intake/p09-103-20261114T233342Z-ed1fc777/P09_103.MP4`.
It is2,252,549,936bytes, SHA256
`c94c6cf88542ef08057224f5834b911737fb5e25515af3a705bd8b92a7bb4338`,
regular/nonsymlink, independent inode,0444 inside0555 directory. Adjacent
`copy-receipt.json` and the retained local SSD copy preserve provenance. Original
mode0644 and source bytes were unchanged. Copy wall44.1202s and scoped CPU29.374161s
are not complete remote administrative CPU accounting. No encoding, inference,
GPU work or source preparation job occurred in this copy. The readback receipt
permits ROOT to submit its separately identified corrected preparation request;
it does not relabel the unpublished original request.

### 2026-11-15 — bounded external search for confirmation alternatives

A new primary-source web search was performed after discovering all43RacketVision test rallies had influenced tracker selection. Search terms included tennis native-video player segmentation masks, annotated tennis video datasets and VISOR annotation provenance. No inspected release is established as a ready replacement providing broadcast-native video, dense independent player/racket masks, dense ball labels and verified unexposed source groups together. This is a bounded negative search, not proof no such dataset exists. Existing footage remains suitable for development; the missing parts are independent confirmation and reference quality.

The [official VISOR page](https://epic-kitchens.github.io/VISOR/) explicitly distinguishes manual masks and dense interpolations. Retain VISOR; do not call every interpolated frame human-annotated. Choose new source-video groups from its wider release before outcomes, preserve interpolation provenance, and audit source frames against manual keyframes.

For broadcast confirmation, the [official precise-spotting tennis release](https://github.com/jhong93/spot/tree/main/data/tennis) supplies original video identifiers and specifies1080p,25/30fps. This improves source identification over anonymous RacketVision matchIDs, but its event labels are not player/racket segmentation truth. Original-video access, actual decoded format, rights and source/training deduplication remain unverified here. Its [classification derivative](https://huggingface.co/datasets/OpenSportsLab/OSL-cls-tennis-public) states256×144 lossy recoding and very short event clips; do not adopt that derivative as native1080p codec footage or mistake action labels for object masks.

[CalTennis's paper](https://arxiv.org/abs/2606.20542) describes synchronized60Hz tennis practice/match recordings from40players; its [dataset card](https://huggingface.co/datasets/S34085068/caltennis) offers a mini subset. This is a plausible independent acquisition condition, not an authenticated new cohort. Its documented task is pose estimation, not the required independent ball/player/racket masks. Availability, mini-file integrity, actual resolution/colour/cadence, session/player grouping, usage conditions and annotation effort need verification. Practice-camera footage also changes the broadcast target condition. Do not download120GB or qualify another model merely to expand frame count.

The [TenniSet repository](https://github.com/HaydenFaulkner/Tennis/blob/master/data/README.md) documents originalMP4 videos and event/commentary annotations, offering another source-identifiable candidate rather than dense object truth. The [ACASVA project](https://kahlan.eps.surrey.ac.uk/acasva/Downloads.html) explicitly cannot provide its original videos, so its feature release cannot support this codec experiment.

Decision: use the already authenticated development inputs and source-only review pilot now. For confirmation, prefer a small source-identified broadcast cohort with human-reviewed model-assisted masks; CalTennis is an alternate condition if broadcast access/deduplication fails. Freeze source groups and mechanical windows before codec outcomes. No newly found release was downloaded, licensed, scored or claimed as confirmed by this search.
