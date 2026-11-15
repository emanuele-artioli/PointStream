# Experiments: submission ladder confirmation intake

### 2026-11-15 — new source groups reserved; annotation completeness fails

**The installed wider VISOR release supplies three new participant groups, but
none of their mechanically selected windows is ready for confirmation.** This is
an annotation eligibility result, without source-pixel inspection, codec outputs,
model inference, quality scores or new downloads. The
[machine receipt](submission-ladder-confirmation-intake-20261115.json) records
exact paths, actual selected-video hashes, annotation-member hashes, indices,
omissions, prior exposure and commands. Status is **provisional source reservation;
confirmation not opened**. ROOT decides the final freeze.

The existing installation contains all179 original videos, plus official sparse
human polygons and dense interpolation archives for158 train/validation videos;
the21 test videos have no released masks. No better dataset or bulk acquisition
is necessary for these candidates. Dense masks, including their filtered/redrawn
keyframes, are interpolations at854×480, distinct from1920×1080 sparse human
polygons. They cannot be advertised as complete human reference masks.

## Selection and exact eligibility

Exclude the participants represented by the34 B2/G1 development sources. Rank
participants and their installed annotated videos by
`SHA256("pointstream-ladder-confirmation-v1:" + id)`. Probe metadata in that order;
reserve the first three participant groups with an installed1920×1080,50fps,
BT.709 native video. Nine probes produced the following reservations. All three
are HEVC, full-range `yuvj420p`, with BT.709 primaries/transfer/matrix and equal
nominal/average50fps. Actual MP4 SHA256 readbacks match the dataset manifest.
Participant grouping is established; independent physical kitchens are not.

Start64 consecutive native frames at the **first manual sparse keyframe**.
Centering there would request frames before the first manual anchor, where dense
alignment/provenance is unknown. Require all64 annotation records, exact
zero-drift native alignment and source-only review of human/dense contradictions.
Endpoint consistency flags do not prove visibility between human anchors.
Do not require two hands, remove known-empty frames, or replace unknowns by zeros.

| Reserved source | Native indices, inclusive | Annotation result |
|---|---|---|
| P35_105 |175–238|38 known dense records;26 missing records at native213–238.|
| P12_101 |217–280|64 exact aligned records; no contradiction at four human anchors. Dense `tap` absent on10 intermediate frames, native261–270: visibility uncertain.|
| P11_103 |229–292|64 exact aligned records; human/dense contradictions: `left hand` and `oven` at native229, `left hand` at279. Other frames require visibility review.|

One-hand frames were retained. P35 has no contradiction at its two human
anchors. P11 dense records lack `left hand` across64 and `oven` across50 frames;
these are consistency flags, not proof of visibility away from human anchors.
No alternative window was searched after these eligibility failures.

## Exposure and independence

These participants are outside the prior34 B2/G1 evaluation source registry. They are **not untouched**:
B1b's CPU label-gap survey previously covered all115 train videos, including
these three. Its selected rows and immutable receipt are retained; it did not
perform train filling or E1/F1 training.

HInt's named TRAIN/VAL annotations include all three source videos (respectively
57/17,79/22,10/2 records). No named TEST source match or named exact-window match
was found. However,120 TEST EPIC records lack a video identity, so intersection
with H2's evaluated HInt test set remains unresolved. HInt TRAIN/VAL overlap is candidate-specific source history: the native/pixel
candidate does not use the H3D generator. Generic resource/judge training
provenance remains a separate qualification gate.

## Cheapest preparation path

Reserve these source groups and windows before outcomes. Use a separately
budgeted CPU source-only preparation job to export64 native frames per source,
official polygon proposals, explicit unknown/omission flags and a blind-overlay
review subset. Review all192 source frames: establish the26 unknown masks, repair
verified same-frame contradictions, and resolve intermediate-frame visibility;
retain explicit uncertainty and human-reviewed/model-assisted provenance. No
new official download is needed. Freeze corrected scoring masks and the final
candidate before opening confirmation outcomes. Existing unreviewed interpolation
must remain a separate annotation tier.

Tennis confirmation still needs source deduplication and independent human
references; the43 exposed RacketVision clips were neither rescored nor relabeled
as held out here.

Read-only gpu6 intake used no GPU or native decoding. The authorization ceiling
was1800CPU seconds; measured Python CPU was299 seconds, excluding unmeasured
ffprobe-child/SSH overhead. Original evidence and datasets remain unchanged.
