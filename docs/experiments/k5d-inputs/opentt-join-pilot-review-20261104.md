# OpenTTGames join witness: CPU pilot for review

This is a proposed CPU-only test of video/label joining, not a source-judge run.
No pixels, model or GPU were used to select either the source contexts or the
three-frame witnesses. The separately frozen selection and rule hashes are in
the witness entrypoint. Full benchmark labels, including partial-coordinate
unknowns, are not altered by witness selection.

One video, `test_2`, is the bounded first pilot: the original H.264 file is
225,118,372 bytes, SHA256
`91df783ab4d26bf350faffaefabe9fb6c172584dfd053a866729b11cc3261c43`;
the annotation ZIP is 897,976 bytes, SHA256
`c692014ca9074d331700385e4c0b5e0bd32968123bf001fe8b1c9b2e64ac142d`.
The stored 120-fps video has 3,600 reported frames and the frozen selected
witness is 1085–1087. The packed environment archive is 4,936,811,281 bytes,
SHA256 `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`.
Verified packed tools are FFmpeg SHA256
`7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2`
and ffprobe SHA256
`f4e72e0094397518602881be95f16a1d60ec41b03ecfb0aaa6f8e45ffa14583a`.

The seven videos total 5,897,983,838 bytes, with up to 36,000 frames per
video. There is no measured full-stream probe, sequential decode, cache, copy,
publication or end-to-end CPU cost for this witness. Their sum cannot yet be
forecast inside 300 CPU-seconds. Therefore the draft specifies only the single
pilot, not automatic continuation to the other six. The one-core campaign wall
budget is 270 seconds, leaving 30 seconds of the 300-second allocation for
termination/accounting margin; the fleet budget also includes staging. Its
210-second smoke limit and 25-second validator are ceilings, not estimates or
promises. The full stage is a five-second no-work marker. Failed scratch and
partial artifacts must be retained under the fleet's failure policy. Determine
remaining budget and full seven-video feasibility from actual process-tree CPU,
staging, local-cache state, video sizes, frame counts, and observed complete
pilot time before admitting any other video. A single-video measurement does
not by itself prove a bound on the longest stream.

The draft expects new, non-overwriting data-root hard links at
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k5d-opentt-join-witness-20261104/inputs/test_2.mp4`
and `.../inputs/test_2.zip`. These links have **not** been created or verified.
Fleet admission requires regular files under its data root and will rehash the
staged bytes; establish and verify matching source/destination device and inode
before submission. The spec's deadline is provisional and must be set by root.
Verify all eligible hosts, current jobs, local storage, and the budget before
submitting. This document and the draft do not request or constitute a launch.

The worker requires exact 1920x1080, nominal and average 120/1 fps, exact 120-Hz
integer PTS increments from the first decoded presentation frame, and the
registered total frame count. It obtains three source RGB frames by sequential
decoding from the beginning and by integer-second fast seek, comparing both
raw hashes and exact original PTS. It publishes original PNGs, a labeled
contact sheet, the exact commands, source hashes and PTS records. Any mismatch
stops. `REVIEW_REQUIRED` remains the only successful machine status: image and
label overlays require independent human review, and this result alone does not
prove the author's label origin. No offset is selected by a model or score.
