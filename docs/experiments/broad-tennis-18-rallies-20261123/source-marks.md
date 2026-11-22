# Independent source marks on the held-out points

Freeze eight scoring instants per point before opening any reconstructed frame. Divide each original point into eight equally spaced time bins and select the native tick at floor((2j+1)N/16), j = 0,…,7. Retain every tick, including broadcasts that cut to a close-up. If an entire conventionally routed shot has no selected tick, add its native midpoint as a separately flagged diagnostic. Keep this extra sample outside the equal-bin aggregate. Enroll the same original frames for every codec and budget. Existing original start/mid/end inspections informed source authentication only.

The active private review opens on the ball. A single click marks the native centre, with distinct absent and unsure choices. The reviewer may leave every player entry empty. Original-source person predictions, when available, are optional overlays with anonymous accept, reject and unsure choices. Optional rough rectangles define appearance crops only. Accepting an automatic outline records verification of that proposal. Neither rough rectangles nor proposal approvals establish independent per-pixel silhouette ground truth for an IoU claim. Source-only ball marks, absent labels and unsure statuses can support same-tick ball fidelity and detector analyses. PointStream’s own predicted ball positions are available in a separate optional view; a decision made after showing one carries an assistance flag and is excluded from independent localization accuracy. Opening a player proposal does not show the ball prediction.

The ball-first editor has separate storage and a new export schema. It records the private source frame hash and original packet fingerprint with each decision. Its landing page is `tenniset-ball-first-review-20261123/index.html`. The original per-point editors, their browser saves and exports remain available through the original index. Earlier polygon exports document source review only and must not be described as independent validated IoU ground truth.

The packet contains 48 source PNGs across six independent, offline point pages. The fixed grid sampled 40 court frames and eight conventionally routed frames. Every routed shot is represented, so no diagnostic top-up frame was needed. [Frozen source identities and route-stratum counts](source-marks.json)

Within the original polygon-format packet, open `index.html`. Each point has a separate export. Import that export with `python tools/datasets/al028_review_import.py --review POINT_PAGE_DIRECTORY --decisions EXPORTED_JSON --out NEW_OUTPUT_DIRECTORY`. The normalized record contains original broadcast and extract SHA256s, local and absolute native ticks, source RGB hashes, named-athlete pixel masks and ball coordinates on the 1280×720 original grid. Visible body marks require at least one nondegenerate polygon. A visible ball requires an in-bounds coordinate; absence and uncertainty preserve a null coordinate. Score compared displays at these exact source instants. Give missing/uncertain counts next to each endpoint, and present ball false positives on explicitly absent frames separately.

The original private master page is `tenniset-source-marks-20261123/index.html` in the project’s author visualization directory. Six per-point contact sheets sit beside their editors. A synthetic offline browser round trip exercised polygon drawing, native ball marking, absence, uncertainty, export and import. These synthetic decisions were discarded; no reviewer decisions were solicited or imported into the paper.

The ball-first packet now carries **90 real person-mask proposals across all 48 frozen source images**. The same selected source-frame manifest is retained. The 24 available sender-generated ball candidates are explicitly optional. Offline Chromium opened all 48 images and validated the ball-only path, the provenance flags, per-person proposal controls, visible coverage of the mask interior, and export/restore. No author labels were created by the browser tests. [Interface and provenance receipt](source-mark-interface.json)

## Returned London review

The author export contains all 48 frozen frame identities and provenance for
the displayed proposal set. It accepts 88 person proposals and rejects two.
There are nine visible ball clicks and 15 absent marks made without the sender
overlay. Another 24 visible marks used the overlay. Keep the latter separate
from independent ball quality and localization. Use checked source masks for
regional player appearance only. Keep player and ball fidelity and PSNR on
the remaining source pixels separate. [Replay and scoring script](score_london_review.py)

The CPU replay authenticated and scored all 48 source frames at each of the two
ceilings. The stream emitted no position on any of the nine independently
clicked visible frames. Five were court views and four conventionally routed.
It emitted no positions on the 15 independently marked absent frames. The
saved tracker suggestion was available on the 24 assisted frames; 23 of those
clicked centers lay within ten native pixels of the delivered positions. This
last count measures agreement after viewing the suggestion. The independent
subset is associated with suggestion availability, so 0/9 must not be
extrapolated as London recall. The five independently marked court misses
show a sender coverage problem in this sample at either rate.

On the nine independent visible frames, adding the trajectory produced the
same ball-crop DISTS, LPIPS and PSNR as its own base, consistent with the
absent payload on those instants. Only one independent visible frame at
20.4 kbps and two at 33 kbps fall in the two tightly rate-matched controls.
Keep those comparisons as diagnostics. The ball-only deliveries support
comparison of ordinary player-region fidelity and background costs.
They do not exercise the learned player receiver.
[Per-frame results and hashes](london-reviewed-scores.json)
