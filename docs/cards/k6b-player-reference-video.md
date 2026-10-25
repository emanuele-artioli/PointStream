# Tennis reference and mask video comparison

Goal: determine whether cheaper references and temporally coded silhouettes merit
an actual reference-conditioned generation test on the frozen 12-window tennis
cohort. The descriptive target remains >=18/24 strictly bracketed window–metric
wins against the same achievable anchors. CPU image/mask fidelity is a screening
diagnostic, never proof of unchanged generated quality.

Scope: user authorized the bounded comparison and subsequent quality experiment
on 2026-10-25. CPU stage has a new 7200 CPU-second cap (including subprocesses;
four threads, <=1800 supervised seconds); no GPU allocation in this stage. Older
CPU/GPU ledgers remain closed. A concrete new generation GPU budget will be
prepared from measured candidate sizes/runtime and requested before dispatch.
No fitting, segmentation inference, Wan-Animate, new chats/agents or monitors.

Compare actual native SVT-AV1/dav1d streams: sparse full reference crops, segmented
RGB plus explicit alpha as two video tracks in one Matroska container, and RGB
black-key video with foreground floors 1,8,32 and decoder thresholds ceil(floor/2).
All have identical reference indices/boxes and half-size 256x392 inputs restored
bicubically to 512x784; alpha uses bilinear resize and threshold 0.5. Three CRFs
20,35,50, preset8, fixed ordering by unit ID within each match (a bank of 12
references, not a chronological dense-video sequence or a verified player track).
Include half-size JPEG70 and PNG controls; native and quarter-size JPEG70 costs
are existing CPU sensitivities. Pay actual container/index/restore metadata.
Record RGB-only errors, mask IoU/false positives/false negatives, foreground RGB
error and actual bytes; never infer exact alpha from a lossy black guard band.

Separately test all240 masks of each first pilot window Al031 and Med014, both
players, full-frame scale8: binary/SDF AV1 CRF20/35/50 versus existing H5/RLE,
and deterministic translation-aligned XOR/LZMA (including shift bytes). Exact
reversible alignment gates on exact reduced-grid reconstruction. Lossy mask
quality is compared at matched accuracy; proxy IoU cannot decide neural quality.

Stages: smoke one match bank and one pilot mask clip at CRF35 (<=300s); validator
checks real decode/frame counts, hashes, finite diagnostics, metadata identities,
reference mask provenance, guard-band behavior and reversible alignment. Full
both banks/all three CRFs and both pilot mask clips (<=1200s), plus <=60s
validation and overhead inside1800s. Save each completed setting. Stop on failed
correctness, CPU cap, or completed pilot. No unbounded sweep or extrapolated win.

Promotion: preserve a Pareto set by delivered bytes, foreground reconstruction
error and silhouette accuracy. Use unchanged-quality headroom only as an explicit
optimistic screen; it does not dominate potential generation improvements and
cannot close all changed-conditioning routes. Select one content-independent
reference recipe and one mask recipe before opening new generated quality. A
later GPU test must use actual decoded conditioning, same seeds/renderer/windows,
an exact original-reference control, same original R/S and delivered rate, and
save every full window. Sparse reference alpha and dynamic target-frame alpha
are distinct obligations. Dense target appearance is a separate conventional
foreground-video alternative, not charged as sparse reference refreshes.

Limitations: original source/replay independence unresolved; familiar two-match
scope;115px exploratory; historical step5 failure; target-derived Telea background
and exact-matte history explicitly retained. Reference masks are the saved
encoder-side SAM player masks at each recorded reference tick, cropped with the
native reference box. No new masks or source identities are invented.

Branch: codex/k6b-reference-video. Report: docs/experiments/k6b-players.md and
planning chat01a16cfc-1a88-7013-bd79-8489fe7c1eb2. State: preregistered CPU pilot;
no new job submitted. Previous ledger's final job20261025T141239Z-93456964 is
inspected before new dispatch, never replayed.

CPU job20261025T163506Z-af342506 submitted, sourceb3d39171, deadline17:18:03UTC
on2026-10-25.1680supervised seconds at4threads inside7200CPU-s new ledger,0GPU.
Same spec and native inputs as experiment entry. One attended native bounded
wait; no monitors. Inspect existing ID after interruptions, never resubmit it.

Corrected CPU job20261025T163852Z-1d6ee39a submitted after diagnosed validator failure;
sourceb257ee03. First attempt terminal failed;59.019865CPU-s measured and
280.831CPU-s conservative admitted bound charged. New1500s supervisory cap
stays inside original7200CPU-s. Inspect the corrected existing ID before acting.

## Clarified dense foreground extension (before new dense scores)

The user's "that was my idea" also covers continuous foreground appearance.
Include it as a separate CPU carrier pilot if remaining measured CPU allowance
can cover it; do not substitute sparse reference coding for this request.
First two original windows Al031/Med014, all240 frames. Native original limited
YUV420 -> native rounded BT709 RGB, area-downsample to960x540; saved exact native
SAM union for both players. Three CRFs20/35/50 preset8, color+union-alpha tracks
versus black-key RGB floors1/32 (same fixed thresholds as sparse stage). Decode
and split real delivered containers. Upsample RGB bicubic and union alpha
bilinear>=0.5 for later native compositing. Original-background pixel values
outside the foreground are not transmitted; common Telea background remains
granted. This carries appearance every frame and does not use the generator,
references, keypoints or match adapters; pay its entire container/window duration.
It cannot claim a complete codec while the background is granted.

Smoke first8frames of each original pilot window through every declared setting,
full all240, same encoder/decoder path. Pure-black foreground guard-band control
and actual independent container decodes remain mandatory. Record source/native
conversion/frame hashes, FP/FN/IoU and foreground RGB MSE; scores are diagnostics,
not LPIPS/DISTS claims. Finish/save each setting, deadline<=45min. Reserve only
a cap that fits7200CPU-s minus both preceding attempts' conservative admitted
bounds and480local allowance. If unavailable, preserve this concrete stage as
pending; no unbounded CPU extension. New GPU quality scoring/generation remains
budget-gated. This stage is a separate representation, not a silent renderer
change or a new independent-confirmation claim.
