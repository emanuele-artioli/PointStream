# Tennis reference and mask video comparison

Current state2026-10-25: CPU phase complete;5708.866274/7200CPU-s conservative
bound,0GPU. All5jobs terminal, including2explicitly failed attempts with preserved
results. Sparse54points, affine6points, dense11complete window-points (7Al
high-rate +4both-match CRF63),131independent local checks pass. Actual generated
quality and18/24 rule remain unresolved. Three decoded reference candidates are
materialized; only reference arrays change. Await new4.5GPU-h approval below;
no active jobs or recurring monitors.

Goal: determine whether cheaper references and temporally coded silhouettes merit
an actual reference-conditioned generation test on the frozen 12-window tennis
cohort. The descriptive target remains >=18/24 strictly bracketed window–metric
wins against the same achievable anchors. CPU image/mask fidelity is a screening
diagnostic, never proof of unchanged generated quality.

Scope: user authorized the bounded comparison and subsequent quality experiment
on 2026-10-25. CPU stage has a new 7200 CPU-second cap (including subprocesses;
four threads, each job<=1800 supervised seconds); no GPU allocation in this stage. Older
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

Dense CPU job `20261025T164351Z-3d27fbbf` submitted, source`ab54b032`, specSHA256`fd8534155bac7bc26c084d9a4563698a880a9bc2936f5613155b390b87ecb8ea`.
1200s total supervisor cap,4threads,0GPU; native deadline2026-10-25T17:18:29.510627+00:00.
Record immutable native source+packet hashes, original conversion/frame identity,
delivered container RGB/alpha splits; no uncertain replay after interruptions.

## Concrete next GPU stage — proposed, not authorized

Keep separate new sub-ledgers, combined cap4.5GPU-hours; old GPU budgets closed.
First dense saved-carrier judging <=0.5GPU-h (estimate0.25): the7 completed
high-rate Al points and4 proposed CRF63 points on both original windows, all original judged frames, native composite
onto the same original Telea background, fixed original R and oracle-union S;
charge actual container/window registered duration. Original-output metric
control within1e-4, finite CUDA LPIPS/DISTS, peak<=24GiB and input/frame/decoded
hashes. Report PCHIP BD-rate only for actual curves with overlapping quality (Al
alpha4points/black32 3points if refinement completes); Med has one point per
representation, so strict bracketed cells only. This pilot cannot establish18/24 on12 windows. Three
<=600s parts (all18 points split byCRF); substantive<=60s smoke, all stages<=45min.

Then actual reference-conditioned generation <=4.0GPU-h (estimate3.1), if separately
approved as part of this combined request. Fix half-size SVT-AV1 CRF20 preset8
for all candidates: highest tested fidelity, already well inside free-matte
fixed-quality reference headroom on both matches. Compare unmasked bank, explicit
alpha segmented bank, black-key floor32 bank (threshold16), plus original PNG
control on Al031/Med014. Decode real delivered container, bicubic RGB restoration
and alpha bilinear>=.5; segmented conditioning sets outside recovered alpha to
black after restoring. No random seed/pose/crop/renderer/context/adapter change.
This produces3 candidates x2windows plus2 full original controls =8 window renders.
Native original runs took1232.62/1247.34s per window on Ada, about2.76h for8;
reserve0.34h judging/smokes/overhead in estimate,1.24h cap margin. Eight <=1450s
stages including <=60s real smoke, <=1300s full/validation/overhead, then <=600s
metric stage and remaining reserve. GPUclass/Ada existing compatible poolgpu5/6,
fleet recheck required. Save each finished unit/window, stop on failure/cap.

Original PNG control must reproduce original saved judged crops/metrics under
recorded tolerances, or investigate before candidate claims. Exact original
matte is granted and disclosed in this reference-conditioning oracle; separately
report selected paid dynamic-matte costs, with no pretending its image quality
is unchanged. Recompute actual LPIPS/DISTS for each candidate with original R/S,
same source frames and native paste/clipping/fallback. Pay actual bank7200s,
KPS and adapter/guider; common base primary, actual base separately. Candidate
pilot passes >=3/4 strictly bracketed cells (same75% descriptive target); no
window-wise recipe selection, no12-window success inferred. Rank by achieved
strict cells then delivered rate, keep Pareto tradeoffs. A full12-window actual
quality confirmation requires a further explicitly sized budget and unchanged
cohort. Failure of this pilot closes only tested recipes. Background/source
independence/granted-matte limitations persist. No fitting or Wan-Animate.

Sparse banks are preshipped in declared unit-ID order; no live/causal source
chronology is claimed. The one-point-per-recipe reference generation pilot
cannot produce BD-rate; report strict anchor-bracket cells only. Dense carrier
has3 actual coded qualities, so PCHIP BD-rate is defined only on genuine overlap.

Affine CPU guard-band refinement preregistered before outcomes: one<=200s
4CPU stage (<=800CPU-s, total prior worst-case7127.242756<7200 incl480local).
Use32+223*RGB/255 foreground with rounded inverse, threshold16, same3CRFs and
both sparse banks. Pick affine as the black candidate only if CRF20 foreground
RGB MSE improves on both matches and IoU loses<=.002 on either. Otherwise keep
clamp32. This avoids clipping dark colors; expected precodec inverse error<=1.
Generation settings/cohort/three candidate count and proposed GPU cap unchanged.

Proposed generation runtime is the original packed JCV environment
`pointstream-jcv-20261022T124459Z.tar.gz`, SHA256
`bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e`;
judging stays in pinned main environment32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661.
Use original immutable weights/fit/base archives identified in prior generation
specifications; no dependency migration or fitting is authorized.

Dense high-rate full matrix timed out with7valid complete Al points; no Med240
frame point. Targeted CRF63 refinement has4points and measured full-size
projection617.862s <660s cap; smoke55s,validator20s,overall750s. No replay.
The original1800s aggregate wall-reservation shortcut is superseded by recorded
CPU counters plus strict four-CPU bounds on unobserved tail/overhead and new
reservations; total6535.624177CPU-s including480local < unchanged7200CPU-s.
All stages remain<=45min. See experiment entry for exact auditable ledger.

dense-lowrate CPU job `20261025T170625Z-c041d6b7` submitted, source`5469e9c9`, specSHA256`7a192defcd87d25a6374f7ff75ae51482387de94f0859443515275b9dfbd7609`.
Caps750s(dense low-rate)/200s(affine),4CPU threads,0GPU; inside unchanged7200CPU-s.

affine CPU job `20261025T170628Z-2ae1c4f3` submitted, source`5469e9c9`, specSHA256`10a337b3d2c800f4d27f86351663b3f908d5f4da086fdab3cc93ef10ba8e64ac`.
Caps750s(dense low-rate)/200s(affine),4CPU threads,0GPU; inside unchanged7200CPU-s.

Dense GPU judging also reports whole-frame/outside-original-R RGB SSE and mask
FP/FN beside fixed-S neural metrics, so black-key false positives cannot hide
behind a foreground scoring region. The Matroska alpha is an explicitly named
second AV1 video track with custom client metadata; no native RGBA decoder or
standard automatic alpha-compositing support is claimed. Dense alpha is a union
of the two visible players, not separate occluded-instance reconstruction.

Selected black reference is now affine32 after its preregistered CPU dominance
check passed (both matches lower foreground RGB MSE and higher IoU than clamp).
The clamp-only prepared archive remains historical. Selected affine input
materialized at /tmp/k6b-reference-affine-variant/black-affine-q20-prepared.tar;
all6 units have exact original pose/timing/geometry/seed-array hashes.
If the CRF63 CPU stage cannot complete both windows, the proposed0.5GPU-h
dense stage may complete only missing deterministic native codec preparation
and judge it inside that same cap (no learned generation/fitting). It must
record actual encoded input hashes/bytes and validate native source conversion
against original controls; never combine an old rate with changed pixels.
Prioritize both alpha/floor32 CRF63 points in both windows, then saved high-rate
points if the remaining allowance covers them. Scope reductions and missing
curves stay explicit; no extrapolated full-cohort win. The4.0GPU-h generation
substage and combined4.5h request remain unchanged and unapproved.

CPU phase closed: low-rate native jobcomplete,4full240frame points atCRF63.
Al alpha45.7717/black32 18.8675kbps; Med71.0517/31.7100; IoU.94045/.73465
and.93961/.79818 respectively. These enter measured anchor brackets; actual
neural quality remains unknown. Preserve all11completed dense points and11
missing original matrix points (18planned-7completed=11missing in the original
matrix; the4new points are a separate refinement). No full12window claim.

Within proposed0.5h dense stage, additionally test affine32 atCRF63 on both
pilot windows (2new conventional encodes, no learned generation), then score
actual delivered decoded data/rates. This extends the CPU-proven guard-band
idea to dense carrier without increasing the4.5h request. All newGPU work is
unapproved; original0.75h/8h ledgers remain closed.

## Frozen data and claim

All cohort windows are240frames. Alcaraz–Hurkacz25fps:
`alcaraz_hurkacz/031@129` frames9–248,
`013@213`93–332,`023@228`108–347,`016@273`153–392,
`008@261`141–380,`047@123`3–242. Medvedev–Struff50fps:
`medvedev_struff/014@194`74–313,`037@349`229–468,
`002@338`218–457,`017@206`86–325,`029@138`18–257,`021@246`126–365.
Al031/Med014 above mean the first full IDs in each match; only these2pilot
windows are in the proposed new4.5h stage. The full12window18/24 claim requires
a further scaled budget if the pilot passes; the pilot cannot establish it.

Frozen manifest `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-confirm-20261024-execution/manifest-v2.json`,
SHA2568b81c4b9645a24bb5b613bdda908440735fe932d23ea251b0c567a366bda090b.
Original final diagnostic SHA256e12e296fcb505bc8d5da9a9f0d4398b99a29071da8d0c8a5b193527449a72e3f.
Read exact scored/judged frame sets and R/S from immutable original native
anchors/controls; never substitute allframes as the learned-judge sample.
Original encoder-computable court registration remains >=228/240 finite court
fits with error<=6px; it is not retuned using new CPU/GPU quality.115px remains
exploratory, source/replay independence unresolved. No new model training,
segmentation inference, sources, reference-lifetime changes or Wan-Animate.
