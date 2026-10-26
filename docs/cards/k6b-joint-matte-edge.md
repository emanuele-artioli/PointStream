# Joint compressed-reference/matte pilot and bounded edge mechanism check

Current state (2026-10-26): COMPLETE; stop because both decisions are reached.
Each actual paid1/8,1/16,1/32 dynamic-matte recipe scores0/4 strict metric cells;
exact/granted-matte control reproduces4/4 and all saved composite hashes within
unchanged1e-4 metric tolerance. No unresolved primary cells;0/4 remains under
±0.001 sensitivity. No new generation/fitting/segmenter inference/Wan-Animate.
GPU cost373.340391/1800s =0.103705664/0.5h, all attempts included.
CPU preprocessing conservative721.524817/3600CPU-s, including failed smoke and
local allowance. All current native jobs are terminal; no recurring monitor.

Small RGB ramps save8–17% on the short excerpts but damage shape even before
compression. Accurate separate-luma ramps are slightly larger. No secondary
alternative passes the frozen engineering promotion bar; no secondary neural
quality test is claimed. Written support/crop gaps and8-frame excerpt limits
are explicit. All source/mask/frame/packet/score/native client checks pass.

This rejects these tested recipes under the two-window pilot rule. Ten other
cohort windows remain untested for this combination; target-derived background,
source independence,115px and historical step5 limitations remain. Full
self-contained outcomes are the final2026-10-26 entries in the experiment file.
The protocol and operational amendments below preserve the execution history.

Goal: test the new saved full-RGB-reference generations with actual compressed
mattes, then determine whether small boundary ramps justify a later codec study.
User's explicit “Perform that test” is recorded in planning chat
01a16cfc-1a88-7013-bd79-8489fe7c1eb2 on 2026-10-26. This is an execution card.

Branch: `codex/k6b-joint-matte-edge-pilot`, from current main after PR242.
Report: `docs/experiments/k6b-players.md`, new 2026-10-26 entry; notify that
planning chat. No other chat or parallel agent. Prior budgets remain closed.

## Primary frozen experiment

Use only PR242 FULL unmasked-reference outputs for Al031@129 / Med014@194,
both original players, all240 context/coded frames and original80 judged frames
per window. Saved crops contain only those judged frames; never invent RGB for
unretained times. Reuse original R/S, crop boxes/order and Telea background.
Pair these outputs with existing decoded dynamic matte packets at1/8,1/16,1/32.
Each is one global recipe; no per-window choice after scores. All packet bits,
actual keypoints, compressed reference banks and adapter/guider are paid with
PR242's7200-second horizon. Common-base and base-charged cases are separate.

For each scale, report all4 LPIPS/DISTS cells against the original achievable
SVT-AV1/DCVC envelope, with no extrapolation; pass at>=3/4 strict wins. Exact
matte control must reproduce PR242 FULL composite RGB hashes and saved metric
scores within1e-4, on every original judged frame. The prior fresh-render±1e-3
sensitivity remains a caveat/sensitivity table, never a new control tolerance.
Report exact-matte-paid alongside granted-matte control. No full12 or
independence inference. All sources, masks, packets and crops must hash-verify.

Crop coverage is explicit: expand the decoded matte, paste only available saved
generated RGB inside the unchanged crop box, keep native Telea fill elsewhere.
Report requested matte pixels outside available crop RGB and resulting missing
pixels. Never replace missing foreground with target RGB. A missing crop/time
fails the completeness gate, rather than silently dropping a scored frame.

GPU cap: <=1800 admitted seconds (0.5h), including smoke/setup/failures/recovery.
First reservation:60-second representative smoke,600-second full,
20-second validator and70-second overhead =750 seconds. Prior real five-variant
joint judging completed at332.188 admitted seconds for the same160 frames;
this run has four variants. Smoke rechecks original controls, native CUDA/class,
coverage, hashes, finite metrics and memory. Compatible qualified Ada pool,
12000MiB, exclusive/stop-on-contention; inspect jobs/all hosts first. All stages
<=45min; native fleet waits only, no recurring model monitors. Reserve remaining
1050 seconds only for concrete recovery after inspecting actual failed jobs.

## Secondary CPU-only mechanism

Before neural scores, freeze SVT-AV1 preset4/CRF35, yuv420p, half-size960x540,
and widths0 (hard),2,4 full-resolution pixels. Use both windows' first8 saved
judged ticks, preserving their actual rational sampling interval in the video;
report excerpt bytes/rate, never project a full240-frame rate from this GOP.
Smoke uses Al's first4 saved judged ticks; full both first8 only if CPU smoke
projection fits. No rate/width/threshold sweep.

Two separately labeled representations:
1. RGB key carrier: guard-lift generated foreground RGB into[32,255], black key
   outside; for ramps, mix nearest foreground-generated RGB with black using a
   centered signed-distance alpha ramp of width2/4. The client sees only decoded
   RGB and paid constant metadata, uses the unchanged max-channel>16 key test,
   then the guard inverse. No oracle alpha or source mask in its decoder.
2. Mask-only luma control: encode binary/signed-distance-ramp alpha in Y with
   neutral chroma, threshold decoded Y at128 and upsample. This is a paid
   separate stream; it does not carry foreground RGB and cannot be billed as
   a free channel in representation1.

Measure container bytes, IoU, FP/FN, symmetric boundary mean/p95 displacement,
frame-to-frame boundary-error instability, boundary RGB error/contamination.
Zero-codec hard-edge mask recovery and guard RGB error<=1 must pass at coded
resolution. Report inherent ramp/downsample losses separately. Anti-leakage:
client functions take only decoded channels/metadata; changing a reference mask
used for diagnostics must not change client output. Demonstrate the ambiguity
alpha*C=(alpha/2)*(2C) even without compression: RGB alone generally cannot
identify alpha and foreground color. No promise of exact deblurring/deconvolution.

Exploratory engineering promotion bar, not a paper claim: one fixed width must
save>=10% container bytes on BOTH excerpts versus the corresponding hard-edge
family, while IoU drops<=0.005, mean boundary displacement rises<=0.25 full
pixels and boundary-error instability rises<=0.005; all decoder gates must pass.
No secondary LPIPS/DISTS promotion in this task. Keep its findings separate
from the primary actual-quality decision. Binary shapes are not inherently
incompressible; prior H5 XOR/LZMA was already competitive, and smoother signals
may trade rate for boundary displacement or color contamination.

CPU cap: <=3600 aggregate CPU-seconds, <=2h wall. Reserve <=800 admitted
seconds at4 CPU threads (=3200CPU-s) plus400 local preparation/analysis.
Smoke<=60s, full<=650s, validators<=30s, remaining overhead<=60s; use actual
smoke timing and output checks before full. No GPU/model inference in secondary.
All codecs through native fleet, scratch-local archives; outputs under Datasets.

Stop: primary decision reached and secondary bounded evidence complete, failed
correctness, cap/deadline exhaustion, or inability to complete inside caps.
Preserve every raw/partial artifact and report deviations. No fitting,
generation, reference changes, full12 confirmation, new segmenter/model inference
or Wan-Animate. Background remains target-derived; source/replay independence
unresolved, two familiar matches,115px exploratory, historical step5 failure.

Pre-codec source-coverage clarification (before any neural/codec result): local
provenance construction shows some original matte pixels lie outside the saved
generated crop. Primary test retains original masks, original R/S and native
Telea fallback and reports every missing pixel. Secondary mechanism uses the
native written support (exact mask intersect available crop) as its edge target,
reports uncovered original-mask pixels separately, and averages only available
generated colors at half resolution. It never fills a gap with target foreground
or claims those colors exist. Hard baseline is binary at the coded grid; zero
codec full-resolution geometry loss is reported separately. Width2/4 ramps are
computed from the same written support. No quality-based selection or threshold
change; this bounds the mechanism's scope rather than fabricating RGB evidence.

Operational resource amendment before first primary scores: exclusive queues
never admitted and were cancelled at0GPU cost. Use already-qualified shared
Ada metrics with12000MiB limit+8192MiB free margin, deterministic saved-pixel
and1e-4 score controls, no latency claim. Reserve1000s first
(100smoke/800full/20validator/80overhead),800recovery; same1800s cap. Both-window
measured smoke projection plus60s must fit800 before full. CPU phase is closed
at a conservative721.524817/3600CPU-s including failed smoke and local allowance.

Final immutable evidence: `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-joint-matte-edge-20261026T131206Z`. Bundle SHA256 `e39c6031cbc0bd54fffc06d2dcca25dacb4d585c993fb587740fc3ac07ca48be`;8 raw/54 metadata files verified read-only; executed-code hashes match native receipts.
