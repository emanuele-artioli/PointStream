# K6b: independent pose-prompted segmentation at every archived frame

COMPLETE:4/4strict wins, frozen+.001sensitivity3/4; all320player masks nonempty.
Same job20261026T184618Z-186fe8b3COMPLETE. Conservative1720/2700GPU-s
(0.477777778/0.75h),2329.366896/3600CPU-s CLOSED. Full36checks pass locally
and in original runtime; no new generation/integration/correction run. Later
full-accounting and fresh-evaluation cards PREPARED ONLY, unapproved/unlaunched.

Goal: test whether removing temporal tracking persistence preserves>=3/4
strictly bracketed LPIPS/DISTS cells with zero transmitted dynamic matte.
This is the last narrow segmentation profile in the current dispatch.

Scope: same Alcaraz–Hurkacz031@129 and Medvedev–Struff014@194, both players,
all80 archived FULL raw RGB ticks0,3,...237/unit. No generation, fitting, new
models/weights/environments, changed references, full240 recovery, full12,
ball/background integration, prompt/confidence search, other chats/agents or
recurring model monitors. Prior budgets CLOSED. Fresh ceilings2700 admitted
GPU-seconds including setup/smoke/retries/validation,3600 aggregate CPU-seconds,
2h wall, stages<=45min. Branch `codex/k6b-receiver-frame-pilot`. Report planning
chat `01a16cfc-1a88-7013-bd79-8489fe7c1eb2`.

Freeze one global policy BEFORE quality: PR246's pinned SAM3.1 standalone
tracker/931 loaded weights, bf16 CUDA native flash, compile=False. Every saved
frame gets a fresh one-frame state/object1 and that frame's independently
paid-KPS-derived prompt: in-bounds confident joints>=.5 bounding corners with
10% span/minimum4px expansion/clipping; corner labels2/3 plus positive joints
5,6,11,12, fallback first valid if torso absent; fewer than2 valid joints fails.
Same512x784 crop geometry, native1008-square PIL resize and float16 RGB/255,
mean/std.5 normalization. Take returned native point-prompt video logits>0.
NO temporal propagation/preflight/mask memory, no prior mask, no identity-
specific choices. Fixed unit identity/order from paid KPS; prompts/reset
recomputable with0additional per-content bits. All4units use this same policy.

Reuse original frozen input hashes/environment/checkpoint in PR246. Keep
Landlock read allowlist to own scratch/code/runtime/concrete GPU nodes, raw
RGB/KPS/checkpoint; source, original masks, controls/Telea denied. Two fresh
processes must match every input/output mask bit, and record exactly80prompts/
unit. Evaluator reads original R(player support)/S(scoring region) and target-
derived Telea only AFTER prediction freeze. No source foreground fills missing
RGB. Physical ticks3/25 or3/50s;160rawticks/player remain absent.

Smoke2ticks/all4units, both independent process resets, actual native device/
kernels/all weights/memory, denied reads, prompt count/identities/output masks,
exact-control RGB and per-frame metric<=1e-4. Conservative projection includes
both full prediction resets and original two-window quality scoring, must fit
1500s full and total cap. Reserve2000s=300smoke+1500full+40validator+160overhead;
700s recovery only after known failure. If infeasible, stop without dropping
units/frames or changing settings. Save complete unit/window artifacts.

Full real all4x80 independent inference + original LPIPS/DISTS judging. Success
>=3/4 strict bracketed cells at actual total rate with original achievable
SVT-AV1/DCVC envelope. Report frozen additive method-error sensitivity±.001
alongside unmodified primary. Exact control must reproduce saved RGB and
scores<=1e-4. Empty masks, IoU, boundary F1 at2px tolerance and unsupported-RGB
are diagnostics, never substitutes. One point/window, no method BD-rate or
quality/rate extrapolation. Keep all240 paid KPS, actual compressed FULL banks,
7200s adaptation; common installed tennisbase+SAM explicit, checkpoint bytes/
installation and payback sensitivities separate. Shared timings cost only, no
latency claim. No correction-coder rerun: prior sampled XOR already measured.

Stop: validity/runtime/budget gate fails; primary decision reached. No next
model, generator, segmentation-profile search or full240 regeneration. If fail,
record exact cause and recommend whether this generator/silhouette recipe
closes or needs a genuinely new decoded-base enhancement architecture oracle.
If pass, only PREPARE a separate full-accounting card with coded background or
whole-frame decoded base+players+ball, all transport/reference/adapter/fallback
bits, fair adaptation, equal-total-rate metrics, achievable anchors and simple
ROI/pixel enhancement alternatives; read negative T1 fixed-grid findings and
first require cheap valid oracle/rate bound/full240 necessity. Do not launch it.
Prepare fresh-evaluation plan: current two windows are DEVELOPMENT, not untouched
confirmation; new footage/match/train/reference/judge provenance must pass before
any future headline. No new acquisition or GPU evaluation in this study.

Retain granted Telea/background and historical granted-matte limitation,
source/replay uncertainty, familiar broadcasts,115px exploratory and historical
step5 failure. No sampled result proves a full240 delivered codec/end-to-end
win; broader desired claim includes players+ball.
