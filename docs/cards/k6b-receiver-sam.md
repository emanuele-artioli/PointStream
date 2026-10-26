# K6b: sampled pose-prompted receiver SAM tracker

Goal: determine whether a single pose-prompted installed SAM3.1 tracker removes
YOLOE's missed-person failure and preserves at least3/4 strict LPIPS/DISTS
metric-cell wins on the same saved raw FULL player crops. This remains a
component of the broader players+ball goal.

Scope: Alcaraz–Hurkacz031@129 and Medvedev–Struff014@194, both players,80
original judged ticks0,3,...237 each; unchanged240 billed KPS ticks, compressed
FULL references and7200s adaptation. Existing environment and pinned SAM3.1
checkpoint only. No fitting, generation, new weights/environments, full240
recovery, full12, background/ball integration, reference changes, Wan-Animate,
model hunt, new chats/agents or recurring model monitors. Prior budgets closed.
New ceiling2700 admitted GPU-seconds including setup/smoke/failures/judge/
validation;3600 aggregate CPU-seconds and2h wall, each waited stage<=45min.
Branch `codex/k6b-receiver-sam-pilot`. Report to planning chat
`01a16cfc-1a88-7013-bd79-8489fe7c1eb2`.

Frozen profile before quality: standalone installed SAM3.1 multiplex tracker,
all931 weights required, native flash/bf16, compile=False. At every20 archived
ticks start a fresh state, one object ID1 local to paid unit identity. Initialize
from independently decoded confident KPS in current crop: bounding corners of
all in-bounds confident joints, expanded10% of span (minimum4px), clipped to
crop; corner prompt labels2/3 plus positive shoulder/hip joints5,6,11,12. If no
torso joint, use first valid joint; fewer than2 valid joints is an invalid gate,
not source-mask fallback. Normalize by512×784. No correction/retry/reselection
based on target; causal forward tracking within each20-tick segment. The
actual sampled frame spacing is3/25 or3/50s (segment2.4 or1.2s); no missing RGB
interpolation. Native PIL square1008 resize, float16 RGB/255 then(.5,.5)
normalization, native logits>0 mask. Projection uses only paid crop rectangle.
Prompt/reset profile and identity order recomputable;0 new content bits.

CPU audit inspects generated Med004 crops with paid poses, not target masks.
Prediction inherits proven Landlock read allowlist: own scratch/code/runtime,
concrete GPU devices, KPS, raw generated RGB, pinned SAM checkpoint. Source RGB,
control and mask-bearing transport denied. Two fresh subprocess predictions
must agree bitwise before evaluator accesses original R/S/Telea. Smoke2 ticks
per all4 units; actual CUDA/flash/memory, every weight, denied reads, output and
identity coverage, repeatability, exact control pixels and per-frame<=1e-4
metrics; measured conservative projection must fit1500s full and remaining cap.
Initial reservation2000s:300smoke/1500full/40validator/160overhead;700s reserve
only after known failure inspection. Full all4×80 actual inference, exact
control and predicted composites, unchanged R/S/crop order/metrics/anchors.
No source RGB filling outside available generated crop. Missing/invalid outputs
fail validity, empty real masks count as empty; no IoU substitute for quality.

Primary success>=3/4 strictly bracketed cells at actual total rate. Common
base+SAM installed assumption; checkpoint bytes and installation/payback
sensitivity separate. One method point/window, no method BD-rate. If cheap
within remaining CPU budget, a single framed independently decoded sampled
XOR residual may diagnose exact correction cost at80 ticks only. No all240 rate
or dense extrapolation. Corrected target pixels unavailable in crop stay absent.

Stop: contract/interface/input/isolation/kernel/repeatability/runtime gate fails;
quality decision reached; cap exhausted. Do not substitute another model. If
pass, prepare separate full240 capture/correction gate; do not launch it.
Retain granted target-derived Telea background, source/replay uncertainty,
familiar two-match sample,115px exploratory and historical step5/granted-matte
limitations. A sampled player result is not an end-to-end win.
