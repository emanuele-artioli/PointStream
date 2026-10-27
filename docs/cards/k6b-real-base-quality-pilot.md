# Actual-base player enhancement pilot — CONTROL GATE FAILED

Goal: test whether the frozen PR242 lossy-reference generation and PR247
per-frame SAM players improve a REAL decoded whole-frame SVT base at equal
TOTAL rate on the same Al031@129 and Med014@194 development windows. Use only
the saved 80 ticks per window and the three new decoded-base scale8/4/2
settings IF their streams, pixels and identities pass the CPU recovery gate.
No target-derived Telea or uncharged foreground pixels in the delivered arm.
Retain PR247's automatic target R/S solely for frozen foreground evaluation;
use full-frame LPIPS/DISTS/PSNR including seams and background as co-primary
metrics. Source foreground and ball may be granted only in a labeled finite
oracle; include the actual frozen overlay in that oracle's tested envelope
because LPIPS/DISTS are not mathematically monotone. A negative result bounds
only the tested combinations. No new generation, model, fitting, reference or
ball inference.

Account the complete 240-frame IVF bytes including 32+240×12 framing, KPS
packet, actual FULL bank and adapter/guider allocations, model-installation
sensitivity, bootstrap, geometry, timing and outer framing; any unmeasured
required terms are explicit. Compare against complete whole-frame SVT and
DCVC encodes of the same intervals with all container bytes and fair
adaptation. The old player-region allocated rates and quality are NOT whole-
frame anchors. The six newly validated SVT streams can provide actual whole-
frame bases and SVT anchor points after independent quality scoring. Same-
window DCVC metadata exists, but its bitstreams/pixels were not retained in
job 29586984; a DCVC comparison needs separate native reproduction and new
scores, with a measured independent budget. Do not label the SVT-only result a
complete two-codec win. Add a verified native ROI method or paid pixel-crop/
residual enhancement baseline using identical base, side information, costs,
source timing and fair adaptation; demonstrate codec support first.

Predeclare one global compositing and selection policy, no per-window
quality-based settings. Preserve exact original R/S and tick memberships,
reference/crop order, source conversion and source identity. The experiment
validator must independently read/hash every input, decode and validate streams,
verify 240 transport frame identities and the selected 80 RGB ticks, verify
actual composite hashes and original control within 1e-4 where comparable,
prove no target-background/foreground grant in the delivered path, assert
finite real neural outputs and device/weights, complete full-frame/foreground
score arrays, total-byte ledger and anchor envelope coverage. A same-base
no-overlay control must be independently scored; source-target appearance
oracle is diagnostic only.

Before scoring, freeze >=3/4 strict LPIPS/DISTS full-frame cells AND >=3/4
fixed-foreground cells at equal total rate, with fixed +/-0.001 error
sensitivity. If three measured method points have overlapping quality with
an achievable anchor envelope, additionally require mean BD-rate <=-10% and
at least 75% window wins per perceptual metric. Report actual dominance or
nonoverlap, never extrapolate or invent BD-rate. No threshold may be tuned on
new scores. Without a fair valid ROI/pixel alternative or DCVC curve, report
missing controls and do not promote a broad baseline claim.

Resource quote is authorized for this bounded study: total ceiling of 1.0
GPU-hour for this SVT-only intermediate, with <=600 s smoke, <=2400 s full
stage, and 600 s reserved for staging/validation/overhead. A provisional
0.5–0.85 GPU-hour estimate is a planning range, not a measurement; stop if
the unchanged representative smoke cannot project all work within the cap.
A separate DCVC encode/score budget must be measured and approved; it is not
included. Historical actual judge
20261026T092648Z-ccf0633e cost 332.187562 GPU-s for its 160-frame saved-render
comparison; independent-frame study 20261026T184618Z-186fe8b3 used 1639.081948
admitted GPU-s including two SAM passes, judging and overhead, with shared-GPU
timing. These are cost references, not a validated quote for larger full-frame
judges, extra streams, ROI controls or DCVC. The authorized <=600 s
native correctness/resource smoke must measure memory, complete
per-frame real quality time, decode and validator overhead on representative
1080p frames and both streams; use it to quote and gate bounded full stages,
with every waited stage <=45 min. No GPU job was launched: the mandatory
exact source-RGB control failed.

This is players-only intermediate feasibility. No qualified ball metadata or
payload on these exact windows has yet been located; the base retains its
encoded ball, but a separate paid ball-update integration requires its own
capture and accounting. Missing 160 generated ticks per player mean no
full240 semantic overlay/wire claim; sample-only results cannot support one.
The current two broadcasts are familiar development data with unresolved
source/replay independence; fresh confirmation follows method freeze and its
separate lineage audit. Historical 115px cutoff remains exploratory and the
step5 failure remains part of the record.

## 2026-10-27 — Exact source RGB gate failed; stopped

The human approved <=3600 admitted GPU-seconds for this study; the specific
stop rule requires exact CPU-versus-original-GPU RGB identity before quality.
The source YUV hashes agree for all six base settings, and the original 80
judged ticks agree in identity. However byte-exact CPU source RGB hashes match
only **10/80 Al** and **0/80 Med** original GPU hashes. This is a hard control
failure. No tolerance is substituted and no new score or GPU job was run.
It does not establish the magnitude of the pixel differences, identify their
precise numerical cause, or show a scientific gain or loss. The CPU-versus-GPU
identity of the six decoded-base RGB arrays has not been separately measured.
The underlying source YUV, encoded streams, model inputs and prior results remain
unchanged. Exact mismatch ticks and reproducible identity-checked comparison
are in [gate result](k6b-real-base-source-rgb-gate.json) and
[source comparison](../../tools/analysis/k6b_source_rgb_gate.py). The study
stops with **0/3600 GPU-s** spent; old budgets stay closed.

The cheapest next separately scoped decision is a native GPU source and base
conversion diagnostic, using saved YUV and original function to compare RGB
hashes and pixel-difference distributions against CPU while reproducing PR247
source hashes. If successful, reconstruct base RGB in the same native GPU path
and freeze new identities before a quality smoke. Measure that diagnostic's
cost and request its own budget before dispatch; no approximation, threshold
change or assumed float-rounding explanation is authorized here. If exact GPU
source hashes also cannot be reproduced, investigate version/device/environment
rather than judging. DCVC, ROI, ball, missing 160 generated frames, development
and granted-background limitations remain as above.
