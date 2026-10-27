# Actual-base player enhancement pilot — GPU SOURCE CONTROL RECOVERED

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
with every waited stage <=45 min. Initially no GPU job was launched because the mandatory exact CPU/GPU source
RGB control failed; the later same-budget diagnostic is recorded below.

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
only **10/80 Al** and **0/80 Med** original GPU hashes. This was a hard control
failure at this checkpoint. No tolerance was substituted; the later GPU
diagnostic is recorded below.
It does not establish the magnitude of the pixel differences, identify their
precise numerical cause, or show a scientific gain or loss. The CPU-versus-GPU
identity of the six decoded-base RGB arrays has not been separately measured.
The underlying source YUV, encoded streams, model inputs and prior results remain
unchanged. Exact mismatch ticks and reproducible identity-checked comparison
are in [gate result](k6b-real-base-source-rgb-gate.json) and
[source comparison](../../tools/analysis/k6b_source_rgb_gate.py). At that checkpoint **0/3600 GPU-s** had been spent; old budgets stayed closed.

The next decision at that checkpoint was a native GPU source and base
conversion diagnostic, using saved YUV and original function to compare RGB
hashes and pixel-difference distributions against CPU while reproducing PR247
source hashes. If successful, reconstruct base RGB in the same native GPU path
and freeze new identities before a quality smoke. Measure that diagnostic's
cost and resolve its authority before dispatch; no approximation, threshold
change or assumed float-rounding explanation is authorized here. If exact GPU
source hashes also cannot be reproduced, investigate version/device/environment
rather than judging. DCVC, ROI, ball, missing 160 generated frames, development
and granted-background limitations remain as above.

## 2026-10-27 — Same-budget conversion recovery authorized

Planning clarified that the required conversion-control diagnostic is part of
the user's existing 3600 admitted GPU-second study, not a new budget. At this
amendment 0 GPU-s has been spent and no new quality scores have been opened.
Reserve at most 600 admitted GPU-s including setup and validation for the
native conversion diagnostic; all costs subtract from the same 3600-second
cap. Original exact control is unchanged; do not adopt a relaxed tolerance.
A representative smoke must project the full diagnostic within the reservation.
On original A6000-class native code/runtime, reproduce all original 160 GPU
source RGB frame hashes, compare actual CPU and GPU pixel differences, then
only on exact source success regenerate six 80-frame decoded-base GPU RGB sets
with frame hashes, independent conversion repeat and retained-YUV validation.
No quality metrics are accepted in this diagnostic. Stop on any failed identity,
resource gate or control; full hybrid judging may resume only with exact source,
base conversion and original granted-mask/SAM control validation and sufficient
remaining original budget. No generation, new weights, ball or DCVC.

Conversion diagnostic submitted as `20261027T074823Z-135421ee`, identified
from the saved spec/ready record after the client receipt was lost. Pending
at discovery; inspect this ID and do not replay. Total reservation <=600 s,
charged against original 3600 s. Frozen worker source revision `146f06ee`.
Smoke 20/20 checks passed, reproducing 8/8 prior GPU source hashes per window;
the conservative full projection including publication was 306.032/360 s.
Full diagnostic running under the same job; no new quality scores.

Before any hybrid quality: the strict >=3/4 full-frame and >=3/4 foreground
cells use the single global **scale4** operating point for both windows;
scale8/scale2 form the other two curve points. This choice is based on the
pre-existing native geometry/rate inventory, not new quality, and avoids
post-hoc per-window operating-point selection. The achieved anchor envelope
here is limited to the three recovered SVT points; missing stronger SVT, DCVC,
and ROI/pixel alternatives remain explicit. If GPU control validation or the
remaining-budget smoke fails, no hybrid score is accepted.
Full conversion completed with 80/80 original GPU source hashes/window and
six repeated GPU-base conversions. Conservatively charge 370/3600 s including
admission, smoke and receipt overhead; remaining 3230 s. CPU/GPU source max
1 level, with tick0 numerical attribution recorded in the experiment. The
next gate is independent full artifact validation and original granted-matte/
SAM control calibration, then a representative quality-smoke projection.

## Receiver accounting clarification before quality

The planning code audit found that PSTMContainer provides real typed framing
and KPS1 is independently decodable, but the existing quantized adapter/guider
billing does not serialize its receiver tensor-name/shape/dtype/offset and
quantization/bootstrap directory. The diagnostic must not call this fully
delivered. For the authorized development smoke, use a predeclared optimistic
rate floor: full IVF bytes, actual framed KPS bytes, actual reference-bank and
quantized model payload bytes amortized at 7200 s, plus the explicit research
bootstrap JSON emitted by the worker. Installed common model assumption remains
separate. The research header is not proof that every tensor/bootstrap field,
transport/fallback or an independently decodable full240 semantic receiver is
present. Missing metadata cannot be priced as zero for a delivered claim.
A negative result even at this floor can reject this finite configuration; a
positive result is conditional and requires actual complete receiver framing.
No framing selection may be tuned after seeing quality. The original strict
quality rules, source and granted/predicted control gates do not change.

The prospective real-base scoring specification reserves <=2500 admitted s
from the original study after conservatively charging 370 s for conversion;
thus even a full reservation and prior cost total <=2870/3600 s. Smoke <=500 s
uses two predeclared ticks/window, both original granted/predicted controls
before any real-base scoring and all three fixed rates; full <=1800 s. The
substantive smoke validator additionally projects publication and all 160-frame
work within the full-stage reservation. If it fails, full does not launch.
The bootstrap JSON is a counted research header only; omitted adapter tensor
directory and other delivery framing keep the point an optimistic floor.
At full scale, if both central-point metrics in either family lose in the
first window, the 3/4 rule is arithmetically unattainable and stops early;
unresolved cells are not losses. Otherwise evaluate both windows.
