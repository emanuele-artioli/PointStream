# Real-base recovery with separated timing — APPROVED AND EXECUTED — CLOSED 2026-10-27

The user approved the prior same-cap allocation, and execution proceeded.
Request20261027T181542Z-b22436b0 then hit its120s smoke limit. No full stage
launched. The accepted plan says "No retry or transfer from closed studies is
authorized" and "Stop on any failed control, feasibility, resource or decision
gate." The human explicitly accepted this revised model and allocation on2026-10-27.
Exactly one fresh request is authorized after CPU checks and freezing.
It is not acceptance of the failed smoke, a quality result or additional budget.

## Evidence and remaining authority

The experiment ledger records every request and conservative charge:
580s original study +120s cancelled setup +60s import-only failure +270s last
smoke/setup/cleanup = **1030/3600s**, **2570s remaining**. Canonical terminal
reconciliation of the latest request was265.720778s after supervisor start;
270s covers that interval. Its measured primary interval was210.138881s.
The last partial archive is preserved, SHA256
`e6a2a8702871f2de95cc60abf6d65d0192a5f41b249351016a4be9f6d8eb5bcf`.

Independent CPU inspection of ticks0/3 in both windows finds source/composite
hashes identical to the prior controls and maximum LPIPS/DISTS score difference
zero. Only Al's two real-base ticks completed. The missing Med arm, resource
summary and promotion gate make the smoke incomplete; none may be inferred.
This gives no accepted full quality result or BD-rate.

The old model uses calibration total_seconds/frame_count as a per-frame slope,
then adds control_elapsed. Last calibration totals were22.587890/15.300573s;
setup was13.885248/11.595944s. Per-frame costs were Al6.708889/1.993557s and
Med1.912349/1.792083s. Scaling all setup80-fold overstates its full contribution;
subtracting it from an already failed quote is not a valid fresh measurement.
A one-window extrapolation of the truncated smoke is not a promotion quote.

## Prospective resource contract

| Stage | Maximum admitted seconds |
|---|---:|
| Fresh three-tick smoke in BOTH windows | 300 |
| Full worker, both windows, all80 ticks and original controls | 2000 |
| Smoke external validator | 120 |
| All other setup, staging, publication, supervision and cleanup | 150 |
| Total remaining ceiling | 2570 |

The300s smoke exceeds the last accepted240s smoke allocation, and the timing
model changes. The human acceptance explicitly authorizes both, within the original
3600s study ceiling. No closed budget transfers or additional compute are
requested. The150s residual is planning headroom; the native supervisor's
2570s total and remaining-reservation check remain binding. Stop if the full
reservation cannot fit. Every waited stage stays below45min.

## Timing model to implement and test before freezing a new request

Instrument per-frame elapsed time around the existing production operations;
change no images, masks, crops, metrics or GPU processing. Measure the complete
command duration from native execution.json, including imports; the internal
run() timer starts after imports and is insufficient. Also instrument native
publication/release duration directly, rather than mislabeling the entire
dispatch-minus-internal-runtime gap as publication. Use three saved ticks
0/3/6 per window for calibration and real-base scoring, providing a cold sample
and two steady samples in each of four timing sequences. Assert that every
sequence has finite positive times and both steady samples differ by<=25%.
Otherwise the quote fails; do not add GPU trials automatically.

For each sequence let `warm=max(t1,t2)` and
`cold=max(0,t0-warm)`. Let
`fixed=native_command_seconds-sum(all twelve frame times)`;
reject negative/nonfinite fixed time rather than clamp it. Project the full
worker as `1.35*(fixed + sum(80*warm+cold))` over the four sequences. Account
setup once, conservatively repeat the slower observed warm frame, and include
each observed cold excess once. Preserve the35% safety factor.

Add `1.35*(160/6)*measured_publication_seconds` for full publication, deriving
the multiplier from the frozen160/6 full/smoke tick counts. The old40 factor
was160/4; do not drop publication or its safety factor. Projection must be
strictly below2000s, and all data/control/identity/device/memory checks must
pass independently. Sharing/continued contention is recorded as cost only;
unstable samples fail, not an excuse to assume an idle-GPU speedup.

The three-tick smoke is the bounded cost pilot and scale confirmation; the
remaining extrapolation from short samples must be reported honestly. There
is no guarantee it will fit. A failed quote or exhausted cap closes execution,
preserves artifacts and requires a new research decision before another method.

## Science and delivery scope remain frozen

Same two development windows, saved80/240 ticks, native SVT scales8/4/2,
global scale4, original granted-matte and saved-SAM controls, exact source RGB
and composite hashes, original1e-4 score tolerance. Full stage rechecks both
controls on all80 ticks before real-base assessment. Same >=3/4 strict cells
for full-frame AND foreground LPIPS/DISTS; same +/-0.001 sensitivity,
achievable anchor hull, overlapping-range PCHIP BD-rate, >=75% window wins
and <=-10% mean per metric, arithmetic early stop and unresolved-cell handling.
No generator, model, reference, training, ball, DCVC or ROI substitution.

Keep the legacy estimator/default request behavior unchanged. Unit tests must
cover fixed/cold/warm accounting, missing/invalid/unstable samples, publication,
strict boundaries and saved-spec selection. Review the diff to prove changes
are timing/instrumentation only; run configured ruff, mypy and pytest/CI.
Freeze the new source/spec/input identities, inspect all old requests and both
eligible hosts, then submit exactly once under the recorded human acceptance.
Use one native bounded observation, acknowledging meaningful events before
resuming within its original deadline. No recurring model monitor.

A positive result remains optimistic SVT-only feasibility, with missing complete
receiver payload/framing and160 generated ticks, unresolved development lineage,
no fair ROI or neural baseline, and no ball update. Independent full-result
CPU audit precedes accepting evidence. A negative scientific decision closes
this tested route; it does not license a fresh generator search.

## Executed outcome

Fresh gate passed, full stage completed with the registered scientific early
stop after Al loses both whole-frame metrics. See the
[audited report](../k6b-real-base-quality-report-2026-10-27.md).
Final job20261027T211451Z-f6a754eb,1320s conservative new charge; study
2350/3600s,1250unspent. This card is closed and authorizes no additional retry.
