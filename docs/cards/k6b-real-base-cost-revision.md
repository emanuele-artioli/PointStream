# Real-background player study: prospective cost revision

Status: human accepted on 2026-10-27 in the resumed root chat (annotation
"Approved"). This prospectively amends the full-stage reservation to2600s
within the unchanged3020s remainder. A fresh passing smoke, current job/worker
health and all original gates still precede GPU execution. This is a revision
of the existing study, not a new experiment or a budget reset.

## Decision and reason

Finish the frozen two-window test of saved generated players over a real
decoded SVT background before investing in the remaining codec and TOMM
evaluation work. The previous full stage never ran: its smoke projected
2227.523141 worker seconds / 2448.314646 seconds including publication against
an 1800-second stage reservation. The controls passed; no scientific quality
result was accepted. Raising that reservation needs prospective acceptance,
even though the study has unused budget.

Recommend changing only the stage allocation and its matching validator limit.
Keep the original conservative estimator, including its 35% multiplier and
40-fold publication allowance. Do not claim an optimization or a speedup.
An implementation redesign would invalidate more evidence without a measured
reason to expect it to be necessary.

## Proposed allocation

The original 3600-second study has a conservative charge of 580 seconds
(434.291084 actually admitted). Its remaining ceiling is 3020 seconds.

| New request component | Maximum seconds |
|---|---:|
| Fresh representative smoke worker | 240 |
| Full worker, both windows and all original controls | 2600 |
| Smoke external validator | 120 |
| All other admitted setup, staging, publication and supervision | 60 |
| Total remaining admission cap | 3020 |

The full worker reservation changes from 1800 to 2600 seconds, also above the
original card's initial 2400-second ceiling; acceptance explicitly amends both.
2600 seconds is 43 minutes 20 seconds, within the 45-minute stage bound. The
full projection including publication must be strictly below 2600 seconds.
The historical estimate leaves 151.685354 seconds under that gate, after its
existing safety multipliers; this is a planning basis, not a validated quote.
The new smoke must pass anew on the actual admitted host. The previous smoke
worker took 80.282832 seconds, and its entire request admitted 151.469499
seconds; these shared-host observations do not guarantee the new allocation.

The fleet's 3020-second supervisor limit is authoritative: time in setup,
publication or validation cannot create an extra allocation. Its existing
remaining-budget check must still be able to reserve all 2600 full-worker
seconds after the smoke, validation and pre-full work. If not, stop. The 60
seconds is a residual planning allowance, not an individually enforced phase
timer. Unused smoke/validator time remains inside the same total ceiling.
No retry or transfer from closed studies is authorized. Record actual admission
and conservative charge separately; retain the 580-second prior charge.

## Fixed science and promotion gates

The [governing card](k6b-real-base-quality-pilot.md) remains authoritative for
everything except the explicitly accepted resource amendment. In particular:

- Same Al031@129 and Med014@194, all 80 original saved ticks per window,
  original sources, GPU-derived base arrays, masks, references and generators.
- Same global scale4 central point, scale8/2 curve points, LPIPS/DISTS rules,
  fixed R/S foreground evaluation and +/-0.001 sensitivity. Require >=3/4
  strict full-frame and >=3/4 foreground cells. Preserve the PCHIP/envelope
  overlap rule, mean BD-rate <=-10% and >=75% window wins per metric where
  the required overlap exists; never extrapolate.
- Both original granted-matte/predicted-mask control variants on both windows
  must reproduce before opening any actual-base quality. Fresh smoke uses the
  unchanged ticks [0, 3], six base/hybrid paths and existing identity checks.
- Preserve the full-stage arithmetic early stop when two central-point metrics
  lose in either family in the first window. Unresolved cells are not losses.
- Rates remain explicitly optimistic floors until receiver tensor directory,
  bootstrap and complete framing are independently serialized and decoded.
  A positive result is conditional; a negative rejects only the tested recipe.

No generation, training, new methods, missing160 ticks, new baseline, ball
inference or new footage is included. These are development windows, not
independent confirmation. No complete two-baseline or submission-ready claim.

## Accepted execution preparation

1. Recheck the three known jobs through canonical fleet status and resolve any
   management-path failure before dispatch. Inspect unknown jobs, never replay.
   Check workers and all compatible A6000 hosts (gpu2/gpu3); preserve the health
   gate and pending worker rollout. Verify separate host-local disk capacity,
   environment and immutable staged inputs. No storage maintenance is included.
2. Make the validator's full-reservation limit explicit from the frozen saved
   specification; verify that both projection checks use that same limit and
   reject nonfinite projections. Test rejection below and acceptance above the
   saved 2448.314646-second estimate, and preserve the historical1800-second
   limit when that is what an old saved specification declares. No metric or
   estimator change is needed.
3. Derive a new specification from `k6b-real-base-quality-spec.json`, preserving
   all input/environment hashes and processing arguments. Set smoke240,
   full2600, validator120, budget3020 and a fresh absolute deadline. Freeze
   source/spec/checksum identities before submitting exactly one new request.
   Save its ID in the experiment entry before waiting.
4. Attend execution; use one bounded fleet wait and retain its handle. Do not
   assume completion callbacks are verified, install recurring monitoring or
   treat a lost receipt as permission to resubmit. Stop on any failed control,
   feasibility, resource or decision gate. Preserve partial and failed evidence.
5. Independently audit the full published arrays, frame/composite identities,
   rate ledger and comparisons before accepting a scientific result. The fleet
   external validator currently runs only on smoke; successful process exit is
   not an independent full-result audit. Record CPU audit cost separately.

## Route toward the paper after this decision

If feasibility survives, next close receiver serialization and paid delivery,
then qualify same-window native DCVC and fair ROI/pixel-residual comparisons,
and integrate the paid ball component. Each compute stage needs its own
prospective evidence and budget; this proposal funds none of those stages.
Freeze the resulting method before untouched evaluation and source/replay,
training/reference/judge lineage audits. Only then can the final paper claims,
manuscript completeness and TOMM submission requirements be assessed. If this
study fails its scientific rule, report that negative result and decide the
research direction before funding another method.

## Execution preflight outcome — 2026-10-27

The accepted allocation is recorded; the validator reads `full.seconds` from
the frozen request's spec.json and records it in validation.json. Its estimator
and scientific path are unchanged. Sixteen focused projection/curve/compositing
tests and configured ruff/mypy pass. The [revised specification](k6b-real-base-quality-revised-spec.json)
is prepared with a16:00 UTC deadline (17:00 Vienna). It was **not submitted**;
if that deadline expires before fleet recovery, refresh it prospectively before
freezing/submitting, with the same allocation and identities.

Canonical status rechecked all three prior requests as COMPLETE/COMPLETE/FAILED.
The compatible pool cannot pass admission: initial inspect timed out on both
hosts; both subsequent doctor calls verified gpu2 but timed out on gpu3 at the
unchanged20s limit. Worker status also reported stale gpu2/gpu3 heartbeats.
Read-only extended timing diagnostics completed in28.25s/36.02s without the
explicit login-shell wrapper and showed latency across shared-filesystem path
and claim reads, executable discovery and tool-version calls. These are not
passing health gates or qualification timings. gpu2's diagnostic showed enough
free GPU memory for the declared shared margin; gpu3's did not. No worker was
restarted, health check relaxed, compatible host omitted, or job submitted.
The next action is to resolve the fleet-management latency and obtain fresh
health evidence, then run this accepted study through its original gates.
