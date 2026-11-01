# Exclusive-GPU scoring amendment — authorized 2026-11-01

The user explicitly approved this exclusive attempt within the original W8
totals. The earlier 300/300 amendment's two requests remain closed and their
charges remain included below. This approval permits one new frozen request
after the stated checks; it does not permit replaying either old request.

The unchanged objective is to score the saved `nad000` `s2_o20` CRF-67
stream SHA256 `182c7eee4a0ea2e98ef0a967ecfcdf8958898ad7ce035e54f0ea91b5109b6fa3`
and the same three original CRF-63 controls at all 80 ticks, using the same W1
scorer and weights. No encoding, training, changed models, changed quality
metric or changed candidate is proposed. A passing independent audit and strict
four-family raw-point comparison at the fixed W8 rate 46.315062222222224 kbps
remain mandatory before the original W8 gates may resume. Failure, missing
coverage, an identity mismatch, insufficient projected resources, or any
control outside 0.001 stops promotion; no extrapolation.

**Evidence and limit.** First request `20261101T071706Z-afde26d5` passed all
smoke correctness checks but stopped because the original 1.35-times W1
projection was 170.7294 s, exceeding its 160 s full allowance. No full stage
ran. Second request `20261101T073429Z-04a50c10` failed its 20 s smoke timeout
while sharing gpu5 with a detected foreign GPU process; no full stage ran and
no new quality point was measured. Contention is observed, not proven to be the
cause of the timeout. Preserved local diagnostics are
`/private/tmp/w8-amendment-failed-20261101/validation.json` and
`/private/tmp/w8-amendment-second-diagnostics-20261101/{campaign-error.json,run/status.json}`.
Conservative charges are 33 and 34 seconds on each of CPU and GPU; with 20
seconds of amendment preparation, this amendment closes at at most 87 CPU-s
and 67 GPU-s. Original W8 cumulative charges are at most 937 CPU-s and 67
GPU-s of its unchanged 3,600/3,600 ceilings.

**Approved authority.** Permit at most 450 GPU-s and 450 workload CPU-s for
this scoring portion, *within* the original W8 caps and inclusive of the
previous 67 CPU-s/GPU-s and preparation; this is not 450 seconds added to
previous expenditure. Reserve a single 280 s one-core campaign, 30 s for
monitoring/cleanup, 10 s for the independent full audit and 30 s for all
preparation/final publication, including the 20 s already charged. With the
previous 67 s this totals at most 417 CPU-s and 377 GPU-s, leaving slack inside
the proposed caps. Failed work is charged. A new allocation must account for
all work and never replay an uncertain submission.

The proposed `exclusive-prospective-spec.json` keeps gpu2/5/6 as the complete
known compatible pool, uses standard exclusive admission with contention stop,
one CPU, deferred failed publication, 40 s smoke, 195 s full, 10 s validator,
and a 280 s campaign cap. Exclusive admission avoids intentional sharing with
other work; it does not guarantee speed or prevent later contention. The pool
and each worker's guard-aware release, queue, host fitness, native tools and
input pins must be freshly verified. The original W1 1.35-times projection
must pass, and the remaining-time projection must include second staging with
at least 1.25 times observed first setup, full allowance and an additive
publication reserve of 5 s. A hard 200 KiB full-output limit is proposed; it
must be enforced before publication. A projection failure stops before full.
The 10 s audit must be independently hard-bounded; a previous development audit
of the retained W9 full result used 0.122 CPU-s on this Mac, which is not a
guarantee for the new output or environment.

**Execution prerequisites.** The archived prospective spec retains its expired
2000 deadline and is not itself runnable. The newly approved source policy,
focused checks, immutable revision/spec, fresh deadline and preflight must be
frozen in a separate execution specification. Do not infer a scientific result
from the earlier failed attempts or this allocation. The quality/input/receiver
gates remain unchanged.
