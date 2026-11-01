# W8 receiver: source-package repair decision

**Status: user-approved attempt completed and stopped on 2026-11-01.** The user selected “Approve the bounded source-package repair (recommended)” for one replacement cost-only campaign, at most580 CPU-seconds and500 GPU-seconds within the original3,600-second totals. Actual dispatch still requires the recorded source, host, identity and budget gates.

The W8 receiver test is intended to check whether a receiver can reconstruct the original one-window result from its paid packet and installed assets. Its cost pilot stopped before producing learned output because the frozen source package omitted two Python helper files required by the packet decoder. This is a source-packaging failure; it is not an observed generation or quality failure. The original stop rule requires a user-approved amendment before a replacement attempt.

A local static-import regression now detects the omission and confirms that adding the two tracked helper files closes the checked packet-decoder import graph. A separate CPU-only run using the actual isolated decoder has completed successfully, with no GPU work. Its independent audit and root comparison passed: all480 tensor fingerprints, both decoded-unit hashes,237 source identities and the recorded isolation gates match. The CPU repair gate is accepted. Neither result establishes GPU execution or fresh pixels.

The failed cost job `20261101T135101Z-d67fe593` is closed at 208 CPU-seconds and 128 GPU-seconds. The repair qualification is charged its full 40 CPU-second reservation: total W8 usage is bounded by **2,165 CPU-seconds and 352 GPU-seconds**, leaving at least **1,435 CPU-seconds and 3,248 GPU-seconds** under the original separate 3,600-second limits.

**Proposed amendment:** allow one replacement cost-only campaign using the corrected frozen source package: a one-step smoke and a three-step run, with at most **580 CPU-seconds and 500 GPU-seconds** inclusive. These are upper limits within the original budgets, with no added budget, new models, training, new encoding or change to the scientific decision rule. The original 240-frame context, 80 selected ticks, 30-step final recipe, and pixel, mask and metric gates remain unchanged. The replacement outputs would measure cost and runtime behavior only; they would not be quality evidence.

If the replacement consumes its full proposed limits, at least **855 CPU-seconds and 2,748 GPU-seconds** would remain. This does not establish that the final two-unit 30-step witness fits. That work requires a complete forecast from the measured cost pilot, downstream and validation reserves, and separate admission gates; it must stop if those gates do not fit. A failed repair audit, failed smoke, exceeded limit or failed projection also stops this proposal.

**Decision received:** one bounded replacement cost campaign approved. No automatic retry after a new stop or final30-step admission follows.

## Outcome

The sole replacement `20261101T151030Z-df638d2f` passed packet decoding but stopped at CUDA initialization, before model loading, kernel profiling or sampling. Underlying runtime cause is unresolved; there is no fresh quality or cost-slope result. Root accepted292 CPU-s /188 GPU-s inclusive accounting (administrative telemetry limits disclosed in W8), cumulative2,457 CPU-s /540 GPU-s; no live job/reservation remains. Full30-step work is gated off and no retry is authorized. The empirical manuscript retains the unqualified fresh-decoder limitation.
