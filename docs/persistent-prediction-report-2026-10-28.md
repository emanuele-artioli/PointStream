# Persistent prediction search: no measured codec win

The approved plan is now in PLAN.md and the persistent-prediction card.
The CPU accounting/compatibility search and ONE GPU request are complete.
No training, new model download, retry or continuation request occurred.
Raw evidence is preserved at
`/home/itec/emanuele/Datasets/pointstream-data/tennis/persistent-prediction-20261028-05c49090/evidence.tar`;
the [verification receipt](persistent-evidence-receipt-2026-10-28.json) records
its SHA256 and member identities. Both original smoke/full publications are
also verified and read-only; task-owned supervisor/campaign PIDs are absent.

## Accepted finite pilot

Job20261028T181634Z-05c49090 completed on gpu2, RTX A6000 UUID
GPU-fc7e8fff-75be-4984-a185-fe8ce63edcd9. Sourceffa4b4642f9ba7b919e65d65e941c458f60c3960;
snapshotSHAa20c7ca001e5b410b57425af839dfdb6f91d7871a674b5ee06fa7e748026e48b;
specSHA3b0ac006c628e9845b9e4bd2d1613d3657b39a4de47ebe3e8024e831d96d904d.
Al031@129 first96nativeframes at25fps, score32ticks every third frame at1080p.
These are familiar development frames, not untouched confirmation.

Both families select32x32blocks from decoded current/previous scale4base and a
paid, camera-warped first-frame reference. The objects family also offers a
previous decoded-base player patch resized/repositioned by delivered boxes.
The receiver gets no original mask or original history. A native SVT residual
stream corrects the prediction. All base/reference/geometry/header/residual
bytes count. There is no generative player overlay or trained content model.
Three residualCRFs43/53/63 and three SVT anchors at those CRFs, preset4,
960x544, native frames/fps. This restricted anchor grid is weaker than the full
multi-resolution conventional/neural envelope; losing to it is meaningful,
winning would still require stronger anchors. No neural-baseline win is claimed.

| Predictor | CRF | Cold total kbps | Cached-reference kbps |
|---|---:|---:|---:|
| Background long/short |43|220.0125|184.1271|
| Background long/short |53|136.2917|100.4063|
| Background long/short |63|90.7417|54.8563|
| Plus object-aligned history |43|223.0083|187.1229|
| Plus object-aligned history |53|139.9000|104.0146|
| Plus object-aligned history |63|94.1917|58.3063|

SVT anchor rates222.5833/135.1458/48.0938kbps forCRF43/53/63.
Post-hoc PCHIP BD-rate over overlapping quality only, positive means worse:

| Predictor / pricing | Whole LPIPS | Whole DISTS | Foreground LPIPS / DISTS | Background LPIPS / DISTS |
|---|---:|---:|---:|---:|
| Background, cold |+248.98%|+169.90%|+134.60% / +118.43%|+252.58% / +180.00%|
| Objects, cold |+254.16%|+176.22%|+146.80% / +128.90%|+257.51% / +185.89%|
| Background, cached reference |+182.68%|+110.69%|+79.98% / +63.86%|+185.94% / +119.59%|
| Objects, cached reference |+188.27%|+116.33%|+92.09% / +74.50%|+191.29% / +124.75%|

No cold point achieves a robust two-metric whole-frame win. The object-history
candidate does not improve the curve. This closes this finite native shifted
pixel-residual construction, not learned conditional features or arbitrary
appearance memories. Regional masks are the frozen encoder model-derived
two-player reference; no human-label or independent-source claim is made.

## What the third axis changes

The initial reference is17225bytes:35.8854kbps over this3.84-second prefix.
Granting it free is an optimistic warm-cache scenario, not verified future
reuse. Even this discount fails. AtCRF43, both whole-frame metrics have overlap
for pricing equality, but ongoing nonasset traffic already exceeds the equality
rate: no longer reference lifetime alone rescues either family at this point.

Background selector7664bytes(15.9667kbps), camera3059bytes(6.3729kbps),
base11079bytes(23.0813kbps). Objects selector/boxes9356bytes(19.4917kbps).
These dynamic charges cannot become installed-model assets.

Combined optimistic relaxation (cached reference AND zero correction payload)
does expose conditional headroom at the best-quality point. Background's paid
nonreference/nonresidual floor is45.4208kbps; LPIPS permits at most32.0424kbps
of corrections at robust equality, versus actual138.7063kbps. Objects allow
29.5087kbps versus138.1771kbps. Thus a cheaper correction coder would need
roughly77%/79% reduction at fixed measured output quality, or materially better
prediction/less dynamic overhead. These are equality ceilings; strict wins need
less. No zero-payload relaxation is a deployable result or proof that a learned
codec can reach the ceiling. See [sensitivity](persistent-memory-sensitivity-2026-10-28.json).

The old clean-rendered player audit also fails static-only discounts: actual
dynamic KPS3.53kbps exceeds its1.619035kbps fixed-pixel joint equality ceiling,
even if every static overhead is granted. See [accounting audit](persistent-accounting-2026-10-28.json).

## Literature and follow-ons

The [card](cards/persistent-prediction.md) preserves the literature-backed search
over conditional feature corrections, long/short decoded references, general
codec-aware restoration, compact paid adapters and explicit cache economics.
PnP-VCVE's published path uses H.264 metadata; our AV1/DCVC path has no audited
compatible checkpoint/metadata adapter. That branch stops at compatibility,
not at a measured quality failure. No neural restoration run was represented
by this native residual experiment.

The next material design would reuse native motion/context inside ONE coding
path rather than pay a base, extra selectors and a separate ordinary residual
video. First establish a compatible implementation and a useful rate bound;
do not fund another generative appearance model or assume a modest adapter
gain can deliver the required correction-rate reduction.

A separate static review found DCVC decoder report variable shadowing: the
YUV frame bytes replaced the container bytes used for its reported SHA. Fixed
the local variable without changing encoding/decoding pixels. Earlier baseline
gates checked the encoder's container SHA and independently failed on pixel
reproducibility, so this does NOT explain or reverse the native correctness
stop. No DCVC GPU rerun was performed; deterministic decoding remains unresolved.

## Verification, resources and preservation

Smoke passes all23 recorded checks, projection732.275s<2100s. Full command
401.654s, native work397.926s, peak allocated/reserved below5000MiB (reserved3740).
Shared-GPU timings are cost accounting only. Source native YUV SHA and all32
original CUDA RGB hashes match; weights/tools pinned; double native decodes and
wire-only predictor replay agree; all score arrays finite.

[Independent audit](persistent-memory-audit-2026-10-28.json) verifies complete
byte totals, IVF framing, saved array means and replay image hashes. It does
not independently rerun neural metrics. Full publicationSHA
2933ca5bcb82f52b832e20c21db7737acf72103d6905b1d816baacc992fa6554.

Conservative new ledger600/3600GPU-seconds,3000unused; CPUaudit/analysis charged
120/3600seconds conservatively. Admission through canonical terminal is under
600seconds, including smoke/full/validation/publication/overhead. Old budgets
remain closed. No retry follows this decision. No timer or model heartbeat.

Code checks: focused wire/document tests, configured Ruff/mypy pass. Full macOS
suite retains two unchanged platform failures; ps test passes outside sandbox.
Linux CI is the merge gate. [PR272](https://github.com/emanuele-artioli/PointStream/pull/272).
