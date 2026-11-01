# Manuscript readiness, 2026-11-01

The decoder witness is **one gate, not the complete submission checklist**.
The full [task audit](task-audit-2026-11-01.md) and
[claim-to-evidence matrix](evidence-matrix-2026-11-01.md) distinguish completed
studies, remaining scientific gaps, optional improvements and deferred roadmap
items. The author-approved empirical framing can report positive and negative
findings; it does not require every historical PLAN item to succeed. It also
does not turn unmeasured comparisons into completed evidence.

## Completed in this phase

- Updated from main including merged PR #317, and delegated three independent
  Terra 6.1 medium tasks: full task/evidence audit, negative-evidence manuscript,
  and W8 payload/oracle helpers. Requested model/effort are known; actual runtime
  telemetry and the other Codex tier-validation gaps remain unverified.
- Added an online appendix with bounded hand-mesh and object-motion/view-bank
  negatives and egocentric equal-adaptation results. The LPIPS loss remains
  beside DISTS/PSNR gains; post hoc selections and failed decodes are disclosed.
  A cost-stopped smoothing pilot and absent racket masks are not presented as
  quality negatives. See [negative-evidence audit](negative-evidence-2026-11-01.md).
- Overleaf main `eb7e3a3f67a3af7ce3e0da20387f6f138a66648f` includes that appendix, the
  hand-transfer/reference-bank negatives and the passing conditional W8
  saved-quality screen described below.
  PDF `/private/tmp/pointstream-w7-manuscript/build/main.pdf`: **26 pages,
  body plus references 1–23, Online Appendix 24–26**. This meets the user's and
  TOMM guide's allowance of **23 + 5 additional appendix pages**. The empirical
  checker passes; modified results and appendix pages were visually reviewed;
  no overfull/undefined-reference warning remains. Venue-specific review format
  still needs final confirmation against the live filing instructions.
- Assembled a real, self-delimiting W8 candidate and independently verified its
  byte identities, rate accounting, exact original IVF identity and saved-anchor
  arithmetic. The local checkpoint diagnostic matches all 480 tensors byte for
  byte, including the added tensor directory and sign information. A frozen
  implementation reproduction at `8f6d0ece` matches both complete wire objects.

## Current evidence and resources

The [W8 rate screen](../experiments/w8-inputs/exclusive-20261101/rate-screen.json)
now **passes all eight cells** after independently audited exclusive request
`20261101T081828Z-d70bc44f`. The fixed 2,289,836-byte candidate costs 3,816.4 kbps
cold for 4.8 s; 46.3151 kbps assumes exact named reference/match objects reused
for 7,200 s, with reuse quality unproved. The additional 49.6733 kbps ROI-SVT
point closes the original 45.8233 kbps gap. All four families bracket the rate,
without extrapolation, and saved hybrid player quality favors both metrics.
This optimistic one-window saved-quality screen is not fresh output quality,
complete receiver qualification or a change to the W7 population result.

All three scoring jobs are terminal, process groups/claims released, waits
ended and completion events acknowledged. Inclusive scoring charge is 224 GPU-s
and ≤304 CPU-s of the approved 450/450 internal ceilings, inside original 3,600/3,600.
The initial two cost stops and initial audit's positional-name failure remain
recorded; reviewed auditor `142b6cc5` resolved argument binding without changing
scores, identities or thresholds. The full audit passed all 12 checks on 80 ticks;
maximum original-control difference 5.2273e-5 <0.001.

The Linux CPU input oracle `20261101T093351Z-24914a4d` passed all four
independent checks for one profile from frozen source `e8f8827b`; report SHA256
`8047b67694dd64171d2c0d13a9eaff53b18cd833b8fdc45ba054185c382848cc`.
It establishes exact packet-derived conditioning and isolation under that tested
profile, not fresh generated crops, masks, native RGB or composites, sustained
reuse or timing. The job, claim and wait are closed and its terminal event was
acknowledged. Golden qualification closed at150 CPU-s; the15 original reference files are
read-only. Later native inventory and CPU identity/runtime checks passed their
specific gates. CPU acceptance `616abd67…` independently binds11 large assets,
five protected identities,91 AA source/config files, the exact generated-manifest
exception outside source grants,160 SAM hashes and the two isolated runtimes.
Historical PGIDs were not recorded; known child PIDs and current matching task
processes were absent. This does not establish full cached-environment equivalence
or any fresh pixel, mask or perceptual-quality result.

The approved sole replacement `20261101T151030Z-df638d2f` stopped and is closed. Its exact packet worker passed, then CUDA initialization returned `cudaGetDeviceCount` error 2/out-of-memory and CUDA was unavailable before model construction, kernel probe or sampling. The underlying cause is unknown; do not infer a physical-memory OOM, quality negative or 1/3-step cost slope. All task-owned PIDs/process groups, claims and wait are closed and events acknowledged; no live job, reservation or timer remains. No GPU retry or 30-step run is admitted.

The root-accepted phase charge is ≤292 CPU-s /188 GPU-s, including 64 CPU-s preallocation, 188 s one-core campaign wall, 20 CPU-s audit and 20 CPU-s monitoring. Original W8 cumulative is **≤2,457/3,600 CPU-s and 540/3,600 GPU-s**, leaving at least 1,143 CPU-s and 3,060 GPU-s in the accounting envelope. Historical administrative-inspection CPU telemetry uncertainty remains disclosed in W8; these are not invented aggregate measurements. A CUDA-only diagnostic is a future proposal, not authorized.

Overleaf main **`eb7e3a3f67a3af7ce3e0da20387f6f138a66648f`** is pushed. The user-selected PRESLEY anonymous-review profile is implemented: `acmart[manuscript,screen,review,anonymous,natbib=false]`, ACM BibLaTeX `acmnumeric`/`sorting=none` and matching float spacing. The master/named copy retains all four confirmed authors, AAU and ATHENA; the anonymous review PDF hides the author block, funding and self-identifying URLs while retaining AI research methods. Names in cited works remain normally visible; this is a targeted check, not a guarantee that every possible identity inference is prevented. The actual live TOMM portal requirements have not been independently certified.

The frozen package is `/private/tmp/pointstream-submission-20261101-eb7e3a3/`. Both the anonymous review PDF and named author copy have **22 body/reference + 3 appendix pages**. Anonymous PDF SHA256 `c40150cc7801873ca7462e9bd7ef966d263865559761051d79e8de625507d2e6`; named PDF SHA256 `fe24d4958c5fd648d37eeb621edf7216fb2aa216a3e7b8d2434e911535a73e27`; source ZIP `PointStream-source-AUTHOR-EDITOR-ONLY.zip` SHA256 `84fc7723b6deba5a335c601b04fe88d7a350e3176373fa1abfb827ca209c074b` contains identifying master source and is for authors/editors only. Manifest SHA256 `7f1d8c01b332ef79591934cd62d5702dce436dd11a0111b5823676972efa5e70`. Native Biber 2.22 fresh source builds matched both PDF texts; the source archive has 16 active files, no BBL/BST or vendored third-party styles, and declares external TeX dependencies. Checker (26 keys), five registry tests and both-variant visibility checks passed. `b910` was a valid genuine-BibTeX intermediate, now superseded by the chosen profile; the old `f68` style is historical and unsuitable.

The user states the paper is under review nowhere and the coauthors have **not reviewed** it. Remaining author step: coauthor review and approval of the exact package, then an explicit instruction before filing. Source-specific tennis permissions remain unknown and disclosed; no legal clearance is inferred. No paper has been filed. The cover letter remains a draft. Code branch merged latest main `3d93d6e9` including MoQSplat PR #327, with no new paper result; the fixed `48a95400f307999ebb1ca7a80c7abf3d6940beaa` evidence pin remains valid. [PR #326](https://github.com/emanuele-artioli/PointStream/pull/326) will receive a final documentation commit; consult the live PR for checks/merge status rather than assuming a head or result.

The user-selected review format is implemented; it should not be posed again as an author preference question. The paper's bounded evidence limitations remain visible. Coauthor review/approval and explicit filing instruction are the remaining author steps; the paper has not been submitted.

W7's inconclusive unseen-match rule, Med omissions, whole-frame losses, correlation/overlap, modeled reuse, temporal artifacts and incomplete fresh decoding remain visible limitations. Matched ablations, generative peers, independent all-frame labels and operational claims are unqualified where evidence is missing. W6b, W9 live adaptation, W10, W11, broad historical sweeps and tool migration remain deferred or separately gated; none is silently promoted or automatically funded.

**Technical manuscript preparation is complete for the bounded empirical scope. Filing awaits coauthor review/approval and explicit authority to submit.** No paper has been submitted; no new compute is authorized.
