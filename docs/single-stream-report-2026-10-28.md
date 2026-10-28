# Single-stream search: no whole-frame win; native reproduction passes locally

All four requests are terminal. Temporal restoration completed; conditional
coding sources were audited; the supported24-frame DCVC reproduction passed.
Latent refinement was not launched because native optimized-latent transport
has not been established. No training, model download or larger run followed.

## Restoration result

Job20261028T191318Z-b4727331, gpu2/A6000 UUID
GPU-40e9a544-b3d6-e5f3-a736-7020a6401bfc, source49ee1b931c7771876d5dd5aa55f7f2a96f8d1bcb.
Frozen Al031@129 prefix96frames/25fps; score32ticks at1080p. Three saved native
SVT streams43/53/63 at960x544, preset4, with exactly matching source pixels,
anchor image hashes and unfiltered LPIPS/DISTS controls (maximum error0).
These are familiar development frames, not untouched confirmation.

The decoder estimates DIS flow from decoded neighbouring frames only, aligns
previous/next frames, rejects inconsistent/photometrically changed pixels and
blends conservatively. Three strengths were fixed before outcomes. Every
candidate pays the complete original IVF plus an8-byte decoder profile;
there are no masks, poses, models, references or flow packets to discount.
One-frame lookahead; no live latency/performance claim.

PCHIP BD-rate against the restricted unfiltered SVT envelope, overlapping
quality only; positive means more bits at equal quality:

| Strength | Whole LPIPS | Whole DISTS | Foreground LPIPS | Foreground DISTS | Background LPIPS | Background DISTS |
|---|---:|---:|---:|---:|---:|---:|
|0.15|+0.531%|+0.297%|-0.023%|-0.028%|+0.548%|+0.318%|
|0.35|+2.287%|+1.555%|-0.114%|-0.051%|+2.361%|+1.692%|
|0.60|+4.892%|+4.141%|-0.653%|-0.044%|+5.065%|+4.499%|

No robust two-metric whole-frame point wins at the predeclared0.001 margin.
Every filtered point is slightly dearer and has worse whole-frame means than
its own unfiltered encode. The weakest changes are inside that margin; the
strongest causes robust whole-frame losses. Foreground has small curve gains
and two bracketed LPIPS-only robust cells; neither wins both region metrics.
Highest-rate cells outside the anchor bracket remain unresolved, not extrapolated.
The8-byte profile adds only0.016667kbps: removing it cannot repair the quality
regression. Increasing strength monotonically worsens background curves.

Sampled temporal residual distortion (stride3nativeframes) improves slightly
on foreground and worsens slightly on background/whole frame; this is not a
full experience-track or human preference result. No new semantic-partition
advantage is claimed: the tested filter is generic codec postprocessing.

Smoke promotion passes, projection948.070<1200s. Full command503.666s,
internal498.829s; peak allocated3082.722/reserved3740MiB. Source native YUV
and all32CUDA source RGB hashes match; independent native decodes agree.
[Independent audit](single-restoration-audit-2026-10-28.json) checks bytes,
profiles, IVF framing, saved means and image hashes, not a new neural rescore.
Full publicationSHA8502056b82256105551a062144b39fcf2077eff05f46da60365af47765d5b890.

This rejects this simple confidence-gated convex fusion on this cohort. It
does not bound learned temporal restoration, other domains or general models.
Further strength tuning here is not justified by these monotonic results.

## Native decoder diagnosis

Initial20261028T190838Z-3dea4d74, gpu2/A6000
GPU-fc7e8fff-75be-4984-a185-fe8ce63edcd9, sourceafe86f57620d85161aba5a8dc60e17c8e820fb4e:
the legacy4-frame arm passes. The proposed default-stream alternative fails
BEFORE encoding with CUDA graph-capture error at dmc_common.cpp:15. That API
assumption was wrong. There is no default-stream quality/determinism result
and it must not become the production default. Failed native evidence retained.

Complete the unaffected original reproduction inside the SAME600GPU-s diagnosis
cap: charge250 for the initial request; one fresh request capped350, with a
new4-frame smoke and the original24-frame full test. No additional alternative
or changed correctness rule. No further request after this completion.

Supported completion20261028T192649Z-867e69f2, gpu3/A6000
GPU-6f70a27e-4171-22d7-3ffd-647e7f186a49, sourcecc94686fcd8b9f8ae885d0987ebaf0af7999ccee:
QP21,960x544,HT-L,nativeYUV24frames PASSES. Fresh encoding, two decoder
processes with two passes each, all decoded frame hashes equal; first frame
matches encoder reconstruction. Container/report identities and memory pass.
Full command59.573s; smoke projection85.014<130s.

[Saved-pixel audit](single-legacy-audit-2026-10-28.json) independently checks all
four pass arrays against the two saved raw YUV files and container bytes.
No stored stream was decoded again. PublicationSHA
8a9a40c5254570855849283a59b9e3c02a14f813e1272cd37a2fbf58a229c98b.
[Initial audit](single-diagnosis-audit-2026-10-28.json) distinguishes the native
legacy pass from the unsupported alternative.

The original failure was NOT reproduced in this fresh24-frame run. No remedy
was established, no broad reliability claim follows, and old nondeterminism
and cross-device portability limitations remain open. This is not a quality
curve or a contemporary neural-baseline win. Production stream policy unchanged.

## Conditional-code audit and refinement gate

CPU job20261028T190840Z-eb011d4b COMPLETE on gpu5, no GPU allocation. Audited214
installed files plus SHA-pinned primary public source files:
DCVC-DTe3deeda1cd99566498504231f6853fd554ec00e9,
DCMVC4aa95ef4b0176b6076b1ba0396d2cd178258d506.
[Source findings](single-code-findings-2026-10-28.json) preserve exact file hashes
and excerpts; the raw publication includes the original selected sources.

UF has differentiable training forwards. It is incorrect to say gradients do
not exist. Its native compress methods, however, delegate to C++ proxies that
accept pixels and do not expose an externally optimized latent. Training
forward bpp is an estimate, not a delivered optimized stream. DCVC-DT provides
an optimizer for DCVC-DC's different IntraNoAR/DMC/checkpoints, not a drop-in
UF transport; DCMVC is another DCVC-DC-based architecture. They are useful
source starting points, not qualified methods in this batch's frozen runtime.

The1200GPU-second refinement reservation remains UNSPENT: an actual matching
native entropy transport has not been demonstrated. That is a compatibility
gate, not a negative optimization result. The audit does not rule out encoder-
only weight adaptation with a frozen decoder as a separate implementation.

Next concrete engineering options: expose/test native latent injection using
the current UF decoder, or qualify the DCVC-DC-family method with its own
checkpoints/runtime while retaining a contemporary neural anchor. Establish
native/gradient reconstruction parity before tuning region-aware objectives.
Do not report estimated-rate gains or substitute an older weak baseline.

## Budget, checks and preservation

Conservative GPU allocation-wall charges: restoration750, initial diagnosis250,
supported completion300, refinement0: **1300/3600GPU-s**,2300unused. Old ledgers
remain closed. Dedicated CPU source audit489.457allocation-wall seconds with
two threads is charged1000CPU-s; local audit/analysis120: **1120/3600CPU-s**.
This includes staging/publication wait conservatively; source commands themselves
were3.560s(smoke)/0.684s(full). GPU-job CPU support is inside those job budgets.
Shared-GPU timings are cost accounting only. No future compute is implied by
unused balances.

All requests/waits terminal, events acknowledged; no model timer or recurring
monitor. Exact source archives reproduce all three snapshot SHAs byte-for-byte.
Raw publications, source, native metadata and analyses are preserved read-only;
[archive receipt](single-evidence-receipt-2026-10-28.json) records SHA/member
verification. No original dataset/model/checkpoint was deleted or overwritten.

Configured Ruff/mypy and eight focused tests pass. Full macOS suite retains
the two known platform failures; process test passes outside sandbox. Linux
CI is the merge gate. [Implementation PR274](https://github.com/emanuele-artioli/PointStream/pull/274)
merged with passing CI. No complete PointStream baseline win or TOMM readiness.
