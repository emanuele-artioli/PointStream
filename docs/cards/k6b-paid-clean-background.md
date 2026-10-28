# Paid clean-background oracle — accepted 2026-10-28

Goal: test whether removing players BEFORE coding the background creates enough
whole-frame headroom to pay the existing player side information. The previous
base coded the original full frame, so it still paid bits for players that the
overlay later replaced. Appearance alone failed; simply wrapping pose bytes
and freeing one group of side costs also fail the saved-score constraints.

Hypothesis: encoder-side foreground removal reduces base bytes sufficiently
without unacceptable damage outside the replaced region. Competing explanation:
players are too small, and interpolation/fill artifacts consume the saving.
This is a finite development diagnostic, not a promised codec improvement.

## Frozen proposal

- Same Al031@129, 240 native source frames and the same80 scored ticks0:240:3.
  Same original source identity, saved R, saved generated crops, saved receiver
  support, metric weights/definitions and scale4/CRF63/preset4 native SVT path.
- Replace the union of saved source R with OpenCV Telea radius3 BEFORE encoding
  the base. Freeze color conversion/order from the native source pipeline during
  CPU preflight and record it before any new scoring. No search over fills,
  masks, scales, codec settings or generators. Encoder R is an explicit oracle
  grant; only the encoded cleaned base reaches the receiver. No free target
  background is used at reconstruction. Source access and masks are not new
  decoder inputs.
- Independently count the new complete240-frame IVF and retain every original
  side term5.607129kbps, including existing KPS, amortized reference/adapter and
  bootstrap. Account any changed bootstrap bytes; never subtract costs to pass.
  Previously unmeasured delivery terms remain explicit optimistic limitations.
- Test two new global policies: the frozen actual overlay/support over the
  new paid base; source pixels inside saved R over that same new paid base.
  Include the actual arm because perceptual metrics are not monotone. Neither
  policy is a mathematical bound on arbitrary future reconstructions. The
  source-pixel policy is explicitly an oracle, not a delivered codec.
- In the SAME scoring request, retain original paid-base/actual overlays as
  two exact historical per-frame controls (four policies total). No training,
  generation, SAM inference, ball updates or untouched-footage claims.

## Decision and stop

At new base rate plus unchanged side costs, require ONE global new policy to
beat BOTH full LPIPS and DISTS by >0.001 against the original achievable SVT
envelope. No mixing policies by metric, no extrapolation, no BD-rate from one
point. Outside measured anchor coverage means unresolved, not a win. Failure
closes only this fixed cleaned-background construction. Success establishes
local headroom; fair pixel ROI, DCVC, full serialized rate and fresh evaluation
are still needed before any paper claim. No automatic renderer funding.

Before GPU submission, CPU preparation must verify all original inputs/native
paths, source conversion identities, mask coverage, complete stream decode and
240 transport frames. Inputs/output streams are separately archived and hashed.
If original source/masks/native tools cannot be recovered exactly, stop without
GPU use. CPU preflight/encoding is one bounded stage <=30min; preserve outputs
separately under Datasets and use host-local scratch through a CPU fleet job.
No new model environment or dependencies.

Proposed fresh GPU request: **at most800 of the870 remaining GPU-seconds**:
smoke120, full350, validation80, overhead250. Four-policy historical full work
took144.14s; new base conversion adds work, so that is only a planning reference.
The representative smoke must use real1080p metrics and native GPU RGB conversion
and project total work within350s full and800s overall. Timing variation stays
secondary; correctness, slow-sample cost projection, CUDA,8000MiB reservation,
hash identities and hard time caps remain binding. Same qualified A6000 pool,
inspect existing jobs/eligible hosts first, new frozen source/spec/input hashes.
Every wait <=45min, one native completion wait, no recurring LLM monitor.

Exactly one new GPU request after explicit acceptance and successful CPU
preflight; no retry after failed correctness/cost/scientific gate without a new
decision. This would leave at least70s of the original3600s ceiling. If preflight
finds the quote cannot fit, report that before submitting. The prior study's
unused870s is not an automatic grant to this different method.

Status: human accepted “Approve the paid-background oracle” on2026-10-28.
CPU preparation and at most ONE <=800s GPU request are authorized after the
stated checks. Nothing has been submitted yet; original spend2730/3600s.

Preflight choice, fixed BEFORE new encoding/scoring: apply Telea radius3 to
native uint8 Y/U/V planes independently, using the original R union projected
by nearest-neighbor onto each plane. Fill at source resolution BEFORE the
unchanged scale4 area-downsample. This introduces no RGB round trip or change
to source samples outside the corresponding plane mask. Native source YUV hash
must match the original. Source and new decoded-base RGB at judging time use
the original CUDA conversion, followed by original bicubic base upsampling.
No alternate fill/color order after examining results.
Branch `codex/paid-clean-background`. Parent report:
[joint side-rate feasibility](../k6b-side-feasibility-report-2026-10-28.md).


## Execution checkpoint

CPU-only20261028T091736Z-bf9f97d2 COMPLETE and independently audited. Clean
base19659bytes vs original20557; total21.989628889kbps with all existing side
costs unchanged. Native source and per-frame controls pass preparation.
The ONE GPU request20261028T100153Z-27f39d64 had an extraction acknowledgement
timeout before publication. All214source files were verified exact and the
original extractor confirmed gone; FIRST publication recovered under this SAME
ID, no resubmission or extraction replay. Source10a1ae76/archive7bc3be42/native
identity82eee1d0. Usage review failure executed no action; after human resume,
fresh inspection proved no readiness/admission, and only the expired absolute
queue deadline was refreshed before first publication. New normalizedspec
e1e7b877, deadline2026-10-28T13:01:15Z. All inputs, policies, thresholds and
800s cap unchanged. Publication confirmed PENDING; no new quality accepted yet.
