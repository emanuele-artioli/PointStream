# Parallel development batch — 2026-10-28

Human authorized the concrete batch in
../cards/parallel-development-2026-10-28.md, followed by “Go ahead with these
parallel tasks.” Three GPU task caps total7200seconds, independent of the
closed previous2980/3600study. CPU gap/accounting cap3600CPUseconds.
No training/generation and no terminal-job replay. No recurring model timers.

## Frozen hypotheses and competing explanations

- Static paid court reference plus half-resolution coded residuals may retain
  more background detail per bit than the old quarter-resolution base.
  Competing explanation: camera/background motion makes residuals too costly.
  Identity motion is explicitly serialized; this is not a panorama bound.
  Al only initially, three residual CRFs45/55/63, referenceCRF35,preset4.
  Original actual overlay and its full5.607128889kbps side retained.
- Independently coded rectangular player tiles may preserve appearance with
  fewer bytes than pose/reference/adapter delivery. Competing explanation:
  geometry and large source-mask outliers make the crop streams expensive.
  Frozen full-window max tile sizes, CRFs32/45/58,preset4, quarter-resolution
  baseCRF63; all geometry and complete streams charged. No secret decoder mask.
  CPU preflight found Al tiles1216x684/1146x676 and Med600x522/456x434;
  retain them for this explicitly limited fixed-rectangle test, do not retune
  after quality outcomes. Adaptive resizing would be a different construction.
- The required native DCVC-UF and SVT ROI comparators may erase any apparent
  split advantage. DCVC points(QP,scale)=(0,4),(21,2),(42,1); SVT ROI at
  scale4CRF43/53/63,preset4,+32quantizeroffset outsideRblocks. Source R is
  encoder-only; no free decoder foreground. Native limited-range YUV path.

All codecs process240nativeframes; metrics on original80ticks0:240:3 at1080p,
whole-frame LPIPS/DISTS. Original source hashes and saved neural controls must
match. Timing variation secondary; hard correctness/memory/cost/deadlines stay.
Only one same-policy point winning both metrics by>0.001 at its fully paidrate
is promising; no extrapolation or combining policies. Missing anchor overlap
is UNRESOLVED, not a win. No untouched or full-codec submission claim.

## Preflight and resources

Before submission, original job20261028T100153Z-27f39d64 was re-read from
k6b-players.md and canonical status confirmed COMPLETE. No replay.
All six hosts probed; all reachable. gpu2 has twoA6000s with13397MiBfree each;
gpu3A6000has23183MiBfree. Each job declares5000MiBplus8192sharingmargin,
compatiblepoolgpu2/gpu3. All worker heartbeats fresh. Other GPU classes are
not substituted. gpu4 busy; other users' jobs untouched.

Task1 cap2700: smoke350/full1700/validator100/overhead550.
Task2 cap2700: smoke350/full1800/validator100/overhead450.
Task3 cap1800: smoke450/full1100/validator60/overhead190.
Each stage and total reservation fit45minutes. Staging and publication included
in caps. Work stops on smoke failure; no automatic retry/newrequest. Actual
charged time comes from native admission/terminal intervals, rounded upward.
No request submitted yet; record each allocated ID immediately below.

## CPU task — saved evidence analysis

Reproducible output ../parallel-gap-2026-10-28.json from
experiments.tennis.parallel_gap, source hash recorded in output. Both raw
publication SHA256s verified, all80old neural score controls exactly equal.
Runtime0.34seconds on local CPU; conservative task charge60CPUseconds includes
the bounded geometry/accounting inspection, zeroGPUtime.

At fixed cleaned pixels, robust total-side equality ceiling is1.619035kbps
for actual overlay and2.218932kbps for original-in-R; current side5.607129.
At currenttotal21.989629kbps, actual needs LPIPS reduction>0.03450747 and
DISTS>0.00148768 for strict robust wins; source-in-R needs LPIPS>0.02931678.
These are finite fixed-pixel accounting constraints, not universal bounds.
Confirmation split requirements recorded; no verified untouched source manifest
is claimed. It needs source inventory and training/judge-exposure audit before
confirmation; this development batch spends no compute on unseen confirmation.

## Dispatch and results

Pending source freeze and local checks. No new GPU spend yet.
