# Experiments: parallel development batch

### 2026-10-28 — Approved background, pixel and baseline pilots

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

Source frozen at140b56ea7f54a386cfa607cee099c2ffdeabd9ad. Ruff/fullmypy262files
and13focusedtests PASS. Full macOS suite retains only the two known sandbox/
server-storage platform limitations. PR270 created, auto-merge requested.

Background allocated **20261028T143341Z-c50f6395**. Local record PREPARING,
native submit command still being collected; no receipt/admission confirmed.
Snapshot211b4e6961f02fe8c657112f9401f3548ef674578d1504963a8954025f3ba57b,
normalizedspec821a7a41898c7e128edb42531ddb8ce97df2ed425fba6fd055a78d5fdcfee8bc.
Launch delegate GPT-6 Luna/max; native processhandle82096 belongs to that
delegate. Never resubmit. Root inspecting canonical state before any recovery.
Local receipt /private/tmp/parallel-background-submission.json currently empty.
FIRST publication confirmed, no recovery/retry: background c50f6395 PENDING
receipt; canonical firststatus RUNNING on gpu2, noerror. Exact source archive
211b4e69 matches frozen140b56ea. No recurring monitor (native receipt explicitly
heartbeat_requiredfalse,recurring_pollingfalse). Earlier PREPARING is historical.

Pixel corrections allocated **20261028T143713Z-a3ae0af0**, PENDING receipt;
same exact source211b4e6961f02fe8c657112f9401f3548ef674578d1504963a8954025f3ba57b,
normalizedspecd1ff16e84f06c904234182454ff66196a9288352b3ea78c329e6220ed72f2a1a.
Receipt /private/tmp/parallel-pixels-submission.json. Launch delegate Luna/max,
single submission confirmed. Baseline launch now in progress; no allocatedID
confirmed yet. Task caps unchanged2700/2700/1800, no retries or budgettransfers.
Actual GPU charge pending native terminalintervals; 0 last finalized newcharge.

CI40707362997 caught the new experiment log's required header/dated-entry
format. The full local run had completed before this new documentation file
was added, so it did not cover that file. Corrected to the repository schema;
tests/test_experiments_docs.py3/3PASS. Scientific source/snapshot unchanged.
Pixel canonical firststatus RUNNING on gpu3, noerror; background on gpu2.

Baseline allocated **20261028T143856Z-ae2c81a7**, local PREPARING at inspection;
submissioncommand still being collected by Luna/max delegate. Never resubmit.
Receipt target /private/tmp/parallel-baselines-submission.json. Same frozen
source140b56ea expected; exact native receipt/identity pending. All three
allocations now have IDs; no extra request is authorized to replace an unknown.

Baseline FIRST publication confirmed PENDING, same frozen source/archive
211b4e6961f02fe8c657112f9401f3548ef674578d1504963a8954025f3ba57b,
normalizedspec1614cd363193a8b20da4b1cbdaf2e25841c03460b01fdb6f127390317135ec7d.
All three requests submitted exactly once by Luna/max delegates; no unknown
submission remains. Canonical prior/status checks completed before batch wait.
One bounded attended batch observation begins with all IDs, max2700seconds;
handle/receipts under /private/tmp/parallel-batch-*. No verified automatic model
completion wakeup in this harness; no recurring timer or paid API substitute.

### 2026-10-28 — Two full runs complete; native DCVC correctness stop

Backgroundc50f6395 COMPLETE, all80Al ticks/controlmaxerror0, full331.216329s.
Paid total rates216.607129/125.634629/52.920462kbps, LPIPS
0.143709/0.173960/0.224457 and DISTS0.081906/0.094040/0.110906.
OldSVT33.685kbpsLPIPS0.139336 is better by>0.001 at lower rate for ALLthree;
no joint win possible for these fixed settings. Do not extrapolate the oldhull.
Native admitted436.652432s, throughcanonicalclosure483.641327s; charge500.
Rawfull491520bytes SHA7b8766283f8fd3788850792e9e1795c1f0725f7bffe88a69b9a4e0227cd54916.

Pixelsa3ae0af0 COMPLETE, bothwindows/all80ticks each, full695.731800s;
native metadata/export and independentaudit pending. Rawfull63528960bytes
SHAd462366799275314839b16d19b4d373ed7ceb259ace9ad9a7a20bb97946facee.
No decoder masks/oldrenderer sidecosts; all actual tile/header/base bytes paid.

Baselinesae2c81a7 FAILED in correctness smoke, NOT timing/cost: first native
AlQP0/scale4 passes; QP21/scale2 has two decodes disagreeing24/24frames and
firstframe disagrees with encoder. Streamcontainer hashes/counts match,
peak740.2MiB; no fullmetrics or ROI comparator ran. Preserve nativefailurereports
and partial064e46352bb386e4581a9fc25aaa160bb62aea68557a070f1d681d1b41dc3e9c.
Native admitted134.690518s, throughcanonicalclosure206.330955s; charge250.
This DCVC configuration is closed, no retry. All terminal/shared events in
nativewaits1/2delivered/acknowledged; finalpixel completion event pendingack.

The independent SVT comparison remains authorized and unexecuted. Root will
finish it under the SAME original1800 baselineallocation:250charged plus a
newSVT-only cap1450 plus100unallocatedsafety. This is not a DCVC retry or
budgetincrease. Original ROI settings unchanged; add nativefullSVTCRF63/preset4
as missing upper-ratecoverage, frozen before its outcome. All newmethodpoints
exceedoldAlmax33.685kbps, motivating the stronger comparator. No stronger
method setting selected from these results. New specification roi-only.json,
smoke240/full900/validation60/overhead250; CPUchecks before one newindependent
request. If this smoke fails, stop. Batch source/rules preserve all oldfailure.
