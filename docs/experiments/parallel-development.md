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

### 2026-10-28 — Independent SVT completion published within baseline cap

SVT-only **20261028T150051Z-445e051d**, PENDINGreceipt, code0c1381c4,
source3d4aa20888aea024376980f5b9076b936f65810e6412347b20150ff856da9fd5,
normalizedspecdd940868446b332ea36f55add409012f46584762dc17ff25d4633c0a56680ae8.
Exactlyone newindependent request, cap1450; prior250charge+1450+100buffer=1800.
BothA6000hosts re-inspected, no PSclaims, sufficientmemory. No DCVCretry.
Receipt /private/tmp/parallel-roi-submission.json. Originalthree requests all
terminal; events delivered/acknowledged. Separate same-batch observation for
this alreadyapproved unfinished comparator must retain original15:29:20Z
observation bound. All request absolute deadlines remain16:00Z.

Camera-motion version from approvedTask1 is being examined CPU-first, not yet
submitted. Prototype uses2000ORBfeatures,RANSAC2px,>=20inliers,medianfiterror
<=1.5px, invalidfits identityfallback; compressed float32 homographies are paid
and decoded before encoder residual construction and decoder reconstruction.
Nativeplane warp uses linearinterpolation and explicit16/128/128borders;
reference/residual/geometry/header/oldside allcharged. No dynamic reference
selection or training. Sourcecamera characterization over all80frozen ticks is
the immediate cheap check; nativeproduction smoke would still be needed before
any newquality result. No new GPU request or allocation yet.

CPU characterization found all80/80fits valid: median24.1474half-resolution
pixels of camera displacement, p95=50.5314,max55.5433. Thus the static pilot
does not exercise the approved camera-compensation hypothesis. CPU runtime
2.397s, no neural quality measured; source-ticks from originalSHA56f54ccd,
uncoded reference for this characterization only. Production uses the paid
decodedCRF35reference and serializeddecodedfloat32geometry. Unit checks verify
identity, luma/chroma translation, borders, packetroundtrip and singular/cohort
rejection; all21focused/docchecks, Ruff andmypy266files pass.

Complete this originally approvedTask1camera-motion configuration under its
UNCHANGED2700cap: staticpilot500charged + newmotioncap2000 +200unallocatedbuffer.
No staticidentityretry, newrenderer, training or transferredbudget. Three CRFs
45/55/63/reference35 remain unchanged. Motion spec motion.json:
smoke300/full1300/validation80/overhead320. Require>=22/24validsmokefits,
substantivepaidgeometry/native/reconstruction/source/neuralcontrols, memory
and35%costprojection beforefull. If motion smoke fails, stop this configuration.
No newrequest submitted yet; preserve all completed static/pixel/DCVC evidence.

### 2026-10-28 — Motion published; SVT full blocked by projected cost

Motion **20261028T152201Z-244ab9ae**, PENDINGreceipt. Executedsource31421e6e,
archive1d46ff03cd2cfa325bc13dbe7be4f08a1eabc1aef5bb25389837cb4c47c139de,
normalizedspec40d25b21fc644e13e2137931e12cd4c2ad2e0196a86a82e33fe5f8b8734b34ac.
Receipt /private/tmp/parallel-motion-submission.json. One new camera-motion
configuration under originalTask1cap, not an identityretry. No oldjob replay.

SVT-only445e051d FAILED COST GATE: smokecommand112.474733s succeeds, every
correctness/weights/source/neuralcontrol/nativebyte/memory check passes, but
projectedfull1195.292118s exceeds its900s full allowance. No full run launched;
smoke scores are not accepted quality evidence. This is an absolute budget
projection, not a timing-variation gate. No further SVT request under this
configuration. A future CPU-encoding/GPU-scoring split could avoid GPU idle
time, but is not executed or claimed. Preserve native smoke outputs/metadata.

Original three-job observation ended allterminal; its events acknowledged.
Independent SVT also terminal. Motion is a new bounded stage observation
(<=2000s and absolute16:00Z), within the per-stage45minute recovery rule; no
indefinite/recurring timer and no extension of any runningrequestdeadline.
Current source branch31421e6e; all old archivedsource identities remain immutable.

### 2026-10-28 — All terminal, independently audited, batch closed

Motion244ab9ae COMPLETE before the later usage interruption. Humanresume only
inspected sameID, no replay. All240camera fits pass, geometry7581paidbytes,
all80Al scores; old source/neural controls exact0. Full310.295421s,
projection533.132041<1300; allocated3082.722168/reserved3740MiB<5000.
Paid rates206.052129/108.182962/46.733796kbps; LPIPS
0.135084/0.158170/0.179777, DISTS0.078526/0.090225/0.100966.
Fullraw440320bytes SHA337421ee4597ed028d47336e3fb30648c05607581662a609d29b4ce75787e68e.
Nativeadmitted454.451627s,throughcanonicalclosure490.483598s,charge500.
All motion points outside originalAlratecoverage. Higherpointnotacceptedwin;
two cheaperpointsLPIPSdominatedby33.685kbpsSVT. Post-hoc protocolPCHIP on
overlappingquality givesLPIPS BD-rate+230.284898%, robust+224.34to+236.28under
candidate-distance+/-0.001; DISTSnoqualityoverlap. Pointwisegate remains
unresolvedoutsidecoverage; post-hoccurves do not upgrade it or establishDCVCwin.

Pixel full nativeadmitted899.894516s,throughclosure935.706388s,charge950.
Bothwindowsfull80, native/source/historicalscorecontrolswithin1e-4.
Al236.074to1183.439kbps; Med357.325to1852.233kbps. Noqualityoverlap with old
Alanchor; noBD-rate or equal-ratewin inferred. ROIstopthroughclosure185.398313s,
charge200, rawsmoke276480bytes SHA36de384e6cfc543ef5dc37b2a3cf843c58007121c4b0a1a13f4afce58e0b2919.
All partialsmokemetrics remain unacceptedfullquality. DCVC no retries.

Independent CPUaudit ../parallel-development-audit-2026-10-28.json PASS for all
three acceptedfullpublications: fullmeans/rawarrays/controlcohort, camera and
rectangle geometry, nativeframing, completepaidbytes and savedpixelsupport.
No neuralmetricrecomputation claimed. ../parallel-development-curves-2026-10-28.json
contains post-hocnegative/overlap-qualifiedBDresults and sensitivities; report
../parallel-development-report-2026-10-28.md explains limitations and exact jobs.

Budget CLOSED: background1000/2700, pixels950/2700, baselines450/1800;
newtotal2400/7200GPU-s,4800unused. DedicatedCPUanalysis conservatively120/3600s.
Prior2980/3600study untouched. No furtherrequest, budgettransfer or modeltimer.
All10own supervisor/campaign PIDs absent on their exacthosts; noPSGPUclaims
on either qualifiedhost. Every completion/decision/contention event delivered
and acknowledged. Existingworkers/foreignprocesses untouched. Sourcearchives
byte-reconstructed fromGit match211b4e69/3d4aa208/1d46ff03 exactly; temporary
transfer tars are normally removed after successful extraction, not lostevidence.
PR270mergedCIpassing,adeab13cb713995c7f7c4d05f55138618c4c0b6a.
Result/evidence publication follows; no new quality result or compute pending.

Preservation COMPLETE: newexclusive
/home/itec/emanuele/Datasets/pointstream-data/tennis/parallel-development-20261028-244ab9ae/evidence.tar,
SHA71900d0da6b7076b1ba4978da3663a1f5ec8a3f710f503cfa26fa2b524e20d5e,
100720640bytes/105members. Wholearchive and everymember identity verified at
destination; archive and all5canonicalpublished/partialpayloads protected0444.
The first upload wrote completebytes, but verification used an unavailable
Python3.10hashlib.file_digest API. A CPU-only compatible verification read the
SAMEfile in place; no retransmission/overwrite/newdirectory/newcompute.
All oldDatasets/Models/sourceinputs preserved. Byte-identical replay of archived
portableauditor PASS against all three acceptedpublications. Ruff/fullmypy268
files and22focused/docchecks PASS. FinalresultPR/CI publication pending.
