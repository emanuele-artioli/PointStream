# PointStream continuation — 2026-11-05

User authority: complete submission-relevant PLAN autonomously; no PRESLEY edits; no submission. Staged20 GPUh/24 workloadCPUh approved, initial2/3h; no training or legacy200h study. Ledger: `docs/experiments/submission-research-budget-20261104.json`, inclusive conservative ceilings; initial sums must filter release_stage on both charges and reservations. Old allocations closed.

## Active batch

- `20261105T020925Z-97a4cf02`: fresh coalesced-source checkpoint prepare16/48, A6000, source59844fa9daa863ff64f09e16e02a564454fa3c4598288cbdcd3dbc65aed24423, canonicalspec93a9923c782869e5cfc73cbe58cc8831c422f5fe2b1527d66ed72af1e4f73c85. Frozen f7695b70 checkout `/private/tmp/pointstream-coalesced-frozen-20261105`. Ceil700GPU900CPU; smoke200/full420/campaign650; deadline02:48:24UTC. Do not resubmit.
- `20261105T021035Z-04e27283`: new full-wire Matroska transport-only oracle, no neural scoring, inclusive0GPU100CPU; source02720720ea4149becefc4dc11ce9aac08aaa5cc6af4bdc310eeb25f8c408424c, canonicalspec0d758463bbdfef579daff52fd472445a69c76ea22fb92a6666460627f63c8d8b. Old metadata-preserving remux remains failed.
- One batch wait handle94136, result `/private/tmp/pointstream-coalesced-mkv-wait-20261105.json`. Native completion wakeups unverified; no recurring/one-shot automation created. Check existing jobs before any action, canonicalstatus first; no replay of uncertain submission.

## Accepted recent results

- d8b5c76b real16/48 sixpoint calibration complete; exact old16parity; fixed/marginal138.7221s+4.19043/frame. Full48 archivef0dd118a5d64ca148be675a097606aa6dacb8ae49ffcac639ad30911ae74a727. Naiveold188gate failed and stays failed.
- cc450baf new bounded16/188 confirmation complete, audited c6a1f88e; all six188points,2256metricrows4512judgerows; fullarchive622fd68d5380b97449a09872bc15402117dd9c3e7a2576a757cd6aff8fd090d7. Source174/188WASB183/188BallTrack; streak160–177/188 and182–183 vsbaseline1–127 and2–153 respectively. No absentframes, no matchedrate/corpusclaim. Full2500GPU2800CPU ceilingcharged.
- 30f6379a W6 smoothing239differences: flicker28.63655→27.11929, playerLPIPS.346167→.341986,DISTS.252420→.250956. Descriptiveonewindowrulepasses, originaltemporalvsSVTruleallfails. Full303GPU1500CPU charged.
- 7f12c20a repaired checkpoint prepare16/48 exactd8b5 native/wire/source/reconstruction, unscoredhonestly; auditretained. Full700GPU900CPUcharged. Firstd72ef178 failedordinaryvalidatoronunscoreddata700/900charged. To preserve strictwhole-source identity newcoalescedsource prepare above replaces six UNLAUNCHED perpoint jobs. Newprepare700/900+all6score700/900active; cumulative2800GPU3600CPU<prior4100/5250 cap.
- OpenTT test2 source-only a4067c30 judgefailed0/85visible,0/6false,149unknown. No crosssportcoding admitted. Zerooffsetjoin empirically checked3framesonly. gate.smoke_seconds262.773includesstagingetc; not proof of210s command overrun, docs corrected, immutableoldreceipt retains mistake explicitly disclosed.
- GVC26df2a36 nativefeasibility stopped beforemodel because isolatedCUDAerror2; genericCUDAcontrolpassed. No GVCencode/qualityresult. Full900GPU1200CPUcharged. NativeC++entropyCPU-onlyroundtripvalid earlier.
- W8factorial372411b3 and stracefe074da8 linkcurrentCUDAfailuretooldreadpolicy, NNP-onlypassed; no individualcausalityclaim. Fleetfactorialvalidatorfailedremovedscratch, retainedmeasurements independentlyaudited, no relabel.
- W8c33fa60e fixed12publicsystemleaves tinyCUDApasses with namespaces and strictABI4mutationmask32767, oldpolicyfails. 10protectedread denials/7safe mutationdenials andpositiveoutputwrite verified. Audit5db207d1, cleanupsdd977134. No actualgeneratorqualified. Full150GPU200CPU charged. W8strictactual1/3stepcostpilot700GPU1000CPU prospectivereservation; source/CPUreview required beforelaunch.

## Next gates

Coalesced scorer/modules/tests f7695b70 independentlyreviewed. Need fresh prepare actualpublished archive+checkpoint hashes, then one sixpointshared-model score16/48 exactreferenceparity and actualcost. Full-study gate/evidence helpers independentlyrecheckfleetprovenance, actual24metric/judge events, allguardsactive/sourcevisible, sixpointlineage; NOT a full30 release yet. Full30fixedinventory30clips5675frames(197absent6unknown), retrospectivepriorselection, historicalthreefailuresretained. A6000only; Adabytesdiffer (17B) fromA6000, no own-classjointqual yet. Firstlargercostconfirmations292/586frames then frozen30withoutqualityselection.

Ball sensitivitycontrol fivearms native/streak/disk/wrongx+64/white255, frozencontract18c88b...,1200GPU1500CPU reservationconditionalafter188. Agent revising naive16→188costgate to real16/48fixed+marginalcostpilot; nojobsubmitted.

W7Matroskaoracle ifpasses is newtransport sensitivity, not originalanchorfamilycoverage: need allrelevant53physicalratesrerated, pixel/conversionidentity or rescoring plus fournewpoints+controls; completecostreview first.

## Repositories and paper

Codebranch codex/submission-benchmark-2026-11-04. PR333 MERGED afterCI41952060509success. Newrootcommits54780,f7695b70,977aa627 neednewPR forpostmergework. FullRuffpasses; localmypy7priorcv2Noneerrors5unrelatedfiles. Fullpytest localrunning34202 logfile `/private/tmp/pointstream-pytest-full-20261105.log`; CIprevious1645passed31skipped. Preserveuncommittedagentwork.

Manuscript `/private/tmp/pointstream-w7-manuscript`, Overleaf main d40b791 pushed. Bothformats28pages:body1–21refs22–23appendix24–28;8main+2appendixfigures. Mainballqualitative source/DCVC/disk/streak retained. Nativefontsrestored, nofont/margintricks. LatestPDFs `/private/tmp/pointstream-opentt-native-anonymous-20261105.pdf` (07725e5b...) andauthor4bcb3a2d.... OpenTTnegativeappendix andW6figurepublished; new188/30/strictW8notyetintegrated. No newfinalZIPyet; coauthorshavenotreviewed, tennisYouTubepermissionsunresolved, no submissionauthorization.
