# K5d: saved 188-frame failure transitions

Status: ROOT releases one CPU-only descriptive analysis, 60 CPU-seconds / GPU0,
inside the unchanged programme/Ball ceilings. No model, SSH, decoding or input
repair. This is an existing-evidence mechanism diagnostic, not new-clip credit.

Question: does streak rescue compression-induced misses on originally detectable
frames, and does it harm frames where the native baseline succeeds? Competing
explanation: the aggregate recall difference mainly reflects source-undetectable
frames or repeats existing aggregate results without explanatory value.

Use original job20261105T003756Z-cc450baf, b26e5b43/exclusiveA6000; accepted
publication SHA622fd68d5380b97449a09872bc15402117dd9c3e7a2576a757cd6aff8fd090d7
at `/Users/manu/.cache/pointstream-evidence/legacy188-20261110-a1/full/published.tar`.
Independently join Label.csv SHA247c81967bfb05b53f2b65c0a691a7f14c98802ca93c95a68298b2244b5ba72f
from `/Users/manu/.cache/pointstream-evidence/legacy188-label-method-20261110-a1/Label.csv`.
All188 labels are visible. Preserve calibration/selection exposure and the
historical10/197 negative-calibration failure; absence is not identifiable here.
WASB game10 is author-reported test; optimizer/checkpoint logs are unaudited.

Before viewing transitions: include all188 frames/all6points/botharms/bothjudges.
WASB10px is primary,5px secondary. For each point/judge/threshold, report bothhit,
baseline-only(hit harmed), streak-only(rescued), neither; stratify by that judge's
source hit/miss at the same threshold. WASB is primary; BallTrack is diagnostic
because it is also the encoder tracker. Report all denominators, point-level
guard-off state, serialized native/PSBL bytes and paired full/crop metric deltas.
Do not invent frame-level guard state, matched-rate claims, independent-frame
CIs or selected examples. Save every frame-indexed row plus compact tables.

Complete CPU envelope:10input/label/pins +15implementation/synthetic transition
checks +10analysis/writeout +10independent arithmetic review +15failure/admin
reserve =60. One single-thread analysis process/no children/wall180s. Code can be
retained with new outputs under `/private/tmp`; completed inputs stay read-only.
Stop on missing/inconsistent hashes, counts, joins, predictions, labels or cap.
No repair/fetch/model rerun. If redundant with aggregates, record negative novelty
and close without expansion. Source qualification and all current GPU jobs stay
separate. Root independently reviews outputs before paper use.

## Root-reviewed schema correction

The first analysis stopped before transitions: a raw TrackNet visibility2 was
incorrectly asserted equal to normalized kind1. Existing `tracknet_labels()`
maps raw1/2 to visible kind1,0 to absent0, other to excluded-1. Root independently
checked that exact reader. Original archive/label/audit pins passed; inputs are
unchanged. Preserve the failed a1 trace (0.073941CPU/nochildren). Authorize ONE
corrected create-only a2 run using that established normalization and retaining
raw visibility as a separate column, with no new outcome-based strata. Cumulative
implementation/run cap remains35CPU (a2 at most34), rootreview10/reserve15 remain;
the already charged60CPU ceiling is unchanged. Any further inconsistency stops.

## Accepted result

A2 passed; root independently recomputed all4512 prediction/distance/transition/
byte joins and48 table counts from the original archive. WASB10px source-hit
stratum174/source-miss14: rescue/harm are DCVC0 169/0 and7/0; DCVC21 140/0
and6/0; DCVC42 41/1 and3/1; SVT48 43/0 and0/0; SVT55 98/1 and1/0;
SVT62 146/0 and7/0. Most repairs occur where the source judge could detect the
ball, while occasional harms remain visible. This adds a retrospective mechanism
diagnostic; it does not establish universal per-frame improvement. All six
point-level guard-off flags are false. Paired bytes/full/crop deltas and all
secondary tables remain retained. No absence, new-clip, matched-rate or current-
source claim. A1+A2 analysis CPU0.208592/nochildren; full60 charge retained.
Root review used0.101075CPU on its successful receipt, plus an earlier identical
check whose receipt write to the read-only output directory was denied. Inputs
and completed outputs stayed unchanged. Bounded durable archive preparation is
using the original15CPU failure/admin reserve, not a new allocation.

Durablearchive publication79394 passed/rootphysicallyaccepted:265106B/SHA5c181ba8, file0444/dir0555/readback. Fullselectedanalysis/code/CSV/rootreceipt/a1failure is at`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/checks/k5d-legacy188-transitions-5c181ba8d5b1397addf6eb06/analysis.tar.gz`. Publication remote.038103/local.066909/SSH.017070CPU,4.265402wall. Full60CPU charge retained; GPU0. This subitem is CLOSED with the scoped mechanism diagnostic and explicit occasional harms, not new confirmation.
