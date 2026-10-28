# Ideal-player appearance bound — stopped at resource stability gate

The separately approved ideal-player diagnostic has **no accepted quality
result**. Repaired request `20261028T073233Z-36f21aa5` completed its three-tick
smoke, reproduced both historical metric controls exactly, then failed the
predeclared timing-stability gate. Full80-tick evaluation never launched.
This is neither HEADROOM nor NO-GO nor a mixed scientific result: the research
question remains unanswered. Do not infer quality from three smoke ticks.

## What passed and what stopped

The same four frozen image policies ran at full1920x1080 on gpu3's qualified
A6000, one image at a time, with the original LPIPS/DISTS weights and heads.
Source/base/actual/oracle hashes matched. On ticks0/3/6, base and actual
LPIPS/DISTS controls were byte-for-byte equal numerically to the previous
scores: maximum absolute error0. CUDA assertions passed; peak allocated
3082.784668MiB and reserved3740MiB were below the8000MiB allocator cap.
No training, generation, SAM rerun or new RGB conversion occurred.

The accepted rule required the slower warm timing divided by the faster warm
timing to be <=1.25 for EVERY policy. Observed complete per-policy traces:

| Policy | Cold seconds | Warm1 seconds | Warm2 seconds | Warm ratio | Rule |
|---|---:|---:|---:|---:|---|
| Paid base | 2.448414 | 0.425357 | 0.428889 | 1.008304 | pass |
| Frozen actual overlay | 0.740940 | 0.415452 | 0.454163 | 1.093178 | pass |
| Source pixels in original R | 0.708303 | 0.237550 | 0.456889 | 1.923337 | FAIL |
| Source pixels in predicted support | 0.763333 | 0.238660 | 0.460465 | 1.929378 | FAIL |

The validator rejected `ideal_R` with `steady timing samples differ by more
than25%`. It exited nonzero before publishing a passing validation record or
cost projection. The second ideal policy independently violates the same rule.
No full run was promoted. This establishes instability under the accepted
contract, **not a measured550s budget exceedance**. Do not relax the rule after
seeing it fail. The job ran under the approved shared-GPU continue policy and
its timing was marked contaminated; these records do not establish the cause
of the variation or predict exclusive-GPU performance.

## Independent CPU audit

A separate CPU-only program independently checked all80frozen input ticks:
400full-frame byte identities, exact original source and paid base, old actual
composite hashes, and160source-pixel oracle reconstructions using `np.where`
on independently decoded original R and saved projected receiver support.
It also compared the saved three-tick score arrays directly to the old full
arrays and independently recomputed the failed timing ratios. All checks pass.
Cost25.405012wall/25.602478CPU-s, CUDA hidden; no new neural scoring or quality
verdict. The originally planned full-result audit was not run because no full
result exists. The complete immutable input archive remains available.

## Provenance and resource closure

- Repaired job `20261028T073233Z-36f21aa5`, code
  `c379966462013177604dc9cff793e97d0d40edd8`.
- Selected source archive
  `0787aba9a89ef3620bdd9a3c4c8ab2f0e65fd55ac9624eeb75b94fbde3460445`;
  spec `ea4f2b02ac19f04f030fb721cc7b0ec4acba2157f3861037583412f61c00b264`.
  Native source tree `5d99fb963277d6159aa012a06cff63502ab217c67d8a32af3bb65dafd803feea`;
  all40files independently match the CPU-checked source archive.
- gpu3 index0 A6000 UUID `GPU-6f70a27e-4171-22d7-3ffd-647e7f186a49`;
  task supervisor2767729/campaign2767734 verified gone. Foreign2655895
  remained untouched.
- Input `468250a72a1ce764e9451fc7b4cd51bd29372d3e688b832e1e9c1f002e11bc6c`;
  runtime `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`.
  Native staged-input verification remained enabled.
- Smoke payload SHA256
  `b63f204361e699368b0931a8e7995376bbd28d215ac77abeb5994e82748a5d22`.
  Native command20.722436s; worker17.408332s; direct publication0.208046s.
- Native run1793172787.302797–1793172838.105210 =50.802413s;
  through canonical terminal reconciliation1793172847.4052958 =60.102499s.
  Charge **70GPU-s conservatively**, rounded up to the next ten seconds.
- Original audited recipe2350s + first launcher failure60s + repaired smoke70s
  = **2480/3600GPU-s**, **1120s unused** (0.688889/1GPU-hour charged).
  Both one-request grants are consumed. No retry, active wait or recurring
  model monitor remains; contention, terminal and rejected-gate events acknowledged.

The [first launch failure](k6b-ideal-player-launch-report-2026-10-28.md) and its
repair are separately preserved. The fleet now carries its campaign runtime
when workload snapshots are narrowed. Repaired launcher commands were verified
on both qualified hosts with CUDA hidden; Linux CI passed for the repair.

## Implication for the paper

The previous paid-background player recipe still fails the registered whole
frame rule, despite foreground gains. The ideal-player oracle provides no
accepted evidence to change that result or fund another renderer. It also says
nothing new about the unscored Med window, fair pixel ROI, neural-codec anchor,
full wire serialization or untouched evaluation. TOMM readiness remains blocked
on those scientific requirements. A future execution/resource-contract change
needs a new explicit decision; this stopped attempt authorizes no further GPU
request. Preserve inputs/evidence and start from this report rather than replaying
any of the request IDs.

Final evidence is published read-only at
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-ideal-player-20261028-36f21aa5/evidence.tar`,
SHA256 `eea076e548e24262aebfe8254bc8e489e3277f7cae273ead6606816057e3583f`,
1239040bytes/26files, with whole-file and member hashes verified after transfer.
Includes native metadata/logs, raw smoke arrays, independent400-image/source
audits and their code, acknowledgments, and the first-attempt evidence archive.
