# Joint side-rate feasibility after the appearance NO-GO

The next test should change the background representation. On the current
scale4 decoded base, neither free keypoints nor free non-keypoint costs rescues
even the best tested source-pixel oracle. This is a CPU accounting result on
saved pixels, not a new codec result or a universal limit on pose coding.

## Measured constraints

Invert the achievable SVT upper concave hull in rate versus negative distance,
using exactly the parent study's linear equal-rate comparison. The equality
boundary is interpolated only within the observed quality range. PCHIP BD-rate
integration is a different calculation; no BD-rate is claimed here. At the
boundary there is a tie; a robust win requires a strictly smaller side rate.

| Fixed scale4 policy | Central side ceiling, kbps | Ceiling with +0.001 adverse distance |
|---|---:|---:|
| Actual frozen player overlay | 0.784306 | **0.668734** |
| Original pixels inside fixed R | 1.345937 | **1.230365** |
| Original pixels inside predicted support | 1.068787 | **0.953215** |

Each row uses one policy for both metrics and the smaller of their ceilings.
LPIPS is the limiting metric in these three rows. No choice across metrics.
The base itself has zero central headroom and negative robust headroom.

The original side rate is 5.607129kbps: KPS 3.530000 and all other existing
optimistic terms 2.077129. Both parts individually exceed the best oracle's
1.230365kbps robust allowance. Thus making either part entirely free while
retaining the other fails. Keeping these pixels would require reducing the
combined side rate by more than 78.06% for ideal_R, or 88.07% for the actual
overlay. Unmeasured wire costs only tighten these constraints.

This does not prove that improving background quality is the only possible
route: joint side reduction or a different reconstruction could also work.
It does show why another appearance-only renderer, packet wrapper, or longer
adapter amortization alone is poorly targeted at the present failure.

## Bounded lossless packet pilot

The canonical transport archive was copied read-only after the old scratch
location proved unavailable. Its SHA matches the independently preserved input
identity. The Al packet is exactly 4236 bytes; its SHA matches the original
paid stream's bootstrap. Duration is 9.6s, derived from native IVF bytes/rate,
not the number of sparsely scored ticks.

| Encoding of the original KPS bytes | Bytes including 8-byte wrapper | kbps |
|---|---:|---:|
| Original KPS | 4236 | 3.530000 |
| zlib level9 | 4255 | 3.545833 |
| bzip2 level9 | 4705 | 3.920833 |
| LZMA preset6 | 4304 | 3.586667 |

All three independently decompress exactly to the original packet, with its
internal integrity/framing retained. All grow it. Stop this three-setting
pilot; no wrapper is adopted. This does not bound changing the symbol
representation, prediction or quantization. No new images or neural scores
were generated. Local analysis took 0.033s; GPU spend stays **2730/3600s,
870s unused**. No fleet request was submitted.

## Reproduction and provenance

Machine-readable result: [JSON](k6b-side-feasibility-2026-10-28.json).
Implementation: `experiments/tennis/k6b_side_feasibility.py`; its exact SHA is
recorded in the JSON. The result checks full input archives before reading
members, all80 per-frame base/actual controls exactly, and shape/finite/tick
coverage. Tests exercise hull domination, inverse sensitivity, out-of-range
rejection, negative headroom, packet framing and changed-input rejection.

Parent job20261027T211451Z-f6a754eb, code01888c76ce2dd2d7ffbb5ceb63f9a28e774a41f1,
gpu2 A6000 index1; actual quality payload
e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1.
Parent oracle job20261028T080453Z-34e50ec2,
codee20b587ee9709bacd110c561776695d48fd38848, gpu3 A6000 index0;
payload d571699973a7bea8e4c65c2fb82a56c318520129e2cc3385ca6a9988c0c8105d.
Native encoder/decoder, runtime and metric identities remain those in the
[audited full report](k6b-ideal-player-full-report-2026-10-28.md) and its
referenced real-player report; this analysis invokes none of those networks.

Original transport job20261025T082944Z-29f021a5 full/published.tar:
SHA6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7.
Al KPS SHA38c29a952e59e2aeb72de71284286dc2855df3cd0de948571c493bc23349ed50.
The original sources and all completed evidence remain read-only.

```bash
python -m experiments.tennis.k6b_side_feasibility \
  --result /absolute/full-ideal-published.tar \
  --prior /absolute/prior-real-player-published.tar \
  --transport /absolute/original-transport-published.tar \
  --out /absolute/new-analysis.json
```

One familiar development window and one background setting remain the scope.
The [next proposal](cards/k6b-paid-clean-background.md) changes that background
without another generator fit; it needs a new resource/research decision.
DCVC, fair pixel ROI, complete wire costs and untouched evaluation still follow
a candidate passing this SVT feasibility gate. TOMM readiness is not established.

## Verification

Ruff and full configured mypy pass in a clean environment with the pinned
project dependencies. The complete local pytest run has only the two previously
recorded macOS limitations: sandbox-denied `ps` and the protected-storage symlink
assertion against a server path absent on this Mac. The process-control test is
passes outside that sandbox, along with the new analysis, original oracle and
document-ledger tests:23passed. Earlier checks using an incomplete recovered
temporary environment were discarded and replaced by the clean pinned run.
The new analysis introduces no changes to either failing code path. Linux CI is
the merge gate; do not describe the local full suite as passing.
