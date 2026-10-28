# Ideal-player full evaluation — finite appearance NO-GO

Full evaluation completed all80Al031@129 ticks in job
`20261028T080453Z-34e50ec2`. The independent CPU audit passes. Neither of the
two source-pixel oracle policies rescues either whole-frame metric at the
original total22.737962kbps, including the fixed+/-0.001 sensitivity. This is
the predeclared **NO-GO for these finite same-background combinations**.
Do not fund an appearance-only renderer at this operating point without a new
dominating bound. This does not rule out arbitrary future images or renderers.

## Result at the original total rate

Same80saved ticks0:240:3, full1920x1080, same scale4 decoded base, same fixed
masks/support, same LPIPS/DISTS weights and definitions. Lower scores are better.
The anchor is the achievable upper concave envelope of the three previously
encoded SVT points, interpolated at the original optimistic total rate. This is
one operating point; no BD-rate or two-window average is fabricated.

| Policy | Whole-frame LPIPS | Whole-frame DISTS |
|---|---:|---:|
| Achievable SVT anchor at22.737962kbps | **0.234057** | **0.164586** |
| Paid scale4 base alone | 0.282573 | 0.192684 |
| Frozen actual player overlay | 0.275787 | 0.169612 |
| Source pixels inside original fixed R | 0.270927 | 0.167818 |
| Source pixels inside saved predicted support | 0.273325 | 0.169666 |

The best tested oracle is source pixels inside R. It still loses LPIPS by
0.036870 and DISTS by0.003232, both larger than the0.001 sensitivity. A single
ideal policy therefore cannot win both metrics; neither even wins one.
The actual candidate remains in the tested envelope: perceptual scores are
not mathematically monotone, and the predicted-support oracle is slightly
worse than the actual overlay for DISTS. No theorem about arbitrary neural
outputs follows from source-pixel substitution.

The source pixels are explicitly an oracle grant, not a delivered method.
Original base17.130833kbps + optimistic side5.607129kbps =22.737962kbps;
no cost is silently removed for the primary comparison. This retains all
previous bootstrap, framing, amortization and development-footage limitations.

## Timing policy and execution

The user explicitly directed launch because timing variation was the only
reason the previous full run had been blocked. The accepted
[timing-secondary revision](cards/k6b-ideal-timing-secondary.md) removed that
veto prospectively. It did not change any pixels, scientific thresholds,
controls, memory or hard budget limits. The saved spec and summary both record
`--timing-secondary`; the old default still rejects unstable traces.

Fresh smoke passed every retained check. The estimate used the slower warm
sample, cold excess, setup once, direct publication and35% safety:
225.031039s including publication, below the unchanged550s full allowance.
Full command actually completed in144.138064s (worker140.967117s). These are
shared-GPU resource records, not performance benchmark claims. Both historical
base/actual score arrays reproduce exactly across ALL80ticks, max error0.
Peak allocated3082.784668MiB/reserved3740MiB<8000; CUDA confirmed. No training,
generation, SAM inference or RGB conversion. No additional method search.

## Independent checks

A CPU-only audit independently verifies400image byte identities, original
source and paid base, all actual-overlay hashes,160oracle reconstructions from
original R and independently decoded/projected saved support, all80score controls,
finite4x80arrays, means, the achievable anchor at the fixed rate, sensitivity
and NO-GO. It reports PASS;25.178952wall/25.162208CPU-s. A separate check
matches all40native source files to the reconstructed Git archive. Full payload
SHA matches the native publication metadata. No GPU fallback or extra samples.

Secondary predeclared accounting point: if every non-KPS side cost is free,
total rate is20.660833kbps. Its anchor LPIPS0.252029 still beats the best
oracle0.270927; DISTS0.174995 loses to oracle0.167818. This secondary mixed
point does not promote the primary result. It supports investigating background
quality/bit allocation and the KPS floor before another appearance renderer;
it does not establish any unmeasured codec or bitstream improvement.

## Provenance and cost

- Job `20261028T080453Z-34e50ec2`, code
  `e20b587ee9709bacd110c561776695d48fd38848`.
- Source archive `45d4f23ecc8c386d650a8295d68897b4f996b57ae0f13eee9ad4aee03de0dc1b`;
  native tree `4d86c3a12991fac25ccd65ac19aa4ea8c5bae2ebf820cf970fba9b300abf3267`;
  spec `72235ba850224ea8f87d6446001d8cd49d860233e0b97c7a2fc07c67ca1f70f7`.
- Input `468250a72a1ce764e9451fc7b4cd51bd29372d3e688b832e1e9c1f002e11bc6c`;
  packed runtime `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`.
- LPIPS backbone `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`;
  DISTS backbone `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`;
  heads/normalization unchanged in the frozen runtime and native helpers.
- Full result `d571699973a7bea8e4c65c2fb82a56c318520129e2cc3385ca6a9988c0c8105d`,
  61440bytes; raw4x80scores plus hashes and summary, no partial result.
- gpu3 A6000 index0 UUID `GPU-6f70a27e-4171-22d7-3ffd-647e7f186a49`.
  Supervisor2768363/campaign2768368 ended; foreign2655895 remained untouched.
- Native admitted run223.461061s; through canonical terminal reconciliation
  241.753302s. Charge **250GPU-s conservatively**. Cumulative original-cap
  spend **2730/3600GPU-s**, **870s unused** (0.758333/1GPU-hour charged).
  No additional request is needed or authorized by this result.

The original real-player recipe's whole-frame failure remains unchanged. The
oracle is on one familiar development window,80/240ticks and one rate; Med,
fair pixel ROI, the neural-codec anchor, complete wire serialization and untouched
evaluation are not established. The local decision is now made: stop this item,
preserve evidence, and revise the background/rate question before new component
compute. This is useful negative evidence for PointStream, not TOMM readiness.

## Preserved evidence

New read-only archive:
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-ideal-player-full-20261028-34e50ec2/evidence.tar`,
SHA256 `b1ecfb5b5ded26dd11b9a18809b07b13f22b02a4dff4d97c01b21cc12b90a309`,
1536000bytes/26files. Whole-file and all member hashes verified after transfer.
It contains raw full scores/native metadata, the exact40-file source archive,
input identities, original control payload, CPU pixel/source audit code and
records, event acknowledgments and a portable NumPy-only numeric audit.

After extracting to a new directory, reproduce the numeric decision with:

```bash
python audit-portable.py --result full-published.tar --prior prior-f6a-full-published.tar \
  --manifest input-manifest.json --out /absolute/new-audit.json
```

The portable audit passes with zero control error and NO-GO. The separately
archived594739200-byte image pack retains all original/source/oracle pixels;
large original inputs remain referenced by their exact hashes. No existing
Datasets files were overwritten or deleted.
