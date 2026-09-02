# BP45 — Ultra-low-rate AV1/VVC search

**Roadmap IDs:** M1 then E1  
**Preferred harness:** Cursor implements and validates; a detached batch runs the
sweep; Codex/Claude approves bounds and interprets results.  
**Outcome:** determine whether PointStream wins on rate--quality in the low-rate,
long-eligible-scene regime.

## Read

`AGENTS.md`, `plans/ROADMAP.md` §§2–3 and 8,
`plans/TERMINOLOGY.md`, `plans/BP31-findings.md`,
`plans/BP35-perceptual-bdrate.md`, the codec contracts, and the current paired
ladder. Do not read unrelated briefs.

## Owns

- metric direction/curve handling under `src/components/metrics/`
- focused metric tests
- `experiments/tier/low_rate_*.py`
- `outputs/bp45-low-rate/**`

Do not modify background geometry; BP44 owns it.

## Part 1 — instrument calibration

1. Add explicit metric semantics: name, units, higher/lower-is-better, valid
   range and transform used by curve integration.
2. Re-run identical, mild, severe and unrelated anchors for every headline
   metric.
3. Resolve and record AV1/VVC binary paths, versions and presets.
4. Probe each encoder's full legal quality range at fixed source resolution and
   frame rate.
5. Reject empty, undecodable, wrong-size, wrong-frame-count or non-monotone
   outputs.
6. Record the smallest valid bitstream and its measured quality.

Primary quality is full-frame VMAF. Y-PSNR and SSIM/MS-SSIM are secondary.
LPIPS remains diagnostic until its direction and scale are calibrated.

## Part 2 — coherent PointStream sweep

Start generation off. Sweep the following in a small staged search rather than
one Cartesian explosion:

1. background resolution × background quality;
2. correction off/coarse × background setting;
3. appearance resolution/quality/refresh;
4. motion density/precision;
5. add crossover-neighbour points to make valid curves.

Every operating point records the complete transmitted-byte ledger. Assert that
the intended rate-bearing categories move or explain why they do not.

Run AV1 and VVC on the exact same frames, dimensions, frame rate and colour
convention. They encode each whole sequence jointly. No frame dropping or
downscaling is allowed in the main comparison.

## Bounds and controls

Write the bounds file before the first system encode. Use a deliberately broad
two-sided BD-rate interval because the experiment tests the regime:

- PointStream vs each anchor on VMAF: [−80%, +180%];
- no codec point may decode to zero/wrong frames;
- identical-input metric anchors must remain at their calibrated ceiling;
- late-frame VMAF and Y-PSNR changes must be bounded separately;
- the conventional fallback control must reproduce the reference codec result;
- an object-stream-off control must show whether any win comes from background
  reuse alone.

A negative result triggers an extra validation pass. A non-overlapping curve is
not assigned a BD-rate; test strict dominance at the lowest anchor point instead.

## Search disclosure

Persist every tried configuration, not just the winner. The report must identify
the axis that selected the final regime and the date the criterion was frozen.
E1 is diagnostic on two videos. It becomes evidence only after the frozen rule
passes E2 on at least six independent videos/matches.

## Completion report

Follow `plans/SESSION-REPORT.md`. Include calibration, codec floors, every
curve and overlap, size/quality/encode/decode time, byte ledgers, alarms, failed
points, exact commands and the Gate-A decision.
