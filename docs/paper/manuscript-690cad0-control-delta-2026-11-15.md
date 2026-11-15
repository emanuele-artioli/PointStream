# Author-review draft: control updates to manuscript 690cad0

**Disposition: copy-ready proposals, not adopted manuscript edits or submission approval.** The preserved ZIP and every Overleaf/PRESLEY source remain unchanged. The manuscript already presents an empirical study with important limitations; these proposals update the directly affected reference comparisons and competing explanations, rather than invent a new codec-wide victory.

## Audited source and evidence boundary

Preserved source: `/Users/manu/Documents/PointStream-review/2026-11-14-690cad0/PointStream-source-690cad0.zip`, SHA256 `249ee6419ab9b73731d16f5fb69e35d771c12216029803a40226db2f7dddb90b`. Line numbers below count the original UTF-8 ZIP members, including comments; they are not PDF page numbers. Active `main.tex` inputs were inspected selectively. Historical `archive/*.tex` drafts were not treated as the active paper. Read-only extracted members are under `/private/tmp/pointstream-690cad0-paper-audit-fo7c497d/`.

The evidence cut is the accepted records listed below and [readiness](submission-ladder-readiness-2026-11-15.md), read at SHA256 `9c201f46773e7761054142d6041a80ad37ebd8632000b73ba364be60ff68e7f4`. Independent four-case receiver qualification was still pending at that cut. A retained successful worker inside a failed job does not clear that gate. The later status-only update below supersedes receiver status, without changing quality results or establishing fresh sender timing.

**Later ROOT acceptance:** job `20261115T043834Z-4dd0744d`, frozen `f5486a92`, passes all four actual filesystem-denied receiver cases: 208 exact frames and 19 EACCES probes per case. Full stage SHA `0a75c44e`, raw proof archive SHA `279ea808`; [independent evidence](../experiments/submission-ladder-receiver-evidence-archive-20261115.json) preserves the failed predecessor and successful replacement. Scope is inherited filesystem restrictions, not network/MCP isolation, independent native-child birth observation, fresh sender inference or speed.

| Record | SHA256 | What it supports |
|---|---|---|
| `docs/experiments/submission-ladder-ball-roi-pooled-analysis-20261115.json` | `5965d2b1ff2178a9b09d8d97bf426e2e2061e93b954e4a7c5d63b5b9b58ff7c9` | Same six historical paid side-channel points against original and ROI-strengthened references; own-overlap and matched-interval sensitivities. |
| `docs/experiments/submission-ladder-paid-tennis-analysis-20261115.json` | `b8adeea366b7c12586b41d06e4998f32801bddf9b77df3d09cacafe146d8abc7` | Actual paid PTP1 bytes and direct comparisons with three original SVT/native and streak points after exact physical replay. |
| `docs/experiments/submission-ladder-paid-tennis-strong-reference-20261115.json` | `7f0b23002bc7d1d56cd1c60305326d5cf9497093d9de444690c77e89eccab16f` | Paid pixels against GT-ROI3 and pooled native/neural6+GT-ROI3. |
| `docs/experiments/submission-ladder-tennis-analysis-20261115.json` | `c2cee1f7bb993c6d157e5079cb0db0192430aa157af0fbf2c371e2ebf9382998` | Frozen 64-frame tennis ROI/static-patch development screen. |
| `docs/experiments/submission-ladder-visor-analysis-20261115.json` | `fb3dafe21c108df59b5b9ef843b7cb49f7c4db6a610bc7ff53a111d892a993f7` | Unchanged 64-frame egocentric screen; a separate source/objective from the ball study. |
| `docs/experiments/submission-ladder-paid-pixels-evidence-archive-20261115.json` | `50681a9c16260f1aa817aff8b48b293aad18a4039c420710248fb71239f9082b` | Accepted physical evidence index, replay and archive provenance; not new quality measurements. |

The strong-reference calculation authenticates primary scalars `071dd85e…`, ROI scores `b213a4ed…`, original pixel scores `00e2885b…` and the paid reprice above. Paid tennis stage SHA256 is `40c13a6df8b352c4c8813ac31d50678fabc3d021b4e810ad95c3945481f10a0b`; its archive is `66df76d85fa37f4228a7b50abaaa75b0e71dcf28639fcdd2b7c584960db0e418`. All proposed percentages below were checked against those saved scalar records. No codec, model, GPU or new literature search was run for this audit.

## Claim locations requiring an author decision

| Original ZIP member and verified line(s) | Small original snippet | Proposed disposition |
|---|---|---|
| `main.tex:115` | “improves region LPIPS BD-rate by 76.97\%” | Historically correct for the original six-point reference. Replace the abstract emphasis with the stronger-reference and same-position controls; retain the old value explicitly in the body/table. |
| `sections/introduction.tex:19` | “reports a serialized ball representation evaluated against conventional and neural anchors” | Extend the contribution to the matched-objective ROI and real coded-pixel controls. Do not attribute the benefit distinctively to streak appearance. |
| `sections/introduction.tex:21` | “strongest current claim is a task- and region-specific coding trade-off” | Distinguish historical task evidence from the new paid pixel candidate's fidelity evidence. |
| `sections/evaluation.tex:99` | “ROI LPIPS and DISTS BD-rates of $-76.97\%$ and $-44.44\%$” | Preserve as the original-reference result; add the strengthened reference and direct coded-pixel control immediately here. |
| `sections/evaluation.tex:115` | “primary 48 ... pooled native DCVC-UF/SVT-AV1 envelope” | Keep the existing graphic historically labeled; its left panel omits the later ROI/pixel controls. Add the proposed control table and qualify the caption. |
| `sections/evaluation.tex:108` | “streak, disk and brightness controls yield identical hits” | Keep. Add that new paid pixels have fidelity scores, not newly measured task outcomes. |
| `sections/evaluation.tex:499–508` | “egocentric direction explores ...” | Precede the older studies with the separate frozen two-domain screen; retain all older samples and protocols separately. |
| `sections/system_design.tex:55` | “fresh-process transport test can verify the interface” | Add the concrete PTP1 contract and separately scoped, later accepted filesystem-denial evidence. |
| `sections/system_design.tex:78` | “weighted sum ... of ... dB values” | Keep the historical PSNR statement. Separately define the new 70/30 regional perceptual objective; do not silently reinterpret either score. |
| `sections/conclusion.tex:8` | “with small whole-frame rate penalties” | Attach those small penalties only to the historical side-channel result. Current paid pixels have materially larger full-frame costs. |
| `appendices/evaluation_notes.tex:25` | “Only original 16 has gold equality; full 48/292/586/80 have no archived gold” | Preserve the original protocol's limitation, but distinguish the newly reproduced 48-frame SVT62 source/native control from retroactive recertification of the original full matrix. |

The original crossclip task/absence paragraphs (`sections/evaluation.tex:102–108`) remain valid scoped history. They must not become evidence for PTP1 pixels, an egocentric quality gain, held-out confirmation or absence safety.

## Copy-ready abstract replacement

Replace `main.tex:115` with the following paragraph, retaining the surrounding abstract environment. The exact wording remains for author review.

```tex
Object-centric semantic video coding represents scene structure, appearance, and motion for reconstruction at a receiver. A compact description alone does not establish a coding advantage: appearance, correction, and delivery information must also be paid. We study this gap through PointStream, a modular experimental instrument. Foreground-removal diagnostics change the target; a conditional player-region perceptual advantage retains whole-frame costs and unresolved unseen-match evidence. On one exposed 48-frame ball window, strengthening the original conventional/neural reference with privileged ball-region allocation leaves the historical side channel with 60.95\% crop-LPIPS and 36.23\% crop-DISTS BD-rate savings. A paid coded-pixel control at the same transmitted positions improves crop-DISTS by 30.42\% relative to streak rendering; its crop-LPIPS range is strictly better but has no common-quality interval, so no LPIPS BD-rate is inferred. Against the strengthened reference, paid pixels save 54.83\% and 29.28\% on these crop metrics while costing 22.00\% and 12.72\% on full-frame spatial-LPIPS and DISTS. These comparisons use their own supported quality intervals. The controls identify position and local allocation as competing explanations for the benefit. Separately, historical detection rescues coexist with harms and absent-frame false detections. Frozen static foreground-patch screens fail their development target in both tennis and egocentric video. These findings establish scoped object-priority trade-offs and evaluation requirements, rather than a distinctive appearance benefit, general codec superiority, or held-out confirmation.
```

Add/update a `CLAIM` marker with the pooled-reference, paid-reprice, strong-reference and two-domain records above. Keep the existing original ball provenance. Do not clear the abstract's complete-codec/independent-task limitations merely because PTP1 has a functioning public decoder.

## Copy-ready introduction contribution/scope updates

Replace `sections/introduction.tex:19`:

```tex
The study makes three contributions. First, it connects altered-target coding diagnostics to the cost of reconstructing the original scene. Second, it evaluates a serialized ball side channel against conventional and neural references, then tests matched-objective conventional ROI allocation and real coded pixels at the same transmitted positions. These controls separate a useful ball-priority trade-off from a distinctive claim about rendered appearance. Three retrospective conventional-codec clips retain their separate task rescues, harms, and absent-frame false detections. Third, the study reports component negatives and delivery/scoring controls, including an unchanged static foreground-patch screen in tennis and egocentric video. Implemented interfaces, bounded development results, and untested generalization remain distinct.
```

Replace only the final two sentences of `sections/introduction.tex:21` (leave its recovery provenance and footnote intact):

```tex
The current evidence supports objective-specific development trade-offs. Historical ball detection results and the paid coded-pixel fidelity control concern different outputs; neither supplies held-out confirmation, an egocentric quality benefit, or general whole-frame superiority.
```

## Copy-ready primary-ball body and control table

Replace the numerical/interpretive portion of `sections/evaluation.tex:99` with these paragraphs; retain its original sample definition and detector protocol from lines95/99.

```tex
The inputs are the hash-authenticated JPEG frames released with the dataset and their declared limited-range YUV conversion. Against the original pooled six-point native DCVC-UF/SVT-AV1 reference, the six paid ball-side-channel points have crop-LPIPS and crop-DISTS BD-rates of $-76.97\%$ and $-44.44\%$. These historical values remain correct for that reference and common-quality support. Adding three privileged conventional ROI points, allocated around the same evaluation objective using reference ball locations, gives $-60.95\%$ and $-36.23\%$ on the enlarged reference's supported intervals. This is a stronger competing-explanation control, not a deployable automatic ROI sender. The envelope is a constructed hull/interpolation reference, not demonstrated switching between delivered streams.

A second control replaces streak appearance with real coded pixels at each actual decoded sender position. A $32\times32$ source-YUV crop is coded within a padded $64\times64$ AV1 tile; missing sender positions produce no replacement. Positions and visibility come from the accepted serialized automatic sender, not evaluation labels. The paid PTP1 packets include both complete native streams and all required placement/framing metadata. Relative to the three paid SVT streak points, paid pixels save $30.42\%$ crop-DISTS BD-rate. Their worst measured crop-LPIPS, 0.015140, is lower than the streak's best, 0.015685, with overlapping sampled rate support. The quality ranges do not overlap, so no LPIPS BD-rate is reported. This direct control, together with the separate disk/brightness task control, prevents crediting streak appearance uniquely for the gain.

Against the pooled native/neural plus GT-ROI reference, paid pixels save $54.83\%$ crop-LPIPS and $29.28\%$ crop-DISTS, but incur $22.00\%$ full-frame spatial-LPIPS, $12.72\%$ full-frame DISTS, and $30.52\%$ pooled-RGB-PSNR BD-rate penalties. Each comparison uses its own common-quality interval. The magnitude of a historical streak-versus-reference BD-rate therefore cannot rank streaks against pixels measured on another interval; the direct same-position comparison provides that evidence. These are fidelity trade-offs on one previously exposed, all-visible 48-frame window. The new pixel outputs have not acquired the historical streak's task or absence results by metric reuse.
```

The following is a proposed **new** table after that discussion. There is no existing main-paper ball control table to overwrite. Its additional page cost needs checking when adopted.

```tex
% CLAIM(tab:ball-controls-20261115): src=docs/experiments/submission-ladder-ball-roi-pooled-analysis-20261115.json;
%   submission-ladder-paid-tennis-analysis-20261115.json; submission-ladder-paid-tennis-strong-reference-20261115.json.
\begin{table}[t]
\centering\small
\caption{Primary 48-frame development controls, BD-rate (\%). Negative values save rate at equal measured quality. Each row/metric uses its own common-quality interval; rows cannot rank representations by BD-rate magnitude. ``Sep.'' denotes strictly separated observed quality ranges with overlapping sampled rates, not a BD-rate. Full LPIPS is the historical spatial-map mean.}
\label{tab:ball-controls-20261115}
\begin{tabular}{llrrrrr}
\toprule
Candidate & Reference & Crop LPIPS & Crop DISTS & Full LPIPS & Full DISTS & RGB PSNR\\\midrule
Side channel (6) & Native/neural (6), original & $-76.97$ & $-44.44$ & $+1.22$ & $+3.30$ & $+1.00$\\
Side channel (6) & Native/neural + GT-ROI (9) & $-60.95$ & $-36.23$ & $+1.11$ & $+3.28$ & $+1.11$\\
Paid pixels (3) & GT-ROI (3) & $-54.75$ & $-29.08$ & $+17.58$ & $+11.48$ & $+10.50$\\
Paid pixels (3) & Native/neural + GT-ROI (9) & $-54.83$ & $-29.28$ & $+22.00$ & $+12.72$ & $+30.52$\\
Paid pixels (3) & SVT streak (3) & Sep. & $-30.42$ & $+13.33$ & $+13.72$ & $+13.18$\\\bottomrule
\end{tabular}
\end{table}
```

Table label counts mean rate points, not independent clips. The six historical candidate points include neural and conventional bases; the three paid-pixel points use SVT bases only. GT-ROI is privileged. ``Sep.'' means the paid pixel crop-LPIPS range `[0.007459, 0.015140]` is strictly below the three SVT streak values' range `[0.015685, 0.034000]`; common sampled rate support is `[207.425, 456.360]` kb/s. It does not establish a population preference or an unmeasured task result.

Preserve support disclosure: historical side-channel crop-LPIPS support is `[0.030699, 0.059818]`, versus `[0.015685, 0.059818]` with added GT-ROI; corresponding DISTS supports are `[0.093746, 0.174053]` and `[0.083602, 0.174053]`. On the **same original intervals**, the strengthened reference gives `-71.07%` LPIPS and `-39.24%` DISTS, rather than its own-overlap `-60.95%/-36.23%`. Thus the latter change reflects reference/interpolation and support changes; it is not a pure causal estimate of ROI's effect. Retain the full interpolation sensitivities in companion data.

Replace only the left-panel description of Figure6's caption at `sections/evaluation.tex:115`:

```tex
Left: historical primary-48 BD-rate against the original six-point native DCVC-UF/SVT-AV1 reference; this panel excludes the later GT-ROI and coded-pixel controls in Table~\ref{tab:ball-controls-20261115}.
```

Keep the middle/right panel descriptions and their clip-specific denominators. Do not relabel or silently redraw the original76.97 bar as a stronger-reference result.

## Copy-ready separate two-domain negative screen

Insert before the older egocentric study discussion at `sections/evaluation.tex:499`; do not merge the new screen with the four-window H5q sample or the 48-frame ball curves.

```tex
% CLAIM(fixed-foreground-screen-20261115): src=docs/experiments/submission-ladder-tennis-analysis-20261115.json;
%   submission-ladder-visor-analysis-20261115.json; each retains producer/scorer, source, runtime and GPU identities.
\paragraph{Frozen static foreground-patch screen.}
An unchanged three-rate construction tests a full-resolution native AV1 base with an additive fixed $256\times256$ CRF28 foreground patch on separate 64-frame tennis and egocentric development windows. A conventional ROI arm changes allocation within the same native codec. The all-frame objective is $0.7$ foreground plus $0.3$ background regional perceptual distance, with absent-region weight reallocated; it is separate from the historical weighted-dB PSNR score. Tennis scoring uses retained source-model SAM masks, not independent ground truth; VISOR uses dense interpolated annotations, not an entirely human-corrected reference. ROI gives weighted-DISTS BD-rates of $-8.53\%$ and $-5.39\%$, while the static patch gives $+17.10\%$ and $+18.47\%$, respectively. Neither domain meets the registered development screen of at least 10\% weighted-DISTS saving with improving weighted LPIPS. This closes that fixed additive construction on those windows, not dynamic/joint coding or all hand/object representations. It supplies no independent confirmation or promoted added-representation advantage in egocentric video.
```

Optional compact table, if it replaces prose rather than expanding the page budget:

| Separate 64-frame screen, versus native | Tennis weighted DISTS | Egocentric weighted DISTS | Decision |
|---|---:|---:|---|
| Conventional ROI | −8.53% | −5.39% | Both below the10% promotion target. |
| Fixed semantic patch | +17.10% | +18.47% | Neither construction promoted. |

## Copy-ready endpoint and gold-control scope

Add after `sections/system_design.tex:55`:

```tex
The public PTP1 profile accepts raw YUV420 and explicit encoder position hints, and returns RGB24 through a declared BT.709 nearest-chroma conversion. Enhanced packets transmit both complete native streams, timing, geometry, per-frame placement and resource identities; an all-null placement sequence uses the native base without a patch. On the primary tennis window, its three complete packets contain 98,350, 71,152 and 41,485 bytes, including 845 metadata/framing bytes each, and reconstruct the accepted coded-pixel outputs exactly before scores are reused. Retained automatic trajectories are encoder inputs, not a fresh end-to-end inference or throughput qualification. A separately accepted fresh-exec test denies actual source, label, encoder-state and private-model file reads while reconstructing all three tennis points and the egocentric fallback exactly. Its evidence covers inherited filesystem restrictions, not network/MCP isolation, independent native-child births, fresh sender inference or throughput. Process separation alone was not accepted as proof.
```

The separate egocentric fallback is64 full-range1080p/50fps frames in313,310 paid bytes with no patch. It is a functioning receiver mode, not an inference that hands/objects are absent or a quality win. Its RGB conversion differs from historical scores, so do not reuse those scores. These details may be kept in the artifact/appendix if space is constrained.

Add after `appendices/evaluation_notes.tex:25`, leaving its original-receipt qualification historically labeled:

```tex
Later source-matched controls reproduce all 48 original JPEG identities, the declared YUV conversion, and the native SVT CRF62 stream and decoded pixels. All three paid PTP1 tennis points additionally match the authenticated coded-pixel oracle's streams and every reconstructed RGB frame. These new checks apply to those specific controls; they do not retroactively turn the original six-codec-point matrix or the 292/586/80-frame studies into full gold-equality experiments.
```

## Copy-ready conclusion and remaining validation

Replace `sections/conclusion.tex:8`:

```tex
Independent replay, explicit transport and controlled scoring strengthen the empirical evidence, but do not establish a distinctive semantic-coding advantage by themselves. On the primary ball development window, the historical side channel retains a crop-fidelity trade-off after matched-objective conventional ROI strengthens the reference. Real coded pixels at the same transmitted positions then improve directly on streak rendering, identifying position and local allocation as competing explanations. The current paid pixel candidate retains substantial whole-frame costs; the older side channel's smaller penalties must not be transferred to it. Historical task rescues, harms and absent-frame false detections remain separate results, and the unchanged fixed foreground-patch construction fails its development target in both tennis and egocentric video. The conditional player result likewise retains incomplete anchor coverage, assumed reuse and temporal/receiver limitations. Independent references and source-grouped confirmation are still required before promoting these development findings beyond their measured objectives and footage.
```

Add to the discussion following `sections/future_work.tex:9`:

```tex
The next validation stage freezes the candidate before confirmation outcomes and uses corrected, uncertainty-marked references on the fixed source sample. Model proposals or agent review are not human annotation. Missing or inconsistent dense labels remain unknown, rather than becoming empty foreground; fixed windows must not move after outcomes. Source/match grouping and dataset/judge exposure remain explicit. The current evidence contains no controlled human-preference result, held-out multi-match confirmation, fresh complete sender-speed qualification, or promoted egocentric added representation.
```

For the research record, the prepared VISOR human-review pilot is18 frames; the fixed full reference target is192, not the pilot alone. Reserved P35/P12/P11 windows are not yet confirmation-eligible because of annotation gaps. The existing tennis pilot is12 frames; prior tracker selection used the43 RacketVision test rallies, so new clip identifiers do not establish independent matches. These preparation facts justify the visible confirmation limitation; they are not completed human-reference evidence.

## Adoption boundary

This draft supplies text and table changes only. It does not update the original source, Figure6 data, any companion file, marker-disposition registry, Overleaf, PRESLEY or submission package. If adopted, add the cited `CLAIM` provenance, retain applicable visible limitations, refresh the marker-disposition/source-hash bindings, and run the existing empirical-manuscript checker and native paper build. The preserved23+5-page package is not evidence that these proposed additions fit that budget or meet current portal requirements. Author review must choose the final emphasis and layout; submission readiness remains unresolved.
