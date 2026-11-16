# Experiments: combined tennis foreground evidence gap audit

### 2026-11-15 — read-only scientific audit requested by the user

**Decision:** retained evidence supports promising generated-player quality and
a separately confirmed narrow ball-neighbourhood trade-off. It does not yet
establish a winning complete player/racket/ball foreground codec. Combining
the positive results on paper cannot establish the result of combining their
receivers. This audit creates no compute authority, changes no raw evidence,
and reports no new experiment. The root session separately audits merged PRs;
this note concerns scientific compatibility of the retained results.

## What is already answered

| Question | Retained answer | Limit |
|---|---|---|
| Did the intended player demo produce a measured improvement? | Yes. W7 al028 player LPIPS .717985 to .304181 and DISTS .496145 to .242025 versus its native base. | Base14.121 versus corrected hybrid19.240kbps; direct visual pair is unequal-rate. |
| Does a player benefit survive matched-rate controls? | On al028, corrected W9 hybrid player LPIPS .304181 versus best DCVC .384051; DISTS .242025 versus .302989. | Historical sampled-tick regional metrics, modeled asset amortization and incomplete operational receiver. |
| Does that establish a full-frame win? | No. At the same corrected rate, hybrid full LPIPS/DISTS .284529/.175013 versus DCVC .145873/.120687. | Foreground-priority trade-off; no human preference equivalence. |
| Does paid tracked ball coding have a narrow benefit? | Three source groups have supported negative crop-DISTS BD-rate, equal-source log-rate aggregate -26.28%. | Sparse visible point neighbourhoods; descriptive three-source uncertainty; full-frame costs +17.39%DISTS/+24.64%standardLPIPS. |
| Are conventional ROI controls already present? | Yes, both programmes include them; ball confirmation has native/neural/raw-tracker/matched-placement ROI points. | Different masks, sources, native profiles, objectives and support; they cannot be treated as one matched foreground experiment. |
| Is racket preservation/benefit established? | No independent racket endpoint was identified in these W7/W9 and ball records. | Player masks may incidentally cover racket pixels; this is not racket completeness, tracking truth or racket quality evidence. |

Exact records: `w9-inputs/w7-repaired-2h.json:6–185`, native means in
`w7-inputs/ivf-mkv-corrected-tier1-report-20261105.json:6298–6317`,
`w7-untouched.md:289–295`, and
`submission-ladder-ball-confirmation-analysis-20261115.json`.

## Why existing results do not compose into the requested claim

- **Sources and training:** W7 Tier1 has six windows from two physical matches;
  the generator base saw other footage of those matches and each match adapter
  uses other clips. Tier2 has three windows from one new match, with adaptation
  in its first half and evaluation in its second. These are two and one match
  clusters, not six and three independent sources. Corrected W9 Tier1 passes
  its historical window-level rule conditionally on resolved anchors; Tier2
  remains inconclusive on LPIPS. The new ball cohort instead comprises three
  RacketVision TRAIN sources (209/235/269), with disclosed potential BallTrack
  encoder training overlap. No common combined-output cohort was evaluated.
- **Time and geometry:** W7 evaluates80 every-third-frame ticks in240-frame
  native1080p windows, with generated players at those ticks. Ball confirmation
  evaluates all282 frames at native25/25/60Hz after spatial conversion to720p.
  Playback/export is not a demonstration of native-cadence generated output.
  A combined decoder must specify every intermediate frame, fallback, update
  schedule, coordinates, colour conversion and object compositing order.
- **Masks and objectives:** W7 regional LPIPS/DISTS uses a historically frozen
  player/scoring support derived from source masks and drawn support. Ball
  quality uses64px crops around sparse independently labelled visible points;
  a32px transmitted square is not a ball segmentation. Neither result measures
  a common foreground union of players, rackets and ball with fixed70/30
  foreground/background DISTS and corroborating regional LPIPS. Thin rackets,
  missing player limbs and absent/unknown balls must not disappear from the
  reference definition. Existing person/player SAM prompts are explicitly
  known to miss rackets or player parts in the annotation-pilot record.
- **Costs and comparisons:** W9 corrected a fourfold modeled-bank underbill;
  corrected19.240kbps still assumes twelve-crop reuse and amortizes adapter/
  guider/bank over7200s, omitting some operational framing. It is not a cold
  short-clip bill. Ball PTP pays actual base/patch/framing bytes, but its sender
  also prepares six native/neural bases. A combined stream must count its base
  once, include every asset and update, and state the amortization/session
  contract. Whole-foreground ROI and matched neural/enhancement controls must
  be encoded under the same source/timing/accounting contract. W9 still lacks
  Med000 DCVC/half+20 and Med019 half+20; its positive pooled result is
  conditional on resolved families. Native container/cadence sensitivities
  cannot silently replace the registered primary comparisons.
- **Receiver:** public PSC1/PTP1 encode/decode/inspect supports native fallback
  and tracked pixels, not W7 learned-player reconstruction. PTP1 validates a
  complete native base, one64px patch stream,32px placement and pinned decoder/
  NumPy identities. Its render path decodes native pixels then overwrites the
  ball rectangle in RGB; it cannot accept an arbitrary generated-player frame
  as its native base stream. W7's saved crops/composites and W8's exact CPU
  conditioning proof do not establish fresh isolated learned decoding. The
  early November1 W8 learned-runtime attempt failed at CUDA initialization,
  not on observed fidelity. **November16 correction:** later
  `20261106T005057Z-77c276fa` passed real isolated1/3step model sampling with
  kernel/tensor/denial receipts; no production30step quality witness followed.
  See the [current receiver reuse audit](player-receiver-witness-design-20261116.md).
  A combined container,
  generative resource contract and actual learned reconstruction remain work.
- **Nonadditivity:** ball and player BD-rates use different sources, reference
  envelopes and common-quality intervals. They cannot be added, multiplied,
  averaged into a whole-foreground saving, or ranked against each other by
  magnitude. Neither do separate full-frame losses predict the combined loss.
  Shared base costs and overlapping paste regions make that inference invalid.

## Minimum efficient path to answer the combined question

1. **Freeze one combined hypothesis and receiver contract.** Reuse the existing
   generator and tracked-pixel mechanism without a new method/parameter sweep.
   Define native cadence, every-frame fallback, geometry, colour, player/racket
   mask generation, ball-after-player or other explicit composition order, and
   all generic/content-specific resources. First inspect saved compatible
   inputs; do not fit new adapters or launch GPU jobs merely to fill this list.
2. **Run the smallest changed-construction development test after a costed
   release.** On an already exposed source, obtain actual combined reconstructed
   frames and complete bytes, plus native, player-only and ball-only ablations.
   Compare actual same-source native/ROI/neural controls and a relevant matched
   enhancement or super-resolution control. Reuse saved streams only when
   source/frame/geometry/cadence/encoder/decoded identities match. Use three
   frozen rates and measured common-quality support. A short correctness smoke
   first proves non-overlap/overlap composition, absent hints, every frame,
   serialization and fallback. Cost forecast and each waited stage remain
   bounded by the project's45-minute rule. No arbitrary10% gate applies.
3. **Measure the intended foreground before claiming its win.** Obtain blinded
   source-only reviewed/corrected players-plus-rackets references on the frozen
   evaluation schedule, with independent visible/absent/unknown ball labels.
   A minimal annotation schedule can support a clearly declared sampled
   foreground endpoint; it cannot become an all-frame completeness claim.
   The tennis source-only annotation pilot must measure correction effort;
   the separate18-frame egocentric pilot does not establish tennis labour or
   reference acceptance. Preserve unknowns and unprompted missing-object checks. Report each
   object class and foreground/background/full/temporal results, not only the
   favorable aggregate. No detector used by the sender may judge its own
   training footage as an independent task oracle.
4. **Confirm one frozen combined candidate on new physical matches.** Choose
   matches outside previously opened outcome cohorts, with exposure/training
   provenance recorded. If same-match adaptation is part of encoding, freeze
   a disjoint adaptation segment and bill it; distinguish base-generator
   pretraining overlap from permitted encoder adaptation. Mechanically choose
   windows within each match, aggregate within match first, and treat matches
   as the inference unit. Do not call three windows one independent source
   each. Existing opened W7/ball sources remain development/historical evidence.
5. **Size the cohort for inference, not a rate-saving floor.** Freeze primary
   weighted70/30DISTS, LPIPS corroboration, paired source log-rate effects,
   multiplicity/uncertainty method and common-support failure policy. Use
   conservative source-heterogeneity scenarios to choose a feasible power/
   precision target, including effects below10%; report the effect-size
   detection curve when budget limits precision. Independent-source sample
   size cannot honestly be fixed from one development match. Lack of support
   or an interval spanning zero means unresolved, not scientific uselessness.
   The existing prospective inference-policy note gives assumptions and exact
   small-sample limitations. Stop for a substantive correctness failure,
   exhausted budget or a scientifically resolved decision, retaining all data.
6. **Prove the actual complete receiver and runtime scope.** Independently
   reconstruct from the combined packet and declared installed resources with
   source/training/annotation paths denied. Check native children and full
   frame coverage; distinguish filesystem restrictions from network/MCP
   isolation. Measure cold initialization and steady processing separately,
   including player generation, ball sender preparation and all updates.
   Receiver correctness does not itself establish quality or real time.

## Common tennis model versus session adaptation

The approximately49MB tennis base need not be charged anew for each session
under a declared installed-decoder-resource comparison. This was already the
historical design, not a classification invented after the W7 outcomes:
`k6b-players.md:680–685` specifies one rank64 tennis LoRA fitted once for the
sport and shipped with the decoder; `:777–781` records job
`20261023T194934Z-b0dcbb19`,36 windows from five matches,8,597 every-second-frame
crops and1,918 optimization steps. The actual frozen checkpoint is48,878,567
bytes (`:2063–2069`), SHA256
`85e3c65c9777b799f9ed0df3a5c8da567fe8a92e4ac83d115719527acb739eb4`
(`:1493`). Archive075b001a…3285 and W7's `windows.json` bind the shared base.
The user's acceptance of sport fine-tuning as a generic resource is consistent
with this boundary. Prospectively declare and pin all installed resources for
PointStream, native and neural anchors, report their download/storage footprints
and cold initialization separately from transmitted session bytes, and retain
the historical base-charged sensitivity as a deployment scenario. A common
49MB resource alone is not a reason to reject the session-rate result.

This boundary does not make the training source-independent: W7's familiar-match
tier still shares broadcasts with the generic tennis-base training data. Keep
that disclosure, distinguish it from permitted encoder adaptation, and confirm
new claims on independent physical matches with the generic model frozen before
the new outcomes. A model selected, refitted or updated specifically for the
test session is session-specific state until a new, prospectively declared
generic resource/version and evaluation boundary are established. Do not remove
historical overlap or retrospectively rename a match-specific adapter generic.

The rank4 match adapters, trained pose-guider changes and reference bank are
content-specific. W7/W9 bills adapter-plus-guider Al/Med payloads of1,488,252/
1,495,592bytes and a corrected modeled twelve-crop bank over7200s
(`experiments/tennis/w7_rates.py:14–46`). Keep those terms, actual serialization,
refreshes and receiver provisioning in the session contract; a sport-specific
installed base does not justify granting this content-specific state for free.
Both an already-provisioned session and a cold content-session profile can be
reported, with the exact payloads and amortization horizon explicit.

A causal “train on the first N seconds, then improve the rest” claim needs a
different prospective test. Freeze N, adaptation data and update schedule;
admit only frames already arrived at the encoder, and make decoder use of each
update start after fitting, serialization, transmission and loading complete.
Transmit/decode the warm-up and adaptation period using the declared fallback,
include those frames and bytes in the session result, and account separately
for adaptation compute/energy, wall delay, buffer, peak memory and subsequent
refreshes. Concurrent training cannot be assumed free or instantaneous. Audit
reference selection, pose extraction, crop tracking and temporal generator
contexts for future-frame access; any lookahead must be bounded and declared.
Compare with the same frozen system without adaptation, using the same session
timeline and total-cost contract, to isolate an adaptation benefit. W7 Tier2's
three disjoint first-half fit clips and second-half evaluation establish a
recorded temporal partition (`w7-untouched.md:776–780`); they do not demonstrate
first-N-second training, causal online updates, adaptation-completion latency,
or a complete cold-session quality/rate result. No actual fitting or new
runtime measurement is authorized or performed by this amendment.

**Defensible claim now:** historical generated-player reconstructions retain a
substantial player-region advantage on evaluated windows at corrected modeled
rates, while paid tracked ball pixels independently retain a narrow
ball-neighbourhood trade-off on three sources. Full-frame penalties, training
provenance, sparse sampling, accounting and receiver limitations accompany
both. Complete foreground benefit, racket benefit, native-cadence generative
decoding, human preference and a winning fully billed combined codec remain
unestablished. The combined hypothesis is credible enough for one bounded
same-source screen; the historical positives should not be erased by the
failed static256px additive-patch construction, which is a different method.

Additional primary repository records inspected: `w7-untouched.md:15–54`,
`w9-adapter-break-even.md:31–46,224–237`, `w8-wire-audit.md:19–21`,
`submission-ladder-ball-confirmation-2026-11-15.md` under `../cards/`,
`submission-ladder-annotation-pilot-20261114.md`,
`submission-ladder-chapter14-demo-audit-20261115.md`,
`src/pointstream/tracked_patch.py:34–64,167–199` and public CLI routing in
`src/pointstream/__main__.py`. No new statistical calculation, GPU execution,
raw-data mutation or saved-result reinterpretation was performed for this note.
