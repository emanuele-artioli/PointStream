# BP30 — the background as a stream: what landed, and what to distrust

Companion to `plans/done/BP30-background-stream.md` (the design) and
`plans/done/BP24-findings.md` §§16-19 (what it was built on). Numbered to continue
that file's sequence so cross-references stay unambiguous.

Run under `https://github.com/emanuele-artioli/PointStream/blob/ec581e9/plans/prompts/claude-bp30.md`, whose scope excludes wiring anything
into `make_background` (stream D's file this week), adding sidecar codecs
(stream B's), and re-running the paired ladder. None of those were touched.

**In one paragraph.** The background does amortise across scenes: over **five
videos** 16 scenes cost **49.2% ± 6.2%** of 16 fresh plates at native 4K, best
case 29.4% (§29). The first video measured, at 62%, turned out to be the worst
of the five. The marginal scene on that video costs 0.607 ± 0.015 of a fresh
plate — where §18/§19 measured 0.470-0.671 on
isolated pairs. Each scene's payload is independent of every future scene, and
encoder and client hold bit-identical reconstructions, both of which rest on a
low-delay encode being prefix-stable *to the byte* (§20). **Reference selection
is not worth building** — but for a different reason than one video suggested:
at n=5 videos the Canny search does not beat simply predicting from the previous
scene (0.1σ, §29), while `first` is the *worst* free option rather than a safe
default. Use `last`. A keyframe every *k* scenes costs 26.5% at *k*=2 falling
to 4.0% at *k*=8, and every *k* still beats sending a fresh plate (§25). One
video, 16 scenes — not a significance claim. Nothing is wired into the runner
(§26).

---

## 20. A low-delay encode is prefix-stable to the byte, and that is the whole scheme

Everything BP30 claims rests on one property: **the bytes of frame *i* do not
change when frame *i+1* is appended.** If they did, the encoder would have
needed the future to emit scene *i*, and PointStream would be an offline
archiver rather than a codec.

Measured directly (ffmpeg n7.1.1-56-gc2184b65d2, `libaom-av1`,
`-usage realtime -lag-in-frames 0 -bf 0`, CRF 38, 960x540): encoding the first
2, 3 and 4 frames of the same sequence, then slicing each packet out of the
elementary stream by its `pos`/`size`:

| frame | prefix 2 | prefix 3 | prefix 4 |
|---|---|---|---|
| 0 | 95,240 B | 95,240 B | 95,240 B |
| 1 | 1,627 B | 1,627 B | 1,627 B |
| 2 | — | 82,239 B | 82,239 B |
| 3 | — | — | 93,940 B |

Not merely the same sizes — **byte-identical**, compared as bytes. This is §19's
causality result approached from the other side, and it is what lets a payload
be final the moment it is emitted.

**Two things this buys, and they are why the component is shaped as it is.**

1. **Encoder and client never have to agree about pixels.** A scene's payload is
   one frame's packet; the client reconstructs by decoding the payloads along
   that frame's chain. Both sides run the same decoder over the same bytes, so
   their reconstructions are equal by construction rather than by careful
   arithmetic. Drift is not mitigated here, it is unrepresentable.
2. **The growing re-encode is exact, not an approximation.** The brief called it
   "equivalent and much simpler"; prefix stability is why. `stream_linear`
   encodes a whole run once and reads every payload off it, which is what makes
   a 4K sweep affordable at all —
   `test_the_batch_path_agrees_with_pushing_scene_by_scene` asserts it produces
   the same bytes as pushing one scene at a time.

**It is re-checked, not assumed.** `_assert_prefix_stable` compares every
re-derived payload against what was already sent, on every real encode, and a
`B` picture type anywhere is refused outright. A flag existing is not a feature
working, and `-bf 0` is a flag.

## 21. The reference-mode axis is implemented as a prediction *tree*, not a flag

ffmpeg's CLI cannot signal "predict from an arbitrary earlier reconstruction",
so `first`, `last`, `best-scored` and `periodic-i` could not be four encoder
settings. Each scene instead records the **chain** — the path of scenes from its
root keyframe down to itself — and encoding that chain reproduces the reference
exactly. `last` makes the chain the whole prefix, which degenerates to the
ordinary growing stream; `first` makes every chain `(0, n)`; `periodic-i` starts
a new chain every *k* scenes.

**The cost of that choice, stated rather than buried.** The client must keep the
payloads along a chain, and reconstructing scene *n* re-decodes them. Under
`last` with no keyframes that is the whole stream, so a late joiner cannot start
and a lost payload breaks everything after it. Brief §3 accepts this for a paper
on a reliable channel — the point of sweeping *k* is to price the alternative
rather than hand-wave it. `BackgroundStreamReceiver` refuses a chain it did not
receive instead of returning a plausible wrong picture.

## 22. The background amortises: 16 scenes cost 62% of 16 fresh plates

Measured at **native 4K** (3840x2160), `alcaraz_highlights`, 16 point-class
scenes, one mid-scene frame each, `libaom-av1` low-delay P at CRF 38.
`outputs/bp30-background/stream-sweep.json`; bounds in
`outputs/bp30-background/bounds-before-run.json`, written before the first
encode.

All-intra baseline — each plate coded alone — is **9,136,924 B over 16 scenes**.

| arm | total | vs all-intra | marginal ratio | keyframes |
|---|---:|---:|---:|---:|
| `best-scored` | 5,417,958 B | **0.593** | 0.5745 ± 0.0185 | 1 |
| `last` | 5,696,800 B | **0.624** | 0.6072 ± 0.0152 | 1 |
| `first` | 5,756,103 B | **0.630** | 0.6110 ± 0.0097 | 1 |
| `periodic-i` k=8 | 5,921,602 B | 0.648 | 0.6129 ± 0.0158 | 2 |
| `periodic-i` k=4 | 6,429,221 B | 0.704 | 0.6160 ± 0.0203 | 4 |
| `periodic-i` k=2 | 7,206,260 B | 0.789 | 0.6046 ± 0.0291 | 8 |

**The marginal ratios land where §18 and §19 said they would** — 0.57 to 0.62
against that work's 0.470-0.671 on isolated pairs. The bound
`marginal_ratio_last` (0.20-0.85) held, and so did `sequence_ratio_vs_all_intra`
(0.30-0.90). The arithmetic reconciles: with r = 0.607 and one keyframe,
(1 + 15r)/16 = 0.632 against a measured 0.624.

**n is 16 scenes of one video.** That is not six videos, so none of this is a
significance claim in `presley`'s sense. The per-scene spread is reported so a
reader can see what the run can and cannot resolve.

## 23. Reference selection is worth about 6% in total, so `first` is the recommendation

> **Superseded by §29.** This section measured one video. At n=5 the
> recommendation flips: `first` is the *worst* free option, not a safe default,
> and `last` is the one to use. The reasoning below is kept because the error is
> instructive — the single video it rests on turned out to be the outlier.

The three reference modes code the same scenes, so they are compared **paired**
(`outputs/bp30-background/mode-comparison.json`,
`python -m experiments.tier.background_stream_compare`):

| comparison | paired difference | standard errors | finding? |
|---|---:|---:|---|
| `best-scored` vs `first` | 3.65 points cheaper | 2.6 | yes, just |
| `best-scored` vs `last` | 3.27 points cheaper | 2.5 | yes, just |
| `last` vs `first` | 0.38 points | 0.4 | **no** |

`last` and `first` are indistinguishable. `best-scored` is genuinely ahead, but
only just past this project's two-sigma bar — and the reason it cannot be far
ahead is the thing worth recording:

**The whole reference-selection axis is worth ~6%.** Scoring every candidate
reference for four targets by real trial encode
(`outputs/bp30-background/canny-validation.json`), the *worst* available
reference costs on average only **x1.063** of the best one (per target: x1.098,
x1.029, x1.040, x1.084). An oracle with perfect foresight could therefore win
about 6% over the worst possible choice, and `best-scored` captures roughly half
of that. Brief §3 set the criterion in advance — "if it agrees but wins by under
a few percent, `first` is still the recommendation and the search is complexity
for nothing" — and 3.65 points on a base of 61% is a few percent.

**Recommendation: `first`.** It is free, it is statistically indistinguishable
from `last`, and it forgoes 3.65 points against a search that costs one Canny
pass per candidate per scene, i.e. O(n) edge passes at scene n.

## 24. The Canny proxy does not track trial encodes well enough to trust

> **Qualified by §29.** At n=5 the proxy is weakly positive on average rather
> than broken, and on one video it is genuinely good. The conclusion that it is
> not worth searching with survives, but for a different reason: it does not
> beat the free baseline.

Brief §3 required the proxy be validated against real encodes before being
believed. It was, and it **largely fails**:

- **Mean rank agreement 0.31**, pooled **0.22**, over four targets with 4-7
  candidates each. Per target: 0.40, 0.20, **-0.14**, 0.79 — one target is
  anticorrelated and one carries most of the mean.
- **It picked the oracle's reference 1 target in 4**, costing on average
  **x1.021** against choosing by trial encode.
- **The edge maps barely overlap for any pair.** IoU ranges 0.034 to 0.091 with
  a mean of 0.049, so the score is discriminating between candidates that all
  look almost equally unlike the target. A proxy operating in that regime has
  very little signal to rank with.

The harness's own automatic verdict printed "Canny ranks references broadly as
trial encodes do", because the mean agreement of 0.311 cleared a threshold of
0.30 set before the run. **That verdict should not be quoted.** It is a
knife-edge pass, driven by one target, with another anticorrelated, and the
threshold was chosen without knowing how small the spread between candidates
would turn out to be. The threshold has deliberately **not** been retuned after
the fact — the raw numbers above are what the report stands on.

**This does not retract the reasoning behind the proxy.** §18's observation
still holds: the pair further apart in PSNR saved more, so pixel distance does
not predict coding distance, and an edge-based score is the right *shape* of
idea. What is measured here is that this particular Canny IoU, on this content,
is too weak to be worth a search — which is exactly why brief §3 asked for the
validation rather than assuming it.

## 25. A keyframe every *k* scenes costs what the arithmetic says, and every *k* still pays

The keyframe interval is swept as an axis, not imposed (brief §3). Against the
pure P-chain:

| *k* | total | vs all-intra | vs pure P-chain | keyframes |
|---|---:|---:|---:|---:|
| 2 | 7,206,260 B | 0.789 | **x1.265** | 8 |
| 4 | 6,429,221 B | 0.704 | **x1.129** | 4 |
| 8 | 5,921,602 B | 0.648 | **x1.040** | 2 |
| never | 5,696,800 B | 0.624 | x1.000 | 1 |

Monotone in *k*, as the bound required. **The pre-written bound was
conservative and that is worth recording**: it predicted break-even against
all-intra "around k = 2, so any k >= 4 should still pay". In fact *k* = 2 still
pays — 0.789, a 21% saving — because the marginal ratio is ~0.61 rather than the
~0.5 the bound assumed. Nothing was wrong with the measurement; the bound's
assumed *r* was pessimistic. The bound is kept rather than quietly widened.

**This is the number the paper's robustness paragraph needs.** A rate that
assumes no keyframes and no losses is a rate under a stated assumption, and a
reviewer will ask what dropping the assumption costs. The answer is now
quantitative: full random access every 2 scenes costs 26.5% over the pure chain
and still saves 21% against sending a fresh plate each time; every 8 scenes
costs 4%.

## 26. What was not done, and why

- **Nothing is wired into `make_background`.** `src/runner/stages.py` and
  `src/components/background/plate.py` belonged to a parallel stream this week,
  so the component and its measurement were built to stand alone. Integration is
  the follow-up, and it is the step that turns these numbers into a rate on the
  ladder.
- **No sidecar codecs were added**; the parallel intra-sidecar stream owns that.
  This module drives ffmpeg directly for its elementary streams.
- **The paired ladder was not re-run.** Out of scope by the prompt, and it should
  only be re-run once every lever has landed.
- **`background.reference-mode` and `background.keyframe-interval` were not
  added to `BackgroundConfig`.** They would be config that nothing reads until
  integration lands, and this project's own rule is that a flag existing is not
  a feature working. They belong with the wiring.
- **The plates are scene frames, not player-masked plates or stitched
  panoramas.** Deliberate, and conservative: findings §18/§19 measured on this
  kind of frame, so these numbers share their axis, and a frame containing
  players has *more* moving content to mispredict than a real background plate,
  not less.
- **One video, 16 scenes.** Not six videos, so no significance claim.

## 27. A correction to the brief, and to §2.21

Both say `BackgroundConfig.method` declares `panorama-delta` and "nothing
implements it". **It is implemented** — `src/components/background/strategy.py`
`PanoramaDelta` codes a signed pixel difference against the previous decoded
plate, via `src/components/background/delta.py`.

That matters because the mechanism it implements is **pixel subtraction** —
`compute_delta` is `current - previous + 128` clipped to uint8 — which is the
mechanism §17 measured and §18 retracted, and which §18 explained destroys the
spatial correlation a transform coder depends on.

**Two qualifications, so this is not overstated.** `PanoramaDelta` applies the
subtraction *within* a scene, between sub-chunks, which its own docstring says
is the case where plates drift rather than jump; §17's retracted measurement was
across scenes, which is harder. And a within-scene delta was never claimed to be
worth 31-53%. So this is not a second retraction.

What it is: **the slot BP30 was pointed at is not empty, and what occupies it
carries the name `delta` for the opposite idea.** Anyone wiring the stream in
should replace `PanoramaDelta`'s mechanism rather than register beside it, or
`background.method` will offer two strategies whose shared name means block-wise
inter prediction in one and pixel subtraction in the other. The uint8 clipping
limit in `delta.py` also simply disappears under the stream, which represents no
difference at all — it codes the picture, not the difference.

## 28. The 4K source duplicates frames, so "two consecutive frames" is not a control

The control very nearly reported a number that was flattering and meaningless.

Sampling two adjacent frames from `alcaraz_highlights` at 4K and coding the
second as a P-frame gave a ratio of **0.0002** — the P-frame costing 0.02% of a
fresh intra, against the 1.2-3.3% findings §18/§19 measured. That is *below* the
pre-written bound of 0.005, and the bound is what stopped it being written up.

**Cause: the two frames are byte-identical.** Same md5. The source is
60000/1001 fps carrying content shot slower, so frame *t* and frame *t+1* are
frequently the same picture. The encoder was being asked to predict an unchanged
frame, which it can skip almost for free. It measures that the encoder can
detect a duplicate, not that it can predict a changed picture — so it would not
have caught the kind of broken configuration §19 credits the control with
catching, which is the control's entire job.

**The fix, and why it is reported rather than silently applied.** `_control` now
walks forward to the first frame that genuinely differs from the first, and
records how many duplicates it skipped (`duplicate_frames_skipped` in the
results). If the source ever stops duplicating, that field changes and the
reader can see the control's meaning has changed with it.

**Re-measured with the fix: ratio 0.0074, one duplicate skipped.** That is
inside the bound, and the same order as the 1.2-3.3% §18/§19 measured on the
BP21 windows — lower, which is what a 4K intra baseline nine times larger than
theirs should give. The control is readable again, so the arms above are too.

**The general lesson, which is not specific to this dataset.** "Adjacent frames"
was treated as a synonym for "nearly identical but distinct pictures". On a
container whose frame rate exceeds the content's, it is a synonym for "the same
picture". Any control built from adjacent frames on this corpus needs the same
distinctness check — the BP21 cached windows that §18 and §19 used were
extracted at the dataset's own rate and did not have this problem, which is why
it appears now rather than then.

**Two bound breaches, opposite directions, both useful.** This one fired low and
found a broken control. The keyframe bound in §25 was too pessimistic and found
nothing wrong. Recording both is the point: a bound that only ever fires on bad
results is not being read honestly.

## 29. At five videos the recommendation flips: use `last`, not `first`

§§22-24 measured one video. That was not enough, and the single video chosen —
`alcaraz_highlights` — turned out to be the **least favourable of five**. Run on
four more (`experiments/tier/background_stream.py --video`, per-video results in
`outputs/bp30-background/stream-sweep-<video>.json`):

| video | kind | `last` | `best-scored` | `first` |
|---|---|---:|---:|---:|
| `djokovic_federer` | full match, 224 scenes | **0.294** | 0.307 | 0.454 |
| `federer_djokovic` | highlights | **0.448** | 0.455 | 0.512 |
| `sinner_alcaraz` | highlights | **0.471** | 0.484 | 0.552 |
| `alcaraz_highlights` | highlights | 0.624 | **0.593** | 0.630 |
| `alcaraz_perricard` | full match, 88 scenes | 0.624 | **0.616** | 0.634 |

**The amortisation is better than §22 reported.** Mean **0.492 ± 0.062** over
five videos, best case **0.294** — a 71% saving on the plate. §22's 0.624 was
the worst result available.

**`first` is the worst free option, not a safe default.** §23 recommended it
because on the one video measured it was indistinguishable from `last` (0.4σ).
Across five videos it loses to `last` on every one, by 6 to 16 points. That
recommendation was wrong and is withdrawn.

**`best-scored` does not beat `last`.** Paired over five videos the difference
is **-0.0012 ± 0.0083, i.e. 0.1σ**, and `best-scored` wins on 2 of 5. §23's
apparent 3.65-point win for `best-scored` was a win over `first` specifically —
which is to say it was measuring "use a recent reference", and `last` already
does that for nothing.

**So the conclusion of §23 survives with its reasoning replaced.** Do not build
the search — not because reference choice does not matter, but because the free
baseline already captures what the search finds.

**The proxy is content-dependent, and not along the axis expected.** Per-video
mean rank agreement: `djokovic_federer` **+0.69**, `sinner_alcaraz` +0.41,
`alcaraz_highlights` +0.31, `federer_djokovic` +0.21, `alcaraz_perricard`
**-0.14**. Pooled over 20 targets: **0.297 ± 0.122**, negative on 6, one pick
costing 30% over the oracle. So the proxy is weakly positive on average and
genuinely good on one video — the earlier "largely fails" was too strong at
n=1. But *full match vs highlights does not explain it*: the two full matches
are the best and the worst of the five.

**A confound checked and ruled out.** Each run takes the first 16 point-class
scenes, which span different wall-clock in different videos — so a video whose
scenes sit closer together could amortise better for that reason alone. Spans
are 275-526 s and `djokovic_federer`, the best result, is mid-range at 307 s.
The spread is content, not spacing.

**What still stands from §22.** The prefix-stability property, the bit-identity
of encoder and client reconstructions, and the keyframe ladder are unchanged —
they are properties of the scheme, not of the video. Only the reference-mode
recommendation moved.

**n is now 5 videos**, which meets `presley`'s n>=6 bar only approximately; the
per-video numbers are given so the spread is visible rather than averaged away.
