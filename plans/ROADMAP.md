# PointStream submission roadmap — hard deadline 30 September 2026

**This is the authority on priority, dependencies, session assignment and
submission scope.** `PLAN.md` is the current summary; the chronological record
is `done/RESEARCH-HISTORY.md`. A work
session reads `AGENTS.md`, this file, and one named brief.

Last reconciled: **5 September 2026**, BP56/BP57 review and integration.

PR #52 integrated BP44–BP46, #53 repaired recovery, #55 added reference pilot
controls. BP49/BP52 passed the native checkpoint budget only on the short
diagnostic pair. CRF-only search established no win. BP51 accepts zero untouched
confirmation matches; all current sources have prior use. PRs #57/#58 include
review repairs for fail-closed provenance and batch-stop behavior.
PRs #60/#61 merged. BP53 half-scale diagnostic established no win; its driver
identity crossing and historical timing lower bound remain explicit. Later
standalone-client/budget checks are not native reruns. Longer runs are uncleared.
PRs #63/#64 merged at `60a18f7`. BP56 found one PointStream-internal dominant
candidate on a single development pair; it is a search seed, not an anchor win.
BP57 acquired two provisional fresh sources with seven visually checked long
shots; neither source can confirm a 4K-specific claim. **Next: design and run a
bounded Gate-A long-context/coherent-rate search with same-identity AV1/VVC
curves, while preparing fresh-source eligibility for Gate B.** Separate
PointStream encode/decode timing remains open. `PAPER-NEXT.md` is the writing
brief; `SUBMISSION-READINESS-2026-09-05.md` is the current blocker audit.
The workstream table below describes dependencies, not a claim that every row
is still unimplemented. Archived reports are not current launch instructions.

## 1. Submission thesis and non-negotiable core

The paper keeps this core idea:

1. split a video into scenes;
2. identify scenes with low background motion and separable foreground objects
   from known classes;
3. send a reusable background representation for an eligible scene or compatible
   group of scenes;
4. send each object's appearance rarely and its motion compactly;
5. reconstruct those scenes at the receiver, with an optional correction signal;
6. encode ineligible scenes conventionally and count the routing metadata.

Optional modules do not get protected from evidence. Generation, residuals,
per-frame crops, background stitching, or any other module may be disabled or
removed from the winning configuration when its quality gain does not earn its
bytes. The scene router and fallback preserve the paper's hybrid construction.

The architecture creates an opportunity to remove repeated pixels; it does
**not** guarantee a bitrate win by construction. Background references, object
appearance, correction data and metadata can cost more than the redundancy they
replace. The paper earns the claim by measuring the regime where the opportunity
becomes a win.

## 2. Optimization order and evidence gates

### Gate A — find a rate--quality win

First beat **both AV1 and VVC** on size at matched quality in a named eligible
regime. Encode and decode time are measured and reported from the first run, but
time is not a pass condition for Gate A.

A Gate-A result is one of:

- negative BD-rate over a meaningful overlapping range on the predeclared
  headline quality metric; or
- strict dominance at the low-rate boundary: PointStream uses fewer bits than
  the anchor's smallest decodable point while meeting or exceeding that point's
  measured quality.

One point at unmatched quality, a projected crossover, or a result on a metric
chosen after seeing the curves does not pass.

### Gate B — confirm the first-domain result

Freeze the discovered content, rate and scene-length criteria before expanding
the sample. Confirm on at least six independent source videos/matches from the
tennis domain, reporting per-video results, mean, standard error, failures and
the search that found the regime.

### Gate C — explain the win

Run the **core component ablation matrix**: conventional fallback, background
reuse only, +object appearance/motion, +correction, and +generation if generation
has become useful. A component is justified only if the rate--quality curve
improves. Full sixteen-axis enumeration is not required for submission.

### Gate D — broaden the comparison

Only after Gate B:

- add DCVC-RT as the primary learned-codec baseline;
- test one independent public domain/sequence with the same frozen criteria;
- profile and optimize the frozen configuration's speed.

DCVC-RT is selected because its official implementation provides checkpoints,
actual bitstream writing, a wide rate range, YUV420 evaluation and encode/decode
timing. DCVC-UF is a stretch baseline only if it drops into the same harness
without delaying the paper.

### Gate E — freeze evidence and write

The evidence freeze is **20 September**. Later runs may close an alarm or repair
an invalid result; they do not open a new architecture campaign.

## 3. First search: the regime most likely to win

The first search is deliberately narrow:

- **Content:** long, mostly static or smoothly panning broadcast-tennis scenes;
  one stable camera context; small, well-separated players.
- **Rate:** the ultra-low-rate range, below the current ~43 dB operating point.
- **Duration:** 2, 4, 8 and 16 seconds (48, 96, 192, 384 frames at 24 fps),
  extended to 32 seconds only when the pipeline remains stable.
- **PointStream payload:** aggressively compressed background, one appearance
  reference per track unless refresh is measured necessary, sparse motion,
  correction off or coarse, generation off initially.
- **Anchors:** the same source frames, resolution, frame rate, colour convention
  and temporal extent, encoded with AV1 and VVC using each installed encoder's
  slowest valid preset or full reference configuration. If PointStream later
  encodes faster, repeat with faster reference presets to map the
  rate--quality--time tradeoff.

Forty-three dB is a high-fidelity signal regime in which conventional codecs are
strong and PointStream's structural overhead is exposed. The low-rate hypothesis
is therefore credible. It is still only a hypothesis: AV1 and VVC have finite
syntax/tool floors at fixed resolution and frame rate, but there is no universal
minimum bitrate that PointStream automatically undercuts.

### 3.1 Quality axes

Declare these before the first search run:

- **Primary:** full-frame VMAF, because the hypothesis concerns visible quality
  under severe compression.
- **Secondary:** Y-PSNR and MS-SSIM/SSIM, reported on every point.
- **Diagnostics:** foreground-region VMAF/PSNR, background-region quality,
  temporal quality by frame index, and object identity/pose checks.

LPIPS is diagnostic until its lower-is-better direction is represented correctly
in the curve code and its absolute scale has passed calibration. A foreground
metric cannot silently replace full-frame quality after the search. If a
salient-object quality thesis becomes necessary, its utility rule and background
quality floor are written before that experiment.

### 3.2 Codec-floor calibration

Before the system sweep, establish the usable range of the installed AV1 and VVC
encoders:

1. resolve binary path and version;
2. enumerate and verify the available presets; use the slowest valid preset for
   the primary comparison and record every tool/config choice;
3. probe the documented/legal QP or quality range;
4. verify every output is non-empty and decodable;
5. check bitrate and each quality metric are sufficiently monotone;
6. retain the lowest decodable point even when it falls below the BD-rate
   overlap;
7. use at least four useful points per curve and add points around a crossover.

Run two temporal-access controls. The **segmented control** encodes every point
scene independently and therefore pays an intra frame per segment. The
**continuous control** gives AV1/VVC the same ordered eligible scenes and permits
inter prediction across the same boundaries where PointStream reuses a
background context. The headline comparison must use the control matching the
claimed product; PointStream may not claim a saving created only by forcing the
reference codec to reset more often than itself.

Do not downscale, drop frames or alter frame rate in the main comparison. Those
are separate operating profiles and must be offered to both arms.

### 3.3 A coherent PointStream rate sweep

One sweep setting must move every rate-bearing channel coherently:

- background resolution and background codec quality;
- appearance resolution, codec quality and refresh interval;
- motion density/precision;
- correction presence, resolution and codec quality;
- unavoidable metadata.

The current ladder sometimes moves the correction setting while freezing the
largest background term. The new sweep asserts that the intended byte categories
change monotonically or records why they do not.

## 4. Long scenes and compatible background canvases

The current failure is understood. Each scene builds a background panorama in
its own local coordinate frame and chooses dimensions from that scene's camera
motion. At 24 frames, the two tested scenes produce different image sizes.
The AV1 background-sequence encoder requires every image in one sequence to have
the same dimensions.

Padding only to the same width and height is sufficient for the video encoder,
but not necessarily optimal or geometrically correct. The implementation must
also preserve the canvas origin and update every scene-to-canvas transform.

### 4.1 Deadline implementation: offline canonical canvas

For every group of compatible scenes:

1. assign a **background-context ID** (same camera/view/venue background);
2. precompute the union of scene homography bounds in a canonical coordinate
   system;
3. allocate one canonical canvas size and origin for the group;
4. render or pad every scene background into that coordinate system;
5. adjust reconstruction transforms for the shared origin;
6. encode the resulting background images as one predictive sequence;
7. reset with a new independently coded background when the context changes;
8. measure padding bytes, prediction gain and reconstruction equality.

The prepass sees future scenes, so this is an **offline or buffered** codec mode.
It cannot support a “live” title. A causal extension may later use a fixed
profile-sized canvas and reset when motion leaves it, but that is not on the
September critical path.

Tests must cover unequal local canvas sizes, static+panning scenes, transform
adjustment, context reset, sender/receiver equality, causal payload accounting,
and a control showing that shared-background coding changes bytes.

### 4.2 Long-scene experiment

After the fix, run 48/96/192/384 frames on at least:

- a near-static eligible scene;
- a smooth-pan eligible scene;
- a deliberately ineligible high-motion scene that should route to fallback.

Measure total and per-frame bytes for background, appearance, motion, correction,
metadata and fallback. Fit fixed cost plus per-additional-frame cost only after
at least three successful durations; call it the **per-additional-frame payload
slope**, not a marginal estimate. Report uncertainty and do not infer an
asymptote from the former two-point fit.

## 5. Payload simplification order

If the first low-rate/long-scene curve still loses, remove cost in this order,
re-running a small diagnostic curve after each step:

1. attribute every byte and eliminate duplicated accounting;
2. turn off whole-frame correction and measure whether the quality loss earns
   the saved bytes;
3. stop sending repeated object crops; send one appearance reference plus sparse
   refreshes only when measured necessary;
4. reduce background spatial resolution and codec quality jointly;
5. reduce appearance resolution/quality;
6. reduce motion precision and sampling;
7. simplify metadata and container overhead;
8. revisit generation only when a model improves quality per transmitted byte
   over the reference-image paste control.

The lean candidate is expected to be background + one appearance reference +
motion + optional sparse correction. The residual is standard terminology and
remains optional; it is not assumed beneficial because one short-scene point was
favorable.

## 6. Decision dates and calendar

| Date | Required outcome | If missed |
|---|---|---|
| **2–3 Sep** | Branches reconciled; one roadmap; terminology; paper renders | Stop other work until repository state is trustworthy |
| **4–6 Sep** | Codec-floor calibration, typed quality axes, payload ledger, canonical-canvas implementation | Use independent backgrounds for diagnostics, but do not publish a long-scene conclusion |
| **7–9 Sep** | Low-rate × long-scene search on two diagnostic tennis videos | Invoke the lean-payload simplification order immediately |
| **10 Sep** | Gate-A candidate or measured near-crossover | Escalate to the salient-object-quality thesis below; no generator campaign yet |
| **11–14 Sep** | Gate B: first-domain confirmation on ≥6 videos; configuration frozen | Narrow the eligible regime rather than adding features |
| **15–18 Sep** | Core ablation matrix, DCVC-RT, one second-domain check | Cut non-load-bearing component experiments and appendix material |
| **19–20 Sep** | Final validated tables/figures; all result alarms closed; evidence freeze | Only correctness repairs after this point |
| **21–24 Sep** | Full manuscript rewrite, artifact instructions, supplementary video | Freeze code and move all available effort to the paper |
| **25 Sep** | Complete coauthor draft in exact ACM submission format | Remove whole secondary results, not prose fragments |
| **26–27 Sep** | Scientific audit, page-budget pass, anonymous-package rehearsal | Scope claims down to evidence; no new claims |
| **28 Sep** | Submission candidate frozen | 29–30 Sep are upload/metadata/emergency buffer |
| **30 Sep** | Submit | Hard deadline |

### 6.1 Thesis fallback, without abandoning the core idea

If full-frame VMAF/PSNR still has no crossover by 10 September, freeze a second
question before measuring it:

> At the same ultra-low bitrate and with a fixed minimum background-quality
> floor, does the hybrid object-centric route preserve salient-object quality
> better than AV1 and VVC?

This remains a rate--quality comparison and keeps scene routing, reusable
backgrounds, appearance and motion. It changes “quality” from uniform pixel
fidelity to a declared multimedia utility: full-frame quality, foreground
quality and a background floor reported together. The paper must report that
this thesis was activated after the full-frame search and must show both results.

### 6.2 Transparent constraint-relaxation ladder

If the primary claim still does not win, relax constraints in this order. Keep
every earlier comparison in the paper, name the exact tier that wins, and state
that each relaxation was chosen after observing the previous negative result.
A narrower honest claim is acceptable; relabelling a weaker comparison as the
original claim is not.

1. **Narrow eligible content further.** Require longer stable-camera shots, a
   known court view, small separated players, bounded pan, and enough repeated
   background for fixed costs to amortise. Report acceptance rate and fallback
   cost on the full video so this does not become invisible cherry-picking.
2. **Change the operating profile symmetrically.** Test longer buffers, lower
   resolution, lower frame rate or a lower quality range only when PointStream,
   AV1 and VVC receive identical source frames and timing. The title and claim
   name that profile; it is not a native-4K result.
3. **Use the declared salient-object utility.** Compare foreground quality at
   matched total rate with a fixed minimum background-quality floor, while also
   reporting full-frame quality. This is the preferred thesis change if uniform
   VMAF hides the benefit to small salient objects.
4. **Relax anchor speed.** If PointStream loses to the slowest AV1/VVC presets,
   compare against successively faster documented presets. A win licenses only
   “beats AV1/VVC at preset X,” not “beats AV1/VVC.” Keep the slow-preset curves
   as current-performance headroom and quantify the remaining gap. Because
   PointStream may still be slower, do not describe this as a speed-matched
   comparison unless measured wall time actually matches.
5. **Add older conventional anchors.** HEVC, VP9, AVC or another reproducible
   older codec may define a valid lower bar. A win against one is publishable as
   a scoped systems result only if AV1 and VVC remain visible as stronger modern
   anchors and the paper does not imply state-of-the-art compression.
6. **Relax unseen-video confirmation.** The preferred split is at least seven
   training videos and six untouched test videos. If model quality is inadequate
   or acquisition is infeasible, train on the available videos and hold out new,
   non-overlapping contiguous scenes or points from those same videos. Freeze the
   split before training, prevent adjacent-frame leakage, keep the test scenes out
   of tuning, and call the result **within-video held-out-scene evaluation**, not
   unseen-video generalization. Report both this result and any smaller clean
   video-level holdout.
7. **Reduce confirmation breadth.** If six independent videos are infeasible,
   report the available n, every per-video result, paired uncertainty and failure
   rate. Fewer than eight items is underpowered and cannot license a broad
   population claim, but may support a case-study claim.
8. **Scope the system result to eligible scenes.** Demonstrate the codec win on
   the route it was built for and separately measure how the conventional
   fallback affects whole-program rate, quality and time at observed eligibility
   rates. Do not claim the eligible-only curve as full-video performance.
9. **Consider content-adaptive training only with transmitted cost.** Per-video
   or per-venue adaptation is a separate codec profile. Count model updates or
   state the number of videos/hours over which a fixed model is amortised; never
   treat decoder side information as free.

The following do not relax: identical decoded source input for all arms; real,
decodable bitstreams; complete payload and model-update accounting; calibrated
quality instruments and nulls; continuous/segmented anchor parity; reporting
all attempted regimes; provenance and citability checks; and explicit disclosure
of training overlap and post-hoc scope selection. No sequence of defensible
relaxations can guarantee a win. If none wins, the paper must report the measured
boundary or change its central claim rather than manufacture one.

## 7. Workstreams, harness assignment and expected reports

High-level analysis, measurement design, architecture decisions, alarm
adjudication and final scientific writing go to **Codex**, or to **Claude** if it
becomes available. Routine implementation, bounded refactors, tests, extraction,
batch execution and mechanical paper updates go to **Cursor** or **VS Code with
Antigravity**. Long runs use detached shell jobs with hourly checkpoints.

| ID | Workstream | Depends on | Preferred harness | Required report |
|---|---|---|---|---|
| S0 | Merge active work; archive/delete stale branches | — | Codex | branch audit, archive tags, PR/CI links, final worktree list |
| S1 | Terminology and status reset | S0 | Codex | old→new term map; files changed; unresolved ambiguities |
| M1 | Quality-axis typing and ultra-low anchor probe (`BP45`) | S0 | Cursor | tests; paths/versions; AV1/VVC usable ranges; curves; alarms |
| B1 | Canonical background canvas and context resets (`BP44`) | S0 | Cursor, reviewed by Codex | design note; tests; byte/reconstruction control; PR |
| M2 | Payload ledger and duration-slope analysis | M1, B1 | Antigravity/Cursor | per-category bytes; ≥3-duration fit; uncertainty; alarms |
| D1 | Extract/validate long eligible tennis scenes (`BP46`) | S0 | Cursor | manifest, eligibility features, hashes, failures; no result claim |
| E1 | First-domain low-rate search (`BP45`) | M1, B1, D1 | batch job; Codex adjudicates | preregistered bounds; all tried configs; rate/quality/time; decision |
| E2 | Six-video first-domain confirmation | E1 | batch job; Codex adjudicates | frozen rule; per-video curves/spread; null; citability verdict |
| A1 | Core component ablation matrix | E2 | Cursor + batch job | BD-rate contribution per component; time; interactions |
| L1 | DCVC-RT baseline in isolated env | E2 | Cursor | upstream revision/checkpoints; bitstream validation; curves/time |
| G1 | Second domain | E2 | Cursor + batch job | frozen criteria applied unchanged; successes/failures |
| T1 | Profile and optimize frozen winner | E2 | Cursor | stage profile; before/after rate/quality/time; no claim drift |
| P1 | Paper build, page metrics and figure skeleton | S0 | Antigravity | reproducible PDF command; page/float/word report; placeholder map |
| P2 | Results and thesis rewrite | E2, A1 | Codex/Claude | marker sweep; CLAIM paths; title/abstract/conclusion; claim audit |
| P3 | Reproducibility package and supplementary video | E2 | Cursor/Antigravity | clean-machine instructions; legal data statement; asset manifest |
| P4 | Final scientific/submission audit | P2, P3 | Codex/Claude | page limit; citations; alarms; anonymization; upload checklist |

No session receives more than one row. Each row gets its own branch and worktree.
The complete report contract is in `plans/SESSION-REPORT.md`.

## 8. Result integrity

Every published result record includes:

- git commit, exact command/configuration, input manifest and sample count;
- encoder paths, versions, presets, colour format, resolution, frame rate;
- rate, every declared quality axis, encode time and decode time;
- per-video values, mean, standard error and curve-overlap range;
- pre-run bounds, null controls, open alarms, closed alarms with reasons;
- byte ledger by background, appearance, motion, correction, metadata/fallback;
- a machine-readable citability verdict.

The invariant suite must fail for a result marked citable while carrying an open
alarm. Printing uncitable runs without asserting is a dashboard, not a gate.

## 9. Paper and artifact scope

- Remove “Live Video Streaming” unless a causal, measured profile exists.
- Frame the primary system as offline or buffered when it uses a run-wide canvas.
- Replace promises of generative quality with measured findings. Generation can
  remain an optional architecture point and a negative result.
- Replace “dataset will be released” with a legally conservative statement.
  YouTube-derived video is not redistributed. Release code, configs, hashes,
  annotations only when permitted, source identifiers/timestamps when allowed,
  and rights-cleared or synthetic examples.
- Use the ACM manuscript format and stay within TOMM's 23-page main limit plus
  five appendix pages. Render continuously; do not postpone page counting.
- **Measured 2 September:** the reproducible manuscript build is 30 pages.
  Main text plus references end on page 21; appendices occupy pages 22–30.
  The main has only two pages of room before results are completed, and the
  appendices exceed their five-page allowance by four pages. Cut appendix
  survey/history first and reserve main-body room for the result figures.
- Main-body evidence needs at least: architecture, regime-selection diagram,
  rate--quality curves, duration/amortization plot, payload breakdown, ablation
  table, and rate/quality/time comparison.

## 10. Deliberately parked until Gate B

- training or tuning generative models;
- broad second-domain work;
- speed optimization;
- the complete sixteen-axis experiment space;
- causal/live background-canvas growth;
- DCVC-UF in addition to DCVC-RT;
- MOS/user study unless coauthors make it a submission requirement.

These are not declared unimportant. They are parked because none can rescue the
submission before the first-domain rate--quality question is answered.
