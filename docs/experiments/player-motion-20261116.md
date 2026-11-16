# Experiments: shared-tennis player motion pilot

### 2026-11-16 — human review imported; motion pilot prepared

**Prepared, not released. No new GPU work or training.**
The completed [human review](player-replay-20261116/human-review-import.json)
changes the next development priority. The [machine-readable proposal](player-motion-20261116/proposal.json)
contains the exact inputs, caps and unreleased request slots.

## Human finding and model decision

The user supplied a 659-byte export, SHA256
`ae6858db9874ded9ddc66338fec28e6014ecfdf6e39d713f814177a0b54af161`.
Its plan digest matches the retained pilot; every viewer asset matches the
retained manifest. Labels were unrevealed according to the export. Original
bytes were preserved with exclusive creation and readback in the dataset root;
the import record gives their immutable path. The export is review data, not
instructions, a compute release, or source-only annotation.

| Player | A / B / C | User decision | Reported observation |
|---|---|---|---|
| 1 | common tennis / match adapted / stock | Uncertain | Stock sharpens beyond the source; common tennis and match adapted look similar. |
| 2 | stock / common tennis / match adapted | B: common tennis | Stock renders the torso as facing away while turning the head toward the camera, despite the source player facing the camera. |

Use **common tennis as the provisional next development candidate**. No
additional match-specific state earned its cost in this tiny viewing pilot.
That is a prioritization decision, not evidence of equivalence or inferiority
of all adapted models. Preserve the stock structural failure and enhancement
artifacts; do not close every stock configuration or start a guidance sweep.
One author, two players and 24 frames from one match cannot establish population
preference, significance, a player-and-ball win or a useful codec rate.

The stock result is conditional on the retained body-only pose representation.
COCO-17 is mapped to OpenPose-18 with confidence-filtered nose/eye/ear and
side-specific body cues; dense face/hand landmarks are absent. The retained
upstream fork's normal DWPose drawing includes those landmarks and uses a
different detector/visibility convention. Its synthetic body-map audit and
W8 tensor parity establish neither actual Player2 cue accuracy nor the released
checkpoint's response to richer conditioning. No explicit inference RGB-swap
defect was found. Wrong/missing cues, ambiguity and guidance remain competing
explanations; the user observation does not identify their cause or close stock.

Fresh-pilot LPIPS/DISTS are **not scored**. Existing CPU-capable metric helpers
could compare the 144 source/candidate crop pairs, but there is no direct saved
crop scoring command, the inspected local environment lacks judge backbones,
and no aligned fixed source-player mask has been authenticated for that path.
Whole-crop metrics would include generated court texture before compositing.
A later scorer must pin/stage the existing AlexNet/VGG and packaged metric heads,
assert CPU/no-download behavior, benchmark actual cost and record per-tick scores
with that crop/proxy-mask scope. No metric disagreement is asserted or silently
added to this motion allocation.

## Hypothesis, competing explanation and scope

Hypothesis: the shared-tennis profile preserves body/clothing, facing direction
and useful motion over a complete 240-frame native run, including the sampler's
overlapping contexts. Competing explanation: the promising short crop hides
context-dependent hallucination, flicker, missed movement or mere sharpening.
Sharper pixels cannot compensate for incorrect player structure.

This extends an already qualified existing renderer and installed checkpoint;
it builds no new representation and proposes no claimed oracle bound. The
short pilot is admission evidence, not an upper bound on perceptual metrics.
Generate each run in **one Renderer call**. Do not concatenate independently
seeded 24-frame outputs. A 44-frame smoke crosses the single-context case;
the exact context schedule remains that of the pinned production sampler.
The 44- and 240-frame results need not share identical prefix pixels.

Only the already opened `nadal_kyrgios/000@120` source is released by this
proposal if approved: 1920x1080, 50/1 fps, source frames 0-239, 4.8 seconds.
Two singleton configurations retain the previously authenticated per-player
conditioning bytes. The harder camera-facing **Player 2 runs first**, then
Player 1 only after accepted output/cost checks and sufficient remaining budget.
This order uses the development review explicitly, before longer-run outcomes.

Original weights, common checkpoint `85e3c65c...39eb4`, 512x784, 30 DDIM steps,
CFG3.5, context24/overlap4, seeded inputs and preprocessing remain fixed. No
match adapter or guider update is loaded. Uncoded source references and
lookahead geometry remain granted; this is a crop renderer qualification.
The native base, paid references/poses/placement, ball stream, receiver masks
and complete source-denied learned decoding remain the subsequent codec test.

## Exact preparation and native fleet contract

Schema 1 retains its original 24-frame behavior. New explicit schema 2
(`player-motion-development`) permits only the declared 44/240 native lengths
and the `tennis_only` profile. The worker checks all selected consecutive ticks,
run IDs, arrays, hashes, resources and every output frame. The versioned schema
prevents historical receipts from silently becoming longer-run evidence.

The [two configs and inventory](player-motion-20261116/proposal.json) reuse the
original hashes and packed environment, removing the unused match archive.
`player_requalification_specs --motion` prepares singleton requests without
submitting them. Actual release must freeze the checked code revision, publish
and read back immutable config bytes, set a fresh deadline, inspect existing
jobs/all eligible hosts, and use the compatible Ada pool. No stale job is replayed.

Each request uses `single_cpu_campaign`, one CPU core, 18,000 MiB GPU memory
plus the existing sharing margin and 48 GiB local staging declaration. Changed
lengths require the actual smoke; old timing is a feasibility lead, not proof.
Shared-GPU time supports budget admission only, not throughput or real time.

Actual CPU input checks pass for both 44/240-frame selections and preserve the
old schema-1 plan digest. [The current labeled-input audit](player-motion-20261116/cpu-input-audit-labels.json)
matches the final config hashes; the earlier `cpu-input-audit.json` retains
pre-label preparation, not a sampled result. The [source conversion audit](player-motion-20261116/source-audit.json)
also processed all 240 source frames for each player with zero reference RGB
delta: 21.84 s wall, 20.68 s parent CPU, 2.69 GB parent peak RSS. These are
source-only checks; they do not qualify generated motion or full display export.

## Proposed allocation and cost gate

**Approval required: at most 4,200 GPU-seconds (70 GPU-minutes) and 7,200
CPU-seconds (2 CPU-hours), no training.** This is a new allocation; the completed
replay's unused time is not transferred.

| Per-player request | Hard cap |
|---|---:|
| Real 44-frame, 30-step smoke | 450 s |
| One whole 240-frame run | 1,500 s |
| Smoke validator | 60 s |
| Total request lifecycle, including overhead | 2,100 s |
| External CPU preparation, audit and source-aligned review | 1,500 CPU-s |

Two request lifecycle caps total 4,200 GPU-s and at most 4,200 CPU-s under
the single-core campaign. External CPU totals at most 3,000 s, giving 7,200
CPU-s overall. Every request ends within 35 minutes; each external CPU phase
ends within 25 minutes. CPU/viewer work must also respect a 45-minute wall
checkpoint, retain finished items and stop when its allowance is exhausted.
The two requests generate 568 crop frames including smokes (not 568
independent observations).

Recent common-tennis 24-frame render times were 74.17/83.15 s plus 23.96 s
model load in the full worker. They cannot be linearly promoted as a runtime
guarantee for changed context. The schema-2 smoke validator uses actual worker
wall/load times: `1.25 * (load + (wall - load) * 240/44) + 90` seconds.
It admits the full run only when this is no more than 1,500 s and no more than
the native campaign's current remaining time minus a 60 s close allowance.
Missing/nonfinite clock or timing evidence fails closed. This conservative
heuristic and hard deadlines bound spending; they do not prove full-run cost.

Stop on failed source/device/kernel/output/cost gates or exhausted allocation.
No automatic retry, parameter change, new fitting, substitution or extra source.
Inspect unknown submissions before any action. Preserve partial and failed data.

## Viewing and decision

Authenticate the actual sealed publication, whole run and source bytes. The
viewer reproduces the original RGB conversion and exact reference geometry;
it retains all 240 frames and native 50 Hz, with optional explicitly slowed
playback. A loop seam is not a temporal measurement. Source decoding/viewing
requires several GiB of local RAM. Source-only resource use is measured above;
full generated-media viewer export must be measured within the external allowance
before a delivery promise. Each singleton keeps its original authenticated
player label so Player 2 cannot be renumbered as Player 1 in a later review.

This is an **unpaired source-versus-candidate acceptability review**, not a
ranked comparison. Ask whether facing direction/anatomy, pose, clothing and
stability are acceptable; retain uncertainty and report failures with their
ticks. Racket improvement remains optional and harms remain visible. Do not
infer preference over native/ROI/DCVC without showing the paid matched controls.

If useful motion survives, prioritize the minimal paid learned receiver and
matched player-only/ball-only/joint three-rate screen. If structure or motion
fails, identify the limiting conditioning/context assumption before a narrow
repair; neither another broad model grid nor training is automatically released.
Retain all failures alongside narrowed claims and use source-level uncertainty
once genuinely separate sources are evaluated; no arbitrary 10% floor applies.

## Next source and submission boundary

`alcaraz_hurkacz/036@120` is a development candidate from preparation job
`20261022T123413Z-eba7629b`, archive SHA256
`f832b70720acb3704fa83a6b7484fd18f97ef53d004e19452a528f075d737b8c`.
Units end `person_000000000006` and `person_000000000002`. Actual source bytes,
source/window/PTS, member hashes/contiguous runs and training/duplicate exposure
still need authentication. It is **outside this release**. The historical
25 fps declaration is not enough to invent a source join. The shared model
has same-match exposure from other Al windows; this is not clean confirmation.

The saved W6 Al031 full-rate videos are historical confirmation-cohort evidence
with a different match-adapted/CFG2/reference recipe. They are not newly unopened
development and cannot settle the shared-tennis comparison. Retain their
temporal failure, without opening a new confirmation cohort for convenience.

No combined codec result, confirmation, extra sport, model training, rights
approval or paper filing is released by this proposal. The December19 research
cutoff and December26 preparation target are unchanged.
