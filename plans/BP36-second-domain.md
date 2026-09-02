# BP36 — The second domain, and a dataset a reviewer already knows

**Gated (2026-09-02): do not start until Gate B in `plans/ROADMAP.md`.** The
first domain must produce and confirm a credible winning regime before time is
spent on a second domain. This brief remains the specification for that later
work.

**`PLAN.md` §7 P0 item 6, `HOLE(subsec:eval-general)`, and the first of the three
items in `NEXT(paper-wide)`** — which records a second evaluation domain as *the
most-requested item* a referee is likely to raise. Nothing has been run in either
domain.

**Owns:** `src/components/domain/**`, `src/components/domain/datasets/*.yaml`,
`outputs/bp36-general/**`, a new `datasets/skating.yaml`, `sections/evaluation.tex` §`subsec:eval-general`.
**Does not own** `experiments/tier/**` or `src/runner/**` unless a later dispatch
explicitly grants them.

**Read first:** `AGENTS.md` · `PLAN.md` §7 P0 item 6 and §4 ·
`sections/evaluation.tex` `GOAL/HOLE/NOTE(subsec:eval-general)` ·
`src/components/domain/profiles.py` and `datasets/general.yaml`.

**Result dependency:** Gate B, a confirmed first-domain win.

---

## 0. What is already there, and what is not

The `general` profile is **registered and never driven**:
`src/components/domain/profiles.py:build_general` exists, `datasets/catalog.py`
lists `("tennis", "general")`, and `datasets/general.yaml` points at
`assets/davis` and `/home/itec/emanuele/Datasets/DAVIS`. That directory is real
and holds the standard DAVIS sequences (`blackswan`, `bmx-trees`, `breakdance`,
… at 50 frames each).

**A flag existing is not a feature working.** Before anything is planned around
this profile, drive it: construct the backend, run one clip end to end, and
confirm the output changed in the way the profile claims — every detected person
salient, no tennis rules. The tennis profile's court and player heuristics are
threaded through more of the pipeline than the registry suggests, and the honest
expectation is that the first `general` run fails somewhere specific. Find where.

## 1. The domain order, corrected 2026-09-02

An earlier version of this brief put **UVG** first, on the reasoning that a
standard sequence answers "was the regime found or built?" in a way more
self-curated tennis cannot. **That was the wrong call and is withdrawn as the
lead.** The ordering below is the project's standing plan — solve tennis, then
DAVIS, then one more simple sport — and it is better on the merits:

| # | Domain | Data | Why here |
|---|---|---|---|
| 1 | **Tennis** | `assets/raw_4k/`, 7 matches, 4K | in progress; the scope the project already chose, and large enough to be a paper on its own |
| 2 | **DAVIS human sequences** | `/home/itec/emanuele/Datasets/DAVIS`, 50 frames each, annotated | what `general.yaml` already names, what `subsec:eval-general` is written around, and it has ground-truth masks so the perception control is checkable rather than assumed |
| 3 | **Figure skating** | `EvgeniaMedvedeva2018` (23), `ShomaUno2018` (17), `YuzuruHanyu2018` (16) — 56 clips, 1920x1080, 25 fps, ~42 frames each | already on disk. One or two people on a large, bright, near-static rink, with a panning camera: **the claimed regime, in a second sport** |
| — | UVG `Jockey`, `ReadySteadyGo` | `/home/itec/emanuele/Datasets/UVG/1920x1080` | optional, and only with the caveats below |

**Why skating is the better third domain than UVG.** The pipeline is
person-centric — detector, pose, segmenter and tracker are all built around
people — and skating gives it exactly that: one or two humans, whole-body,
against a background with a larger static fraction than tennis has. It is the
*same* claim in a *different* sport, which is what a generalization section
needs. The clips are ~42 frames, close to the 48-frame windows the tennis cache
uses, so `plans/BP33-span-amortisation.md`'s span conclusion transfers.

**And why UVG is not the free win it looked like.** Both locally available
sequences are equestrian — a jockey on a horse, and horses at a gate. The salient
object is a horse plus rider, which the person pipeline handles poorly, so a bad
number there would measure the detector rather than the codec. They are also
1920x1080 at 120 fps against the tennis 4K, which confounds resolution and frame
rate with domain. Only 2 of UVG's 16 sequences are on disk.

Its one real advantage stands: referees know these clips and published anchors
exist for them. So keep UVG as an **optional late addition** — run it if there is
time after skating, state the resolution and frame-rate difference, and do not
let it carry a generalization claim it is not shaped to carry.

## 2. What the section must show, and the constraint on it

`GOAL(subsec:eval-general)` asks the architecture to **degrade gracefully**
outside its training domain, not to win there. `NOTE(subsec:eval-general)` records
that **Animate-Anyone has seen the held-out videos**, so any engine-dependent
number in this domain carries a contamination caveat.

Since no engine beats a pasted keyframe anyway (`PLAN.md` §2.10), the clean shape
for this section is **the codec claim without a generative engine**: plate plus
warps plus pasted crops plus residual, against the same anchor, on content the
project did not curate. That has no contamination problem at all, and it is the
configuration the rest of the evaluation is being built around.

## 3. Bounds — `outputs/bp36-general/bounds-before-run.json` before the first run

Two-sided, and note that the *upper* bound matters here as much as the lower:
this is a different domain, and a result that is suspiciously close to the tennis
result is as much of an alarm as one that is far away.

- **BD-rate on a skating clip against av1, same protocol as the tennis ladder:
  within [0.5x, 3.0x] of the tennis number** at the same span and scene count.
  Skating is the *friendlier* case — a larger static background fraction — so
  meaningfully worse than tennis is an alarm, and the first thing to check is
  whether the perception stage is failing on non-tennis content and the number is
  measuring a broken detector rather than a codec.
- **On DAVIS, expect worse than both.** Handheld, free-moving cameras break the
  homography model the plate depends on. A DAVIS number close to the tennis one
  would mean the domain profile did not switch. This section's `GOAL` asks for
  graceful degradation, not for a win.
- **Person detection recall on DAVIS human sequences ≥ 0.7** against the provided
  annotations. Below that, the pipeline is not seeing the objects and no rate
  claim from it means anything. This is the control, and it runs first.
- **The all-off corner reproduces the source** in the new domain too. If it does
  not, the failure is `DEFERRED.md` D5 (see `BP39`) surfacing in a second place.

## 4. The trap

**Do not report a domain result before the perception control passes.**
`AGENTS.md`: when a component underperforms, check it is being invoked the way
its architecture intends — a temporal video model was evaluated one frame at a
time here for three rounds. A codec BD-rate on a clip where the detector found
nothing is a measurement of the background model alone, wearing the name of the
whole system.

## Done when

- The `general` profile is driven end to end and the failures it exposes are
  fixed or recorded.
- A perception control (detection recall against annotations) passes on the
  chosen sequences, and is reported beside the rate result.
- At least three DAVIS sequences and three skating clips carry a BD-rate against
  the same anchor and protocol as the tennis ladder, reported per sequence with
  the spread — **and with encode and decode time in the same table**, per
  `AGENTS.md`'s three-dimension rule. UVG is optional and late.
- `HOLE(subsec:eval-general)` is cleared by the edit that lands the data, and
  `NEXT(paper-wide)`'s "second evaluation domain" item is struck.
