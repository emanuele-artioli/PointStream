# BP36 — The second domain, and a dataset a reviewer already knows

**`PLAN.md` §7 P0 item 6, `HOLE(subsec:eval-general)`, and the first of the three
items in `NEXT(paper-wide)`** — which records a second evaluation domain as *the
most-requested item* a referee is likely to raise. Nothing has been run in either
domain.

**Owns:** `src/components/domain/**`, `src/components/domain/datasets/*.yaml`,
`outputs/bp36-general/**`, `sections/evaluation.tex` §`subsec:eval-general`.
**Does not own** `experiments/tier/**` or `src/runner/**` while PR #45 is open.

**Read first:** `AGENTS.md` · `PLAN.md` §7 P0 item 6 and §4 ·
`sections/evaluation.tex` `GOAL/HOLE/NOTE(subsec:eval-general)` ·
`src/components/domain/profiles.py` and `datasets/general.yaml`.

**No result dependency.** Start now — the data preparation is the long pole and
it is needed whatever BP31 finds.

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

## 1. The opportunity nobody has written down

`/home/itec/emanuele/Datasets/UVG/1920x1080` holds **`Jockey` and
`ReadySteadyGo`**, 1920x1080, 120 fps, 4:2:0 8-bit YUV.

Those are **UVG sequences**, which is to say they are among the handful of clips
every video-coding referee has seen anchor numbers for. Both are sports with a
moving camera and a small, fast, salient subject on a large predictable
background — which is PointStream's stated regime, in a dataset the paper did
not curate for itself.

That matters for a reason the current evaluation plan does not address: **every
number in this paper is measured on broadcast tennis that this project selected,
segmented and cached itself.** A referee is entitled to ask whether the regime
was found or built. One standard sequence answers that in a way six more tennis
matches cannot.

**Priority order, therefore:**

1. **UVG `Jockey` and `ReadySteadyGo`** — highest value per hour. Standard,
   citable, and in the claimed regime.
2. **DAVIS human sequences** — what `general.yaml` already names, and what
   `subsec:eval-general` is written around. Broader content, closer to the
   "degrades gracefully outside its domain" claim the section actually makes.
3. Football, `PLAN.md` §7 P2 item 19, only if time remains.

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

- **BD-rate on UVG `Jockey` against av1, same protocol as the tennis ladder:
  [+20%, +400%]** relative to whatever the tennis number is at the same span and
  scene count. Inside a factor of two of tennis means the pipeline is domain-
  agnostic in a way nothing has yet suggested — check the domain profile actually
  switched. Far above means the perception stage is failing on non-tennis content
  and the number is measuring a broken detector, not a codec.
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
- At least one **UVG** sequence and two DAVIS sequences carry a BD-rate against
  the same anchor and protocol as the tennis ladder, reported per sequence with
  the spread.
- `HOLE(subsec:eval-general)` is cleared by the edit that lands the data, and
  `NEXT(paper-wide)`'s "second evaluation domain" item is struck.
