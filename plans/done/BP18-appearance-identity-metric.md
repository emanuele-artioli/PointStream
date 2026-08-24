# B′18 — An instrument that can tell "the right body appeared" from "the output moved"

**Owns:** `src/components/metrics/**`, `tests/invariants/test_metric_calibration.py`.

## Why this exists

`BP12` retired the cross-appearance control as a test of whether an engine uses
appearance: **a pasted keyframe tops that scale with no network at all**
(`PLAN.md` §2.10). What is left is a gap, and every proposed next step falls into
it. Retraining, a new dataset, a new architecture — each would be evaluated with
the same yardstick that cannot distinguish

- *the output changed when I swapped the reference*, from
- *the right body appeared*.

**Build the instrument before spending GPU on the thing it measures.** That is
the lesson of the last three weeks in one line.

## Not faces

The obvious literature answer — CSIM / ArcFace face similarity — **does not
apply here and must not be adopted.** PointStream reconstructs *bodies in motion*
and, later, other object classes. A player bounding box averages 88,415 px in a
4K frame (§2.6); the face inside it is a few tens of pixels, often turned away,
often motion-blurred. A face metric would be measuring noise, and it would not
generalise past people at all.

## DISTS is not the answer either

DISTS is already wired (added 2026-08-23) and **should be reported** — Sparse2Dense
headlines its BD-rate on DISTS, so we need it for comparability. But it does not
close this gap: it is a *distortion* metric with the same structure as LPIPS, so
a paste tops it exactly as a paste topped the cross-appearance test. §2.7 also
measured it compressing badly at the far end — blur 0.323 against unrelated
0.333 — at 56.6 ms/frame. **Final-results metric, not the identity instrument.**

## The recommendation: a person re-identification embedding

Person ReID models (OSNet and relatives, `torchreid`) embed a **full-body crop**
into a vector where the same person in a different pose, scale and viewpoint is
close, and a different person is far. That is the general-purpose body analogue
of face similarity, it needs no face, and it extends to other object classes with
a different embedding backbone.

**Score the generated frame against the ground-truth target frame**, cosine
similarity in embedding space. Because ReID is built to be pose-invariant, this
asks a genuinely different question from LPIPS:

| | LPIPS to target | ReID to target | reading |
|---|---|---|---|
| static copy | poor | **high** | right person, wrong pose — and that is exactly what a paste is |
| a generator drawing someone else | can be middling | **low** | the failure we could not name before |
| a working generator | good | high | the target |

Neither number alone separates those three. Together they do, which is the whole
point. **A paste scoring high on ReID is correct, not a bug** — its failure shows
up on the other axis.

## Calibrate before ranking anything

Non-negotiable, and the reason this brief is not just "pip install torchreid".
ReID backbones are trained on surveillance imagery; broadcast tennis crops at
wildly varying scale are out of domain, and an out-of-domain embedding can be
perfectly ordered and still uninterpretable (§2.7). Anchor it on **our** data,
following `tests/invariants/test_metric_calibration.py`:

| Anchor | Must show |
|---|---|
| identical crop | the metric's perfect value |
| **same player, different frame, same track** | high — this is the one that matters |
| same player, different scene | high, and this is the honest generalisation test |
| **different player, same match** | clearly lower; both wear kit, both on one court |
| unrelated crop from another video | lowest |

The decisive pair is *same player different frame* against *different player same
match*. If those two do not separate, **the instrument does not work on this
content and must not be shipped** — say so and stop, rather than shipping a
metric that produces orderings.

Report the absolute scale against the backbone's published range, not only the
ordering.

## A cheap companion worth having anyway

A **colour histogram distance over the player mask**. In tennis, kit colour is
the dominant identity cue; this is crude, zero-dependency, fully interpretable,
and fails in ways a human can immediately see. It is not the headline, but when
the ReID number and the histogram disagree, one of them is wrong and you will
want to know which.

## Traps

**Do not install into the pinned `pointstream` env if `torchreid` drags a torch
version bump.** Several forked models here are version-sensitive. Check first;
if it conflicts, either vendor the single backbone file with its weights or
build a separate env and say which. `pyproject.toml`, never ad-hoc `pip install`.

**The metric must be region-scoped like the others.** It takes a crop; feed it
the player box, and record the box with the score.

**This is an instrument, not a result.** Landing it settles nothing about any
engine. It makes the next question answerable. Resist reporting a roster ranking
in the same commit.

## Done when

- A ReID-based identity metric is registered on the metrics axis, scoped to a
  region, with the box recorded.
- It is calibrated against the five anchors above, asserted as invariants, with
  the *same-player* vs *different-player-same-match* separation quantified —
  or it is **rejected**, with that separation reported as the reason.
- The colour-histogram companion is registered beside it.
- DISTS's role is written down: final-results comparability, not identity.
- `PLAN.md` §2.10's closing paragraph is updated to point at the instrument that
  now exists, or to record that ReID failed calibration on this content.

---

## Delivered — 2026-08-23

Numbers and their reading are in `PLAN.md` §2.12;
`outputs/bp18-reid-calibration.txt` is the record and
`outputs/bp18-reid-bounds.txt` was written before any anchor was scored.

**`reid` is registered and usable.** OSNet x1_0 / MSMT17. Separation between
same-player and different-player-in-the-same-match is **0.3410 ± 0.0226, 15.1σ,
n=52 different-player pairs** — clearing the ≥0.25 gate — with a monotone
ordering across all six anchors.

**`palette` is registered beside it** and immediately justified itself by
disagreeing: sharper on within-match player separation, fooled by an official in
a black tracksuit where `reid` is not, and collapsing across scenes where `reid`
holds. An invariant asserts that disagreement, so a future change that collapses
them into one measurement fails loudly.

### Decisions worth carrying forward

- **Vendored, not installed.** `torchreid` is absent and pulling it risks moving
  torch, which several pinned forks cannot survive. The architecture depends on
  nothing but torch, so it is copied with its MIT licence and the gdown
  machinery removed. Nothing downloads at runtime. `pyproject.toml` unchanged.
- **Licences read before integrating, both of them**: code MIT
  (KaiyangZhou/deep-person-reid), weights MIT (kaiyangzhou/osnet model card),
  checked 2026-08-23 and recorded in the module docstring rather than in a
  commit message.
- **This metric does not require pixel alignment**, and that is the point. A
  `paired()` call in the score path would silently re-impose the constraint it
  exists to escape; a test pins it.
- **Hand labels were unavoidable.** The dataset carries no player identity, so
  27 tracks across three videos were labelled by eye from contact sheets. Two
  videos are left unlabelled rather than guessed at — in `alcaraz_ruud` every
  sampled track wears the same kit. The circularity (labels from kit, metrics
  read kit) is written into `experiments/probe/player_labels.py`.

### What was found on the way

- **Two unrelated white-noise clips score 0.97** on this backbone. A learned
  metric finds noise self-similar. Noise is not a valid "unrelated" anchor and
  was not used as one.
- **The bounds were wrong in a specific, reusable way**: they assumed cosine
  similarity has a natural zero for "unrelated". It does not, for person crops.
  Report against the measured floor.
- **`palette` beats `reid` at within-match player separation.** Worth stating
  plainly before anyone describes the learned metric as measuring "identity":
  on this content, most of the signal is what the player is wearing.

### Done when — status

- [x] `reid` registered on the metrics axis, region-scoped, refusing a mask
      region rather than scoring a person-shaped hole.
- [x] Calibrated against the anchors, asserted as invariants, with the decisive
      separation quantified. **Passed** rather than rejected.
- [x] `palette` registered beside it, with an invariant on their disagreement.
- [x] DISTS's role written down: comparability with Sparse2Dense, not identity.
      Noted also that it is **not currently registered** on the metric axis —
      it is used ad hoc. Registering it is a loose end for whoever needs it.
- [x] §2.10's closing paragraph is superseded by §2.12.

### Not done, deliberately

- **No engine was scored.** Landing an instrument settles nothing, and reporting
  a roster ranking in the same session would have been the exact mistake this
  brief exists to prevent. `BP17` and `BP19` are where it gets used.
- **The 12 probe clips are not re-scored on `reid`.** That belongs to whichever
  brief next drives the roster, so the identity number arrives beside a
  distortion number in one table rather than as a standalone league table.
