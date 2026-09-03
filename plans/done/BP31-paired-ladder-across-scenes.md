# BP31 — the paired ladder, over N scenes, with the anchor given the same footage

**Why this exists.** `plans/done/RESEARCH-HISTORY.md` §2.20 measured PointStream losing to every codec
it is built on (BD-rate +116.8% against av1 at preset 10, worse elsewhere), and
§2.20 named the cause: the plate is 88-91% of the payload and every scene paid
for it from scratch. BP30 removed that: the background now amortises across
scenes at **49.2% ± 6.2%** of coding every plate fresh, best case 29.4%
(`plans/done/BP30-findings.md` §§22, 29), and it is wired into the runner as
`background.method: panorama-stream` (PR #41).

This is the run that says whether that changed the answer.

## 0. Scope widened 2026-08-31: run every plate lever, not just this one

This brief was written to price BP30's cross-scene stream. That scope is too
narrow to be worth a 4K ladder run. `plans/done/RESEARCH-HISTORY.md` §7 P0 item 8 names **three** plate
levers, and re-running the ladder with one of them on answers a smaller question
than the one the paper needs.

| lever | status | in the runner? |
|---|---|---|
| (c) stitch a real panorama | **done**, BP29 stream D — residual falls to 0.22x, delivered Y-PSNR +4.9-6.2 dB on a moving clip | yes, `make_background` calls `build_plate` |
| (b) stop paying per scene | **done**, BP30 — 49.2% ± 6.2% of coding every plate fresh | yes, `background.method: panorama-stream` |
| (a) change the plate's codec | **open** | sidecars exist, never swept end to end |

**(a) is genuinely unfinished and is the cheapest of the three.** `av1` and
`vvc` intra sidecars were implemented (BP29 stream B, `sidecar.py`), but the
plate-codec sweep that ran (`plans/done/BP29-plate-codec-report.md`) compared only
`jpeg`, `roi-video` and `png` — the intra sidecars were added by a different
stream and were never in it. That report's conclusion is also *not* the one
§2.21 assumes: **at the rung the BP24 ladder uses, `jpeg` is the cheaper codec**,
and `roi-video` cannot reach that fidelity at any bitrate. The "3.6-4.1x cheaper"
figure for av1/vvc intra comes from a standalone plate probe, not from a swept
comparison at the ladder's operating point. It has never been checked
end-to-end, and that report's end-to-end arms did not run at all — all four
failed on a weights-path fault that has since been fixed.

So the run this brief describes is: **turn on (a), (b) and (c) together, then
re-run the paired ladder.** That is the single measurement that answers whether
the +116.8% gap has closed, which is the paper's core claim.

**Sweep (a) before spending the ladder on it.** One cheap sweep — jpeg against
av1-intra and vvc-intra at the ladder's rung, on the panorama plate rather than
a source frame — decides which codec the ladder arm should use. Picking it by
assumption is how the ladder gets run twice.

**Read first:** `plans/done/BP30-findings.md` §§22-29 · `plans/done/BP30-background-stream.md`
§5 (fairness) · `plans/done/BP24-ladder-report.md` · `plans/done/BP24-findings.md` §§1, 8,
12, 14.

---

## 1. The fairness condition, which is the whole design

**The anchor gets the same footage.** A codec encoding a multi-scene sequence
can also predict across a scene join. If PointStream is allowed to amortise its
background across scenes, the anchor must encode the same concatenated material
under the same constraint, or the comparison is rigged in exactly the way
BP30 §5 was written to prevent.

The paired arms are:

- PointStream over N scenes, `panorama-stream`, low-delay, keyframe interval *k*.
- Codec X over the same N scenes, low-delay, same *k*, same preset on both arms
  so the preset cancels (findings §1).

**PointStream's *possible* asymmetry** is that composited backgrounds may be more
similar to one another than two arbitrary frames at a cut are. That is a
hypothesis. This run does not get to assume it; it is what the paired arms
measure.

## 2. The thing this run is most likely to get wrong

**Do not run it on one clip.** BP30 measured the background lever on one video,
concluded `first` was a fine reference mode and the Canny search was worth 3.65
points, and **both conclusions inverted at five videos** (§29): `first` turned
out to be the worst of the three free options, and the search stopped being
worth anything. The one video originally chosen was the least favourable of the
five.

The reason is quantitative and it applies directly here: **the per-video spread
is larger than every effect measured inside it.** The background saving ranges
0.294 to 0.624 across five videos; every reference-mode difference is a few
points. A BD-rate from a single representative clip would be a confident number
carrying almost no information about the system.

So:

- **N videos, reported per video, not averaged into one figure.** `presley`'s bar
  is n>=6 before a significance claim; BP30 reached 5. Six is the target.
- **Report the spread before the mean.** If the spread exceeds the effect, say so
  and do not lead with the mean.
- The harnesses already take `--video`
  (`experiments/tier/background_stream.py`, `canny_validate.py`), and
  `experiments/tier/scene_plates.py` will enumerate point-class scenes for any
  video in the dataset — `djokovic_federer` has 224, `alcaraz_perricard` 88.

**The same failure has a second form here.** §2.20's ladder ran on
`alcaraz_highlights/scene_000`, explicitly "the most static of the eight cached
windows" — the friendliest content available. A re-run that keeps that clip and
changes only the background method would be comparing against the old number on
the old content, which is fine, *and* would inherit its unrepresentativeness,
which is not. Keep the old clip as the continuity arm; add the others.

## 3. Bounds, to be written before the first encode

Write them to `outputs/bp31-ladder/bounds-before-run.json`.

- **The background's share must fall.** It was 88-91% of the payload. With the
  plate amortised at ~0.49 over 16 scenes, its share should land somewhere near
  75-85% — still dominant. If it does not move at all, the stream is not
  reaching the ledger, which is the failure PR #41's byte-count tests exist to
  catch and which would mean the wiring regressed.
- **BD-rate should improve, and probably not enough.** +116.8% against av1 with
  the plate at ~0.49 of its old cost is arithmetically still a large positive
  number. A result that shows PointStream *winning* is an alarm, not a triumph:
  check the anchor got the same N scenes and the same low-delay constraint
  before believing it. **This is the asymmetry to watch — the check gets applied
  to disappointing results and skipped on exciting ones.**
- **The anchor's own rate must fall too** when it is given N scenes instead of
  one, for the same reason PointStream's does: it can predict across the join.
  If the anchor is unchanged, it is being run per-scene and the comparison is
  rigged in PointStream's favour.

## 4. Traps already paid for

- **`RunResult.frames` is not the delivered clip**; `delivered_frames` is (§8).
- **A decode naming no `-c:v` re-encodes** and caps every quality it returns
  (§14). A flat quality curve while bytes move means a second encoder.
- **RGB-PSNR cannot be the quality axis against a 4:2:0 codec** (§12).
- **Presets are not equal effort across codecs** (§1) — one codec on both arms.
- **`SizesBytes.panorama` is now a *marginal* cost under `panorama-stream`**, and
  a total spanning scenes is only right because chunk 0's keyframe is in the sum
  (`src/runner/accounting.py`). Do not take the mean per-chunk figure as the cost
  of a plate.
- **One bound model is one stream.** `make_background` binds once per run; two
  runs must not share a stage, or run 2 predicts scene 1 from run 1's last scene.

## Done when

The paired ladder is reported over N scenes on both arms under the same
low-delay constraint and keyframe interval, **per video with the spread stated**,
against pre-written bounds — or the report says precisely which arm failed and
why.
