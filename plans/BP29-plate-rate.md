# B'29 — Where does PointStream win?

**The situation.** The BP24 ladder ran and PointStream loses to every anchor:
BD-rate +116.8% (av1), +166.8% (hevc), +165.9% (avc), +378.1% (vvc), each codec
on both arms at one preset, on the most static clip available. On the most
dynamic clip the curves do not overlap at all. A negative result of that shape
is not a paper.

**The situation is also more tractable than it looks**, because the loss is
concentrated in one component and that component is a placeholder. The plate is
**88-91% of the payload at every rung of every sweep**; the residual is 3-9% and
is the most efficient thing in the system (0.9% of the payload for 5.40 dB).
This brief is the hunt for a win, ordered by cost.

**Read first:** `plans/BP24-ladder-report.md` · `plans/BP24-findings.md`
§§6, 13, 16, 17 · `PLAN.md` §2.16 (metric calibration), §2.20.

---

## 1. Change the plate's codec. Cheapest, largest, already measured on the still

`plans/BP24-findings.md` §16. On the same 4K still, at matched fidelity:

| target | JPEG | av1 intra | vvc intra |
|---|---:|---:|---:|
| ~38 dB | 283,431 B | **79,726 B** | **68,477 B** |
| ~40 dB | 345,558 B | 143,925 B | — |
| ~42.8 dB | 461,771 B | 253,346 B | — |

**A factor of 2 to 4 on 88-91% of the payload, for no architectural change.**
The plate stays a single still, transmitted once. It is not even new code for
the x264 route: `src/components/background/sidecar.py` already offers
`roi-video` — a single-frame libx264 encode — as a value of `background.codec`,
and **nothing has ever measured it against `jpeg`**, because `background.codec`
reached nothing at all until BP24 wired `make_background`.

Do this first:

1. **Sweep `background.codec` over `{jpeg, png, roi-video}`** with the residual
   held fixed. One config axis, no new code. Report plate bytes and plate PSNR
   per rung, then re-run the paired ladder at the best one.
2. **Add an intra-codec sidecar** for `av1` and `vvc` on the same interface —
   `coded_roundtrip` already codes a single frame, so this is a wrapper. Keep
   the plate on the **same codec as the anchor** in each pair, or the pairing
   discipline breaks.

**Bounds, before running.** Substituting av1 intra at matched plate fidelity in
the jpeg75/qp38 rung takes the plate from 463,334 B to roughly 253,000 B and the
total from 525,462 B to roughly 315,000 B — a 40% cut. If the whole curve moves
like that, the rate ratio falls from 2.17x to about 1.30x and the av1 BD-rate
lands near **+30%**. **That is still losing**, which is the honest expectation to
write down before the run. It is a much better place to lose from.

## 2. Go where the codec is weakest: very low rate

The ladder stopped at QP 55. PointStream's floor is a plate; a codec's floor is
blocking artefacts, and the two degrade differently. The hypothesis worth testing
is that below some rate the anchor's quality collapses while PointStream's does
not, because a clean plate plus pasted crops does not fall apart the way a
starved transform codec does.

Extend the anchor to QP 58, 61, 63 (av1's range runs to 63) and put the cheapest
plate configuration against it. This is four extra encodes and it directly tests
the "at some level of compression PointStream is smaller" intuition. **On the
current evidence it is unlikely on frame PSNR** — at QP 55 av1 is already at
85,995 B against a plate that cannot go below roughly 70,000 B without falling
apart — but the crossover, if it exists, is a legitimate operating-point claim
and it is cheap to look for.

## 3. Change what is being measured — but declare it first

The architecture is object-centric. Frame PSNR is dominated by the background,
which is 99.4% of the pixels, so it cannot express the claim the system is built
to make. BP23 measured object region at **14.30 dB** against background at
**39.46 dB** on the same frame: the objects are where the error is and where the
bits should go.

A foreground-scoped claim — *at equal total bitrate, PointStream delivers better
object-region quality* — is a real claim, and the machinery exists:
`QualityReport` already reports object, background and frame roles separately,
and VMAF and LPIPS both run.

**The integrity condition, and it is not optional.** Choosing the metric after
seeing the frame-PSNR result is the exact shape of a result fitted to a
narrative. What makes a foreground-scoped claim defensible rather than
post-hoc:

- **Declare it before the run**, in the bounds file, with the reason stated as
  an *a priori* property of the architecture (the system spends its budget on
  objects, so it should be measured on objects) rather than as a response to the
  frame number.
- **Report frame PSNR beside it every single time.** A table that shows only the
  favourable region is the failure mode; a table that shows both, and says which
  question each answers, is a finding.
- **Calibrate the region metrics at the working resolution first.** `PLAN.md`
  §2.16: VMAF's ceiling on this content is 97.54 and it floors at 0.00 for both
  severe blur and an unrelated clip; LPIPS's ordering *inverted* at 960x540
  while holding at 4K. Anchors do not transfer across resolution.
- **Give the anchor region control.** The evaluation section already commits to
  this: a semantic codec that concentrates its budget on the salient region must
  not be compared against an anchor forbidden from doing the same. That arm has
  never been run, and running it is part of making a foreground claim honest.

There is prior art for this framing in the **presley** project — I have not seen
it, so whoever picks this up should read how the foreground-scoped evaluation was
constructed there before rebuilding one.

## 4. Closed doors, so nobody re-opens them

- **Reusing one plate across scenes of a match does not work on this content**
  (`plans/BP24-findings.md` §17). First frames of different scenes in the same
  match differ by 13.75 dB and 15.10 dB, mean absolute difference 39 and 23 grey
  levels: broadcast tennis cuts between camera positions rather than returning
  to one framing. Registration — panorama stitching — is a different and much
  larger piece of work; simple reuse is not available.
- **And even if it were, the symmetry argument holds:** a long-GOP codec
  amortises its own intra frame across the same footage. Any cross-scene
  amortisation given to PointStream must be given to the anchor as the same
  total footage, or the comparison is rigged. The one genuine asymmetry is that
  a codec must start a fresh intra frame at every *cut* while a registered plate
  need not — but that only pays once the registration exists.

## 5. Sequencing

1. `background.codec` sweep (§1.1) — hours, no new code.
2. Intra sidecar for av1/vvc (§1.2) — a day, then re-run the ladder.
3. Low-rate extension (§2) — four encodes, run alongside.
4. If the rate claim still fails: declare the foreground-scoped claim, calibrate
   the region metrics, run the region-controlled anchor arms (§3).
5. Panorama stitching (`build_plate`) — the largest lever and the largest job,
   and §1 tells it what target to beat.

## Done when

Either the paired BD-rate is materially better than +116.8% with the reason
attributed to a named change, or a foreground-scoped claim is declared in
advance, measured with calibrated metrics against a region-controlled anchor, and
reported beside frame PSNR. **A result that is negative on both is reported as
such** — but not before §1 and §2 have been run, because both are cheap and
neither has been tried.
