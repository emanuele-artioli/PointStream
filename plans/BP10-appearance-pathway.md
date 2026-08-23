# B′10 — Find an appearance pathway that works

**The critical path.** Replaces `BP8`, whose options are now spent or
re-diagnosed.

**Read first:** `PLAN.md` §2.3 and **§2.4** — §2.4 is the one that changes what
this stream should do.

## Where we actually are

Wave 3 tried three things and read them all against the 11.47 dB static-copy
floor. A control on 2026-08-23 shows the floor answers a different question than
the one that matters:

| Engine | Δ correct − wrong appearance | Reading |
|---|---|---|
| `pose-controlnet` (no reference in training) | +0.86 dB | img2img init leakage |
| `pose-ref-controlnet` (retrained with reference) | +0.98 dB | **retrain added nothing** |
| `ip-adapter-controlnet` | +0.08 dB | no appearance path at all |

**The mechanism is now known.** The only appearance signal reaching a ControlNet
is the img2img init image at `strength=0.65`. It is untrained, weak, and
identical before and after the retrain. Painting a reference into the *control*
image does not teach identity, because the control branch reads structure.

## The one test that must run first

**Run the cross-appearance test on Animate-Anyone.** It is the only remaining
architecture with a *designed* appearance pathway — ReferenceNet cross-attention
rather than an init image. Wave 3 fed it a valid reference and scored 8.96 dB
against the floor, which per §2.4 does not establish whether ReferenceNet works.

The driver exists: `scripts/bp8_coding_task.py` plus the control described in
§2.4. Generate twice per clip, same pose, correct vs foreign appearance, score
both against the true target with **object PSNR and LPIPS**.

**Bounds, written here before running:**

- **Δ ≥ +3 dB** → ReferenceNet works. AA's low absolute PSNR is then the
  real-versus-synthetic confound, not a broken model, and the flagship question
  reopens in AA's favour.
- **+1 to +3 dB** → partial; worth tuning the reference path, not retraining.
- **≈ +0.9 dB** → the same img2img-grade leakage as ControlNet, i.e. ReferenceNet
  is not contributing.
- **≈ 0** → the reference is reaching the net but not the attention path; a
  wiring fault, and the most fixable outcome of the four.

## Then, depending on what it says

**If AA works:** it becomes the quality flagship, and `subsec:eval-object`'s
fixed-backbone comparison moves to whatever conditioning AA accepts. Report its
in-domain limitation (§2.5) with every number.

**If AA does not work:** the remaining honest options, cheapest first —

1. **Wire a real IP-Adapter properly.** §2.4 shows the current one has no
   appearance path at all, which is a *wiring* result, not a model verdict. A
   correctly wired IP-Adapter against stock SD-1.5 needs no retraining.
2. **Raise the img2img contribution.** Lower `strength` trades pose fidelity for
   appearance carry-through. This is *not* the forbidden tuning — the forbidden
   tuning was searching for quality on a model with no appearance input. Here it
   maps a known mechanism, and the curve itself is a finding: it measures what
   the appearance channel is worth.
3. **A licence-cleared reference-conditioned model.** StableAnimator is blocked
   on SVD-XT (§2.6); check whether anything equivalent has open weights.

## Traps

**Do not repeat the pose-ref recipe.** Reference-in-the-control-image is
measured, flat, and understood. `assets/weights/pose-ref-controlnet` is kept as
evidence, not as a base to train from.

**Do not use the static-copy floor as a pass/fail gate.** Keep publishing it —
it is the honest reference and it caught the original fault — but "below the
floor" and "does not use appearance" are different claims. Only the
cross-appearance delta settles the second.

**Report LPIPS alongside PSNR for every generative arm from now on.** PSNR
structurally favours real-pixels-wrong-pose over synthetic-pixels-right-pose,
which is exactly the comparison being made. This is a narrow exception to
`PLAN.md` §6.5's PSNR-only triage rule, and it is why.

**n=12 is thin.** Per-clip sd is ≈2 dB, so se ≈0.58 and a +1 dB effect is ~1.5σ.
Use more clips or more offsets before any delta near 1 dB decides anything.

## Done when

- Animate-Anyone has a cross-appearance delta with LPIPS, judged against the
  bounds above.
- Either one engine demonstrably uses appearance through a trained pathway, or
  all four options are spent and the negative result is written with numbers.
- `PLAN.md` §6.2's roster is re-decided on that evidence.
