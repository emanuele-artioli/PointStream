# B3 — Generation, appearance, motion

**The largest stream. Start it first.**

**Owns exclusively:** `src/components/generation/**`,
`src/components/appearance/**`, `src/components/motion/**`,
`tests/components/test_generation*.py` and siblings.
**Implements:** `src/contracts/conditioning.py` and
`src/contracts/objectstream.py` — read both first.

## The defect being replaced

`BaseGenAIStrategy.generate` takes a parameter named `dense_dwpose_tensor` that
actually carries **a pose** for some backends, **a binary mask or canny image**
for others, and **a `(pose, mask)` tuple** for multi-controlnet — with the
compositor string-matching the backend's *name* to decide which
(`src/decoder/compositing/compositor.py:243`). `controlnet_engine.py:590-604`
still carries the comments written while someone worked that out. Separately,
temporal capability is detected by `isinstance(strategy, AnimateAnyoneStrategy)`,
so a new temporal backend must subclass that specific class to be recognised.

`ConditioningBundle` replaces the first: every input in its own typed, optional
field, with `require()` failing by name at the call site. Declared capabilities
replace the second.

## What to build

**Generators**, each a registry entry declaring what it accepts:

- ControlNet variants (canny, seg, pose, ip-adapter, multi)
- pix2pix, SPADE4Tennis
- Animate-Anyone
- **MOFA-Video** — trajectory-conditioned, public checkpoints, SVD-based. The
  leading candidate for the sparse-trajectory route, but **a candidate, not a
  commitment**: it earns its place on the same terms as everything else.
- **upscale-refine** — no diffusion, just upsampling and refinement of a
  low-resolution appearance. The cheap baseline every generative model must beat.

**Appearance representations:** compressed image (with *both* degradation knobs,
JPEG quality and downscale — they are not equivalent), diffusion latent, image
embedding.

**Motion representations:** keypoints, sparse trajectories, per-object encoded
video.

## Traps specific to this stream

**Sparse trajectories, never dense flow.** Dense per-pixel flow costs what block
motion vectors cost, which defeats the purpose. The models that matter consume
sparse points and expand to dense motion themselves — that expansion is the
decoder's job, not the wire's.

**Fix the duplicated pose-rescale block.** Roughly forty lines are copy-pasted
across four ControlNet classes and are visibly wrong. This changes generated
pixels, which is in scope: pre-rewrite generative results are superseded and
being re-baselined anyway.

**Animate-Anyone's checkpoint is not a general model.** It was fine-tuned on
scenes from a *single tennis match*. Any score it posts is scoped to that, and
that caveat travels with every number it produces. It also cannot enter the
training campaign yet — `scripts/eval_checkpoint.py`'s `ARCH_CHOICES` has no
entry for it, so the campaign cannot score it even if it trains. Make it
**evaluable first**; a full retrain is a separate, later decision.

**Check licences before integrating.** MOFA-Video, DragNUWA and Motion-I2V
weights have not been licence-checked. Tora's are gated under
non-commercial-leaning terms.

## Done when

- No caller passes conditioning positionally or by an overloaded parameter.
- Temporal capability is read from a declaration, never from a class identity.
- An appearance/motion pair no generator accepts is rejected at config
  validation with the workable pairings named.
- `config.validate_backends` is wired for this axis.
- `ruff`, `mypy`, tests pass; import direction clean.
