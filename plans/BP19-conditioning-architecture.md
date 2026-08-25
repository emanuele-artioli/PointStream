# B′19 — The conditioning architecture: what to retrain, and the shared backbone

**Owns:** `scripts/train_controlnet.py`, `src/components/generation/controlnet.py`
and any new generator module, `src/contracts/capabilities.py` if a new
conditioning capability is needed.

**All four gates are now passed.** `BP14` landed 2026-08-24. The last training
run burned 14 GPU hours on a series that was flat from epoch 1 because it
stopped on nothing; that is no longer allowed.

This is now **the critical path**. It is also the only stream that spends real
GPU time, so it goes second in its wave, behind whatever else can run free.

| Gate | Brief | Why |
|---|---|---|
| ~~Headroom~~ | `BP20` | ✅ **PASSED 2026-08-23.** A player is ~1% of the pixels and **17–24%** of the bitrate on real 4K, a 15–47× concentration (`PLAN.md` §2.14). The premise holds; this brief is unblocked on that gate. |
| **Stop rule** | `BP14` | ✅ **LANDED 2026-08-24.** `TaskStopRule` observes coding-task LPIPS, never diffusion loss. CI `32747593873`. |
| ~~Instrument~~ | `BP18` | ✅ **DONE 2026-08-23.** `reid` + `palette`, calibrated on ground-truth pairs at 17.1σ, with `IdentityScale` so a score is quoted between its measured anchors (`PLAN.md` §2.12). Use it. |
| ~~Caption channel~~ | `BP17` | ✅ **DONE 2026-08-23, and the answer is nothing.** Switching it on moved three arms inside noise and pose-controlnet 1.5σ *worse* (`PLAN.md` §2.15). The text channel is not where the appearance problem lives — do not spend more on it. |

## Where the conditioning actually stands

Not "the generators do not use appearance" — that is too coarse to plan from.
Three channels are registered and all three are in a different state
(`PLAN.md` §2.11):

| Channel | Registered as | State | What this brief owes it |
|---|---|---|---|
| **text / caption** | `pose-controlnet`, alias `caption-controlnet` | **trained, never driven** | nothing — `BP17` handles it, no training |
| **keyframe / reference image** | `pose-ref-controlnet` | trained, measured, **failed for a known reason** | a *correct* reference pathway, not the failed recipe |
| **latent / image embedding** | `ip-adapter-controlnet` *declares* `appearance:image-embedding` | **declared, never trained** — the checkpoint is the mislabelled segmentation ControlNet (§2.3) | actually train one |

There is also **`multi-controlnet`**, registered with pose *and* mask as separate
conditions, deliberately excluded from the roster drive list and therefore
**never measured**. It is the existing multi-condition arm and the cheapest
possible probe of the shared-conditioning idea. Measure it before building
anything.

**Why `pose-ref-controlnet` failed, so it is not repeated.** The reference was
painted *into the control image*, under the skeleton. The control branch is
trained to read structure; putting identity there fights the branch's job.
+0.12 dB over the un-retrained model, inside noise (§2.4). **Do not repeat that
recipe.** A reference must enter through a path built to carry appearance —
cross-attention or an adapter — not through the structure channel.

## The three candidates, cheapest first

### 1. A real IP-Adapter arm — cheapest, closes a known gap

`ip-adapter-controlnet` claims an image-embedding pathway it does not have,
because line 82 of the training script put `"ip-adapter"` in the same branch as
`"seg"` with `cond_dir = None`. An actual IP-Adapter is a small cross-attention
adapter over a frozen backbone: cheap to train, and it makes the registry entry
honest either way. **Even a negative result here is worth having**, because the
roster currently carries an arm whose declared capability is fiction.

Note the expected ceiling before training: IP-Adapter conditions on a **CLIP
image embedding**, and §2.7's literature note is that those "lack fine-grained
spatial details, causing appearance drift under large deformations". So expect a
*semantic* appearance match — right kit colour, roughly right build — not
identity. That is a real result to report, not a disappointment.

### 2. Retrain a ControlNet on the coding task — medium

Every current checkpoint was trained as *condition + prompt → image*. That is not
what PointStream asks at inference, which is *reference + condition → this
person at this pose*. `ControlNetDataset` yields `{image_path, cond_path, prompt}`
and has an `include_reference` path already; the training task, not the
hyperparameters, is what needs changing.

**The reference must not go into the control image.** See above.

### 3. Uni-ControlNet — the shared backbone with per-condition heads

This is the architecture for "one model that takes any input type, with a head
per input converting it to a common shape". It exists and is SD-1.5 based, which
matches our stack.

**What it does**, to be verified against the paper and repo before any
integration: two adapters over a frozen SD-1.5 backbone —

- a **local control adapter** taking structural conditions (canny, HED, sketch,
  depth, segmentation, openpose, …) through condition-specific entry
  convolutions into a **shared** feature path injected by feature
  denormalisation, so N structural conditions cost one adapter rather than N
  ControlNets;
- a **global control adapter** taking a CLIP **image** embedding, which is the
  appearance channel.

That maps onto our axes almost exactly: the local adapter is the comparison
backbone's conditioning slot (`eval-object` varies exactly these), and the global
adapter is the appearance slot. It would also make the lattice cheaper — today
each conditioning arm is a separate fine-tune.

**UniControl** and **Composer** solve the same problem differently and are worth
one paragraph of comparison in Related Work regardless of what we build.

**Check the licence, and the weights' licence separately, before integrating.**
That is the rule that stranded MOFA-Video and StableAnimator after the work was
done, twice. Do not take a licence claim from this brief or from memory — read
the repository and the model card, record the date checked, and write the finding
into `DEFERRED.md` if it blocks.

## The ReferenceNet extension, deferred from BP12

`BP12` item 4 proposed adding **Champ** and **MusePose** — SD-1.5 ReferenceNet
siblings of Animate-Anyone. That was deferred deliberately, and the reasoning
still holds: the rationale rested on the cross-appearance test showing AA's
pathway working, and that test cannot show it (§2.10). Adding two more arms
before `BP18` exists buys two more rows that lose to a pasted keyframe.

**Revisit only after `BP18` lands**, and then judge them on the identity axis,
not on a distance-to-target metric. Licence check before integration, as above.

## Bounds, to be written properly before any run

`BP18`'s instrument now exists, so these can be concrete. Write the per-arm
bands before the first sample, and evaluate on **both axes together**:

- **`reid`**, quoted through `IdentityScale` — a raw 0.53 is the *stranger
  floor*, not a halfway mark. `TENNIS_SCALE` holds the measured anchors.
- **LPIPS**, against the static-copy floor of **0.4505** and the
  unrelated-image null of **0.7358**.

The combination is the point: high `reid` with poor LPIPS is a paste, good LPIPS
with low `reid` is a convincing stranger, and only both together is success.

What is already known and must not be forgotten:

- **The static-copy floor is 0.4505 LPIPS / 13.51 dB** on this probe set, and no
  engine has beaten it. Any retrain that does not beat it has not changed the
  conclusion, whatever its training loss did.
- **A falling diffusion loss is not evidence.** It fell throughout the last run
  while sample quality was flat from epoch 1. `BP14` exists for this.
- **A large cross-appearance delta is not evidence either.** A paste scores the
  maximum on it.

## Traps

**Never a version bump on the pinned env.** Several forked models here are
version-sensitive; a stray `pip install` has broken them before. New packages go
in `pyproject.toml`, or into a separate conda env that is named in the report.

**Train on the aligned data.** `assets/dataset` carries two frame-naming
conventions inside one track group, and 44% of tracks are offset (§2.2). Pair by
position, never by rebuilding a filename. `src/shared/tennis_dataset.py:95-110`
is the correct pattern; its docstring describes the wrong convention, so do not
"fix" the code to match the comment.

**Animate-Anyone has seen both held-out videos** (§2.8). Any new arm trained on
the same data inherits that problem, and the split must be stated.

**One variable per run.** A retrain that also changes the conditioning, the
prompt and the offsets measures nothing.

## Done when

Deliberately not fixed yet — the shape of "done" depends on `BP13`'s fork. At
minimum, whatever is attempted here reports:

- what was trained, on what task, with what stopping criterion and why it stopped;
- its score against the static-copy floor **and** against `BP18`'s identity
  instrument, with n and a standard error, paired on clips;
- the licence status of anything integrated, with the date checked;
- and, if the answer is that the architecture does not close the gap, that
  written as a scoped finding rather than as a call for more tuning.

## Delivered so far — 2026-08-24

**`multi-controlnet` measured, both axes, not citable.** Pose epoch 10 + seg
epoch 7, seed 42, 12 clips × offsets 1–8. Licences were checked the same day
(IP-Adapter Apache-2.0 code and weights; Uni-ControlNet MIT code and weights)
and recorded in `outputs/bp19-conditioning/bounds-before-run.json` before any
generate.

LPIPS (object bbox of the letterboxed mask): **0.579 ± 0.013** (n=96).
Identical 0, this-run unrelated null 0.736, static-copy floor 0.451. Pre-written
band 0.50–0.78; inside it. `compare_paired` vs the paste: **+0.128 ± 0.013
(10.2σ, n=96; 4.5σ on 12 clip means) — static-copy ahead.** Vs the unrelated
null: 15.3σ, multi ahead. Object PSNR 11.38 dB vs floor 13.51 dB. The harness
labels the arm *not using appearance*.

`reid` through `TENNIS_SCALE` (same-person 0.8663, different-person 0.5315):
**0.628 ± 0.013** (n=96), **+29% of the span**. Pre-written band 0.50–0.72;
inside it. Same-session GT: same-person 0.878 ± 0.009, different-video donor
0.491 ± 0.008. Engine vs GT same: **−0.250 ± 0.014 (17.8σ, 7.0σ on clip
means) — not a paste.** Engine vs donor: **+0.137 ± 0.016 (8.7σ, 3.3σ on clip
means) — some identity signal, not a stranger.** 20/96 rows sit below the
different-person anchor; 1/96 above same-person.

Reading both together: middling LPIPS and middling reid. Two conditions do not
create an appearance path, which is what the bounds said.

**Dataset honesty for IP-Adapter landed** (`4aa7c94`).

**IP-Adapter training loop is wired.** `--condition-type ip-adapter --include-reference` now:

- loads stock OpenPose ControlNet and **freezes** it
- attaches `h94/IP-Adapter` (`ip-adapter_sd15.bin`, ~22M) on the frozen UNet
- optimiser sees only adapter parameters (image proj + IP-Attn); a count outside 10–40M aborts
- reference goes through CLIP vision, not into the control image
- checkpoints write `ip-adapter.bin` next to the frozen ControlNet so the stop-rule generator can load it

Do not repeat pose-ref. `--smoke-check-reference` is refused on this condition because that flag paints the reference under the skeleton.

**Bounds, written before the first training sample** (2026-08-25):

- Coding-task LPIPS through `TaskStopRule` / `TENNIS_SCALE` is not the train metric; stop on coding-task LPIPS vs static-copy floor as BP14.
- After a run that is allowed to finish: object-bbox LPIPS expected **0.50–0.78** (pose 0.60, paste 0.45, unrelated 0.74). A number below 0.45 is an alarm (paste-through). Above 0.74 is an alarm (worse than unrelated).
- `reid` through `TENNIS_SCALE` (same 0.8663, different 0.5315): expected **0.53–0.72**. Ceiling is semantic appearance (kit colour, build), not identity — CLIP image embeds lack spatial detail. A score at the same-person anchor (0.87) is an alarm.

Launch (not a result until it stops on the task):

```
conda run -n pointstream --no-capture-output python -u scripts/train_controlnet.py \
  --condition-type ip-adapter --include-reference \
  --output-dir assets/weights/ip-adapter-trained \
  --batch-size 4 --epochs 10
```

Uni-ControlNet remains last.
