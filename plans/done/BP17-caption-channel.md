# B′17 — The caption channel was trained and has never been switched on

**Owns:** `src/contracts/conditioning.py` (the caption field),
`src/components/generation/controlnet.py`, `experiments/probe/**`.

**Do not** retrain anything in this brief. It costs one probe run.

## The finding

`scripts/train_controlnet.py` reads a **per-track BLIP caption** when one exists
and falls back to a generic prompt only when it does not:

```python
prompt = "photorealistic tennis player, broadcast sports shot"
if caption_path.exists():
    prompt = json.load(open(caption_path)).get("caption", prompt)
```

**114 caption files exist**, one per track, and they carry appearance:

> `"a man in a purple shirt and blue shorts playing tennis, photorealistic
> tennis player, broadcast sports shot"`

**53 of 114 (46%) name a colour**; 57 distinct captions across 114 tracks.

At inference, `src/components/generation/controlnet.py:73` hardcodes

```python
_PROMPT: Final = "photorealistic tennis player, broadcast sports shot"
```

— the *fallback* — and passes it for every frame of every clip.
**`ConditioningBundle` has no caption or prompt field at all**, so there is
currently no way to pass one.

So: the ControlNets were trained with a text channel that describes the player's
kit, and **every number this project has ever measured was taken with that
channel disabled.** `PLAN.md` §2.3 quoted the fallback prompt as though it were
the only prompt, which is how this stayed hidden.

**This is occurrence nine of the standing failure mode**: a pathway exists, is
trained, passes its tests, and is not being driven. It is also the cheapest
outstanding thing that could move the roster, because it needs no training.

## What this is not

**Not a fix for the appearance problem, and probably not a large effect.** A
caption is a handful of tokens through CLIP text encoding — it can say "purple
shirt", not *this* player. §2.7's own literature note is that CLIP embeddings
"lack fine-grained spatial details". Half the tracks have no colour in the
caption at all. Expect a small effect or none.

**It still must be measured**, for two reasons. Every roster number is currently
labelled with a condition that was never true of training, and a channel that is
trained but unreachable is a defect whether or not it helps.

## What to do

1. **Carry the caption on the bundle.** Add an optional text field to
   `ConditioningBundle` and thread it through the probe's `bundle_coding`. This
   is a contract change: run `python -m src.contracts.layers` and the contract
   tests, and keep the field optional so every existing caller stays valid.
2. **Load the per-track caption in the probe set.** `experiments/probe/clips.py`
   resolves channels by position within a track; the caption is a per-track
   sidecar at `{track}_caption.json` in `assets/dataset`, **not** in
   `assets/probe_set`. Decide where it should live — copying it into the probe
   set at build time is cleaner than reaching across — and say which you chose.
3. **Use it in the ControlNet when present**, falling back to `_PROMPT` when
   absent, and **record which was used on every row**. A run that cannot say
   whether the caption was live is the fault this brief exists to fix.
4. **Re-run the four ControlNet arms** through `python -m experiments.probe`,
   captions on, everything else identical: 12 clips, offsets 1–8, seed 42.
   Compare against `outputs/bp12-clip-roster/` with `compare_paired`, **paired
   on clips**. This is a matched A/B; nothing else may change.

## Bounds, written before running

Per-arm LPIPS change from enabling captions, n=12 clips paired:

- **> 0.05 improvement** — surprising and welcome; add a check before believing
  it, and confirm the caption really varies across the clips scored.
- **0.01 – 0.05** — plausible and consistent with a weak semantic channel. Report
  with its standard error and do not call it a fix.
- **inside noise** — the expected outcome. Report it; it retires the idea that
  the roster was unfairly measured, which is worth knowing.
- **worse with captions** — possible: a caption naming the wrong colour for a
  track is worse than a neutral prompt. Check the captions on the losing clips
  before concluding.

**Split the comparison by whether the caption names a colour** (53 of 114
tracks). If captions help at all, they should help more on that half. If the two
halves move identically, the effect is not the caption's appearance content and
should not be described as such.

## Traps

**Confirm the loaded checkpoints were trained with captions.** The training
script reads them *if present*; that they exist now does not prove they existed
when epoch 7 and epoch 10 were written. Check training logs or the checkpoint
metadata before describing this as "switching the channel back on" rather than
"switching a channel on". Say which it is.

**The static-copy floor does not move.** It has no model and no prompt. Any
comparison is still against 0.4505 LPIPS, and any improvement should be read
against how far there is to go, not celebrated for being positive.

**One variable.** Same seed, same offsets, same clips, same device class. A
caption run that also changes steps or strength measures nothing.

## Done when

- `ConditioningBundle` carries an optional caption; the layer check is clean.
- The probe records, per row, which prompt was used.
- The four ControlNet arms are re-run captions-on and compared paired against
  `outputs/bp12-clip-roster/`, split by whether the caption names a colour.
- `PLAN.md` §2.11 says what the channel is worth, including if the answer is
  "nothing measurable".

---

## Delivered — 2026-08-23

**The channel is reachable now, and it is worth nothing measurable.** Full
numbers in `PLAN.md` §2.15; `outputs/bp17-caption/`.

- `ConditioningBundle` carries an optional caption; the ControlNet resolves the
  track caption and falls back to the generic prompt; every row records
  `prompt`, `prompt_source` and `caption_names_colour`.
- **Checkpoint provenance was checked, not assumed.** Captions landed on disk
  2026-07-01, the trainer began reading them in `d1efbcf` (2026-07-06), and pose
  epoch 10 and seg epoch 7 both post-date it — so for those two arms this really
  was switching a channel *back* on. `ip-adapter` loads a stock OpenPose
  ControlNet, so there it only switches SD's text encoder on. That distinction
  is in the plan.
- **The control is exact**: both no-model arms moved by **0.000 ± 0.000**
  (static-copy 0.4505, unrelated-image 0.7358, unchanged to four decimals), so
  the two runs are comparable and nothing else drifted.

| Arm | captions on − generic (LPIPS, lower better) | verdict |
|---|---|---|
| pose-controlnet | +0.020 ± 0.014 (1.5σ) | suggestive that captions are *worse* |
| seg-controlnet | +0.002 ± 0.011 | inside noise |
| ip-adapter-controlnet | −0.002 ± 0.008 | inside noise |
| trajectory-controlnet | +0.001 ± 0.019 | inside noise |

**Only 5 of 12 probe clips name a colour**, which caps how much this channel
could ever have carried. The predicted outcome — inside noise — is what
happened.

**What it bought.** Not quality. It retires the possibility that §2.10's roster
was measured unfairly, and it closes a genuine defect: a trained pathway that no
inference path could reach. The appearance problem is not in the text channel.
