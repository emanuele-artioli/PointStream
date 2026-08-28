# Next session — the plate is the rate

Paste below the line.

---

You are continuing **PointStream**, an object-centric video codec targeting ACM
TOMM on **30 September** (33 days out).

**Read, in order:** `AGENTS.md` · `PLAN.md` §2.20, §3, §7 ·
**`plans/BP24-ladder-report.md`** — the whole thing, it is short ·
`plans/BP24-findings.md` §§6, 13, 14 before trusting any rate number.

Work in a worktree off `origin/main`:

```bash
git worktree add -b wave7/plate /home/itec/emanuele/pointstream-w7-a origin/main
cd /home/itec/emanuele/pointstream-w7-a
mkdir -p assets && for x in dataset probe_set raw_4k real_tennis.mp4 weights; do ln -sfn /home/itec/emanuele/pointstream/assets/$x assets/$x; done
ln -sfn /home/itec/emanuele/pointstream/outputs outputs
```

## Where things stand

`PLAN.md` §7 **P0 items 2 and 3 are closed.** The ladder ran. PointStream costs
**2.2x to 4.8x** the rate of the codec it is built on, per codec at that codec's
own preset, on the most static clip available; on the most dynamic clip it does
not reach the codec's *worst* operating point at any rate.

The ladder also says exactly where the loss is. **The plate is 88-91% of the
payload at every rung of every sweep**, and it is still the *first source frame*
JPEG-coded rather than a stitched panorama. The residual, by contrast, is
excellent value: 0.9% of the payload for 5.4 dB on static content, up to 14.8 dB
over the unaided reconstruction on dynamic content.

So: **the single highest-value open item in the project is the plate.**

## Do these, in order

1. **Wire `build_plate` into the runner.** It exists in
   `src/components/background/plate.py` and `make_background` does not call it.
   Today `background.method` selects a *transmission strategy* over one frame.
   A stitched panorama amortises across the clip, which is the whole argument
   for sending a background model at all.
2. **Re-run the ladder.** `bash experiments/tier/run_ladder.sh 8` reproduces
   every axis; write bounds to `outputs/bp25-plate/bounds-before-run.json`
   first. The av1 low-motion number to beat is **+116.8%**.
3. **Then try a longer clip.** Eight frames is the least favourable
   amortisation a fixed plate cost can get, and the BP21 cache holds 48-frame
   windows. This is the second-largest known lever and it costs nothing but
   wall clock.

## Things that will bite you

- **A decode that names no `-c:v` re-encodes.** That bug capped every quality
  `coded_roundtrip` returned, including the residual the runner delivers, and
  it was invisible for weeks (findings §14). If a curve goes flat while the
  byte count keeps moving, suspect a second encoder before suspecting the
  content.
- **A rung has to move the thing that dominates the payload.** Sweeping
  `residual.rate` alone moved PointStream's total by 6% (findings §13).
- **`RunResult.frames` is not the delivered clip.** Use `delivered_frames` for
  anything paired with a byte count.
- **Do not rank the per-codec gains against each other.** The presets are not
  equal effort (findings §1). State each beside its preset and stop.
- **`CodecCapabilities` declares no QP range** (findings §15), so an
  out-of-range rung is caught only by the encoder refusing it.
- **NFS on this host stalls.** The conda env lives on it. Measured today at
  ~70 KB/s under load; mypy took 34 minutes. Check `wchan` before blaming code.
- **`conda run` swallows pytest's summary and reports exit 0 anyway.** Use
  `--junit-xml` and read the XML.
- Never `git add -A` in a worktree. Confirm CI is green before saying it is.

## Also open, not for this session unless the plate finishes early

Region arms (named in P0 item 2, not in this ladder) ·
`plans/BP28-offset-crossover.md` · `plans/BP19-conditioning-architecture.md` ·
`plans/ENGINE-ROSTER.md` (update whenever an engine is scored).

## Done when

The plate is a stitched panorama in the runner, the ladder is re-run against
its own pre-written bounds, and the report says whether the plate's share of the
payload moved and what that did to the BD-rate — or says precisely what blocked
it.
