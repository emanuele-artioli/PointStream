# Next session — BP29: find where PointStream wins

Paste below the line.

---

You are continuing **PointStream**, an object-centric video codec targeting ACM
TOMM on **30 September**.

**Read:** `AGENTS.md` · **`plans/BP29-plate-rate.md`** (the brief; short) ·
`plans/BP24-ladder-report.md` · `plans/BP24-findings.md` §§13, 16, 17.

```bash
git worktree add -b wave7/bp29-plate /home/itec/emanuele/pointstream-w7-a origin/main
cd /home/itec/emanuele/pointstream-w7-a
mkdir -p assets && for x in dataset probe_set raw_4k real_tennis.mp4 weights; do ln -sfn /home/itec/emanuele/pointstream/assets/$x assets/$x; done
ln -sfn /home/itec/emanuele/pointstream/outputs outputs
```

## The situation

The paired ladder ran. PointStream **loses to every codec** — BD-rate +116.8%
(av1), +166.8% (hevc), +165.9% (avc), +378.1% (vvc) — on the most static clip
available, and does not overlap the anchor at all on the most dynamic one.

The loss is concentrated in one placeholder component. The **plate is 88-91% of
the payload** and is coded as JPEG; the residual is 3-9% and is the most
efficient thing in the system. Measured on the same 4K still, at ~38 dB: JPEG
283,431 B, **av1 intra 79,726 B**, **vvc intra 68,477 B**.

## Do these, in order

1. **Sweep `background.codec` over `{jpeg, png, roi-video}`**, residual held
   fixed. No new code — `roi-video` is already a single-frame x264 encode in
   `src/components/background/sidecar.py` and has never been measured against
   `jpeg`. Re-run the paired ladder at the best rung.
2. **Add an av1/vvc intra sidecar** on the same interface; keep the plate on the
   **same codec as the anchor** in each pair or the pairing breaks.
3. **Extend the anchor to QP 58/61/63** and look for a low-rate crossover.
4. Only if 1-3 still lose: **declare a foreground-scoped claim in the bounds
   file before running it**, calibrate the region metrics at the working
   resolution, and run the region-controlled anchor arms. §3 of the brief says
   what makes that defensible rather than post-hoc — read it before doing it.

Bounds to `outputs/bp29-plate/bounds-before-run.json` before the first encode.
The number to beat is **+116.8%**; the brief's own estimate for step 1 is about
**+30%**, i.e. still losing.

## Things that will bite you

- **A decode that names no `-c:v` re-encodes** and caps every quality it
  returns (findings §14). Flat curve while bytes keep moving → suspect a second
  encoder before suspecting the content.
- **A rung must move the thing that dominates the payload** (findings §13).
- **`RunResult.frames` is not the delivered clip** — use `delivered_frames`.
- **Do not rank per-codec gains against each other** (findings §1).
- **Reusing one plate across scenes does not work on this content**
  (findings §17) — 13.75 dB between scenes of the same match. Closed door.
- **NFS: ~6 file opens/second.** Imports cost ~160 s per process, warm or cold.
  Batch work into long-lived processes; `run_ladder.sh` pays it once per axis.
- **`conda run` swallows pytest's summary and exits 0 anyway** — use
  `--junit-xml` and read the XML.
- Never `git add -A` in a worktree. Confirm CI is green before saying it is.

## Done when

Either the BD-rate is materially better than +116.8% with the improvement
attributed to a named change, or a foreground-scoped claim is declared in
advance and measured honestly beside frame PSNR — or the report says precisely
what blocked both.
