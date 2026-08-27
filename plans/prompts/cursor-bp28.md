# Prompt for Cursor — BP28, the offset ladder

Paste below the line. One worktree, one stream. **Merge PR #27 first** — this
builds directly on BP25's harness.

```bash
git worktree add -b wave6/bp28-offset /home/itec/emanuele/pointstream-w6-a origin/main
cd /home/itec/emanuele/pointstream-w6-a
mkdir -p assets && for x in dataset probe_set raw_4k real_tennis.mp4 weights; do ln -sfn /home/itec/emanuele/pointstream/assets/$x assets/$x; done
ln -sfn /home/itec/emanuele/pointstream/outputs outputs
```

---

You are running **BP28** on PointStream. You wrote the harness this builds on
(BP25), so this should be a short run rather than a build.

**Read first:** `/home/itec/emanuele/.agent-rules/AGENTS.md`, this worktree's
`AGENTS.md`, then `plans/BP28-offset-crossover.md` (your brief) and
`plans/ENGINE-ROSTER.md`.

**You own:** `scripts/bp25_rescore.py`, `outputs/bp28-offset/**`, and the offset
section of `plans/ENGINE-ROSTER.md`.
**You must not touch** `src/runner/**`, `src/pipeline/**`,
`src/contracts/lattice.py`, `config/tier_*.yaml` — BP24 is live in those right
now. Report, do not edit.

## Why this run exists

Grouping your own BP25 rows by offset shows the paste degrading about **ten
times faster** than the fine-tuned model: **+0.0458 LPIPS per offset against
+0.0049**. A linear fit crosses at **offset ≈ 10.4**, just past the range you
measured. Nobody has looked past offset 8.

Extend the same protocol — same 12 clips, seed 42, 20 steps, object-bbox LPIPS
and `reid` — to **offsets 12, 16 and 24**. Arms: at minimum `static-copy`,
`checkpoint-epoch-1`, `unrelated-image`. Add `upscale-refine` and
`seg-controlnet` if cheap: at 0.5585 and 0.5595 they are the two best engines on
the roster, so the crossover question is really about them, not about
IP-Adapter.

## The one thing to get right

**A crossover is not a victory, and the brief will try to tempt you into
reporting it as one.** The paste at offset 8 is already at 0.582 and heading
toward 0.74 — which *is* the unrelated-image anchor. The model sits flat at
~0.70. If they cross near offset 10, they cross where **both arms are about as
good as handing over a photo of a different player.**

So report the crossover offset **and the absolute LPIPS at which it happens**,
and say plainly whether that quality is usable by anyone. "The paste wins
everywhere a codec would actually operate" is a perfectly good result and may
well be the true one.

Bounds go in `outputs/bp28-offset/bounds-before-run.json` **before** the first
generation — the brief gives you starting bands. Report clip-mean **n and
standard error** (n=12), not item-level n=96: eight offsets inside one clip are
not independent, as you established last time.

## Host notes

- `conda run` swallows pytest's summary. Use `--junit-xml` and read the XML; a
  piped exit code is not evidence.
- Long jobs detached in the background; confirm a process is dead with `ps`
  before relaunching.
- GPUs are shared — check first, and say which you took.
- Never `git add -A`. Confirm CI is green before saying it is.

Report: the measured table, the crossover offset or its absence, the absolute
quality there, and whether keyframe interval is a lever worth pulling.
