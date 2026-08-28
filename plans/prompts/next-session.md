# Next session — finish BP24 and run the ladder

Paste below the line.

---

You are continuing **BP24** on PointStream, an object-centric video codec
targeting ACM TOMM on **30 September** (33 days out).

**Read, in order:** `AGENTS.md` · `PLAN.md` §2.16, §2.18, §3, §7 ·
`plans/BP24-encoder-boundary.md` (its Status section lists what remains) ·
**`plans/BP24-findings.md`** — read this one before trusting any number.

Work in a worktree off `origin/main`:

```bash
git worktree add -b wave6/bp24-ladder /home/itec/emanuele/pointstream-w6-b origin/main
cd /home/itec/emanuele/pointstream-w6-b
mkdir -p assets && for x in dataset probe_set raw_4k real_tennis.mp4 weights; do ln -sfn /home/itec/emanuele/pointstream/assets/$x assets/$x; done
ln -sfn /home/itec/emanuele/pointstream/outputs outputs
```

## Where things stand

The rate axis exists. The plate and residual are coded through their configured
codecs, and the ledger **withholds** `transport_to_source_ratio` while any
component is still an array size. What has never run is the ladder, so
`PLAN.md` §7 **P0 items 2 and 3** are still open.

## Do these, in order

1. **`WireCost` honesty pass.** Both residual paths in
   `src/pipeline/residual/signal.py` set `exact=True` with a `basis` describing
   an array. That was true before a codec ran and is ambiguous now. `exact`
   should mean "this is the bitstream size".
2. **Check `actor_reference`.** It is marked raw in the ledger on purpose —
   appearance reports a measured size and nobody has verified it is a coded one.
   Either code it or leave it raw with the reason recorded. Do not clear it
   without evidence, or the ledger silently regains a raw part.
3. **Run the ladder as paired curves.** For codec X, measure X coding the source
   *and* PointStream using X, **same preset, same rungs**, and take BD-rate
   between them. The preset cancels; that is the whole reason this is fair
   (`plans/BP24-findings.md` §1). **Do not rank the per-codec gains against each
   other** — that re-imports the unfairness. Bounds to
   `outputs/bp24-ladder/bounds-before-run.json` before the first encode.

## Things that will bite you

- **Counting coded bytes while reconstructing from the pre-codec array** passes
  every test and produces a fictional RD point. `coded_roundtrip` returns cost
  and decoded frames together so you cannot take one without the other — use it.
- **Both existing ratios are the easy case** (a 2.5%-non-zero residual against a
  static plate). Re-measure on high motion; expect much worse.
- **A QP is not a quality level.** Compare curves, never single-QP totals.
- **A helper script run from outside the worktree imports MAIN's `src`** —
  Python puts the script's directory on `sys.path[0]`, not the cwd. Keep scripts
  inside the worktree or set `PYTHONPATH`.
- **`conda run` swallows pytest's summary.** Use `--junit-xml` and read the XML;
  a piped exit code is not evidence.
- **NFS on this host stalls.** A wedged-looking job may be blocked in
  `nfs_wait_bit_killable` — check `wchan` before blaming the code. mypy took 35
  minutes this way.
- Never `git add -A` in a worktree. Confirm CI is green before saying it is.

## Also open, not for this session unless the ladder finishes early

`plans/BP28-offset-crossover.md` (does the paste ever lose? — Cursor),
`plans/BP19-conditioning-architecture.md` (uni-controlnet, last),
`plans/ENGINE-ROSTER.md` (the scoreboard — update it whenever an engine is
scored).

## Done when

P0 items 2 and 3 are answered with BD-rate curves and their bounds, or the
report says precisely what still blocks them.
