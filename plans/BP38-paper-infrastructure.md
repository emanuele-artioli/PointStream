# BP38 — Paper infrastructure: figures, baselines a referee expects, reproducibility

**Everything the manuscript needs that does not depend on a result.** Doing it
now means that when BP31 and BP41 land, the paper is a writing job rather than a
build job.

**Owns:** the paper repo `67a9ea6275d3d9785ce57026/` — `figures/`,
`appendices/`, `sections/related_work.tex`, `ref.bib` — plus
`experiments/figures/**` (new) in the code repo. **Commit in the paper repo, not
the parent.** Its `AGENTS.md` and marker convention govern there.

**Read first:** the paper repo's `AGENTS.md` and `sections/README.md` ·
`appendices/README.md` · `plans/ROADMAP.md` §6.

**No result dependency.** Four separable pieces; they can be four sessions.

---

## A. The figure inventory is a museum

`figures/` holds eight files, all dated 2026-07-09, and **exactly one is
`\includegraphics`'d**: `PS-overview.pdf`, in `sections/problem.tex`.

The other seven — `PointStreamOverview.pdf`, `avg_vmaf_vs_bitrate.png`,
`hls-vmaf.png`, `per_frame_vmaf.png`, `cgan_performance.pdf`, `players.pdf`,
`vmaf-lpips_vs_bitrate_dualrow.pdf` — are from the ACM MM submission. Several
plot numbers from the **retracted** G2 campaign (`NOTE(abstract)` 2026-08-20:
*"NO architecture ranking or selection may be read out of that campaign, in this
paper or anywhere else"*).

**Move them to `archive/figures/`** with a one-line index saying where they came
from and why they may not be re-used. Deleting loses the record; leaving them
beside live figures means someone eventually `\includegraphics` one.

**Then build what the paper actually needs.** A rate-distortion paper with no
rate-distortion figure will read as unfinished whatever its tables say:

1. **RD curves**, PointStream against the anchor, one panel per codec or per
   arm, with the overlap interval shaded — the overlap is a real constraint of
   the BD-rate implementation and showing it is honest rather than fussy.
2. **The payload decomposition**, stacked by plate / residual / appearance /
   metadata across the rungs. This is the paper's clearest single picture: the
   plate is 88–91% of the payload, and a reader who sees that understands the
   whole argument about where the lever is.
3. **A qualitative strip** — source, anchor at matched rate, PointStream, at a
   rate where they differ. A codec paper without one invites the suspicion that
   the output does not survive being looked at.
4. **The system diagram, redrawn** to match what the system now is. `PS-overview.pdf`
   predates the panorama plate, the cross-scene stream and the corrective
   residual's current role.

Generate 1–3 from the run JSONs with a committed script in `experiments/figures/`,
so a re-run regenerates them rather than a person redrawing them. Every figure
carries a `CLAIM` line citing the run it was plotted from.

## B. There is no learned-codec baseline anywhere

Not in `src/`, not in `experiments/`, not in the anchors. The Related Work
section's only learned-video-coding citation is `lu2019dvc` — DVC, 2019.

For a 2026 TOMM submission positioned in semantic and generative coding, that is
the most predictable referee objection in the paper, and it has two honest
answers. Pick one deliberately, in writing:

- **Add one anchor** from the DCVC family (or a learned image codec for the
  plate specifically, which is the component that dominates the payload and is
  where a learned codec would most plausibly beat av1 intra). This is the
  stronger paper and it is real work.
- **Or state why the conventional ladder is the right comparison for this
  claim** — PointStream is a *container* whose all-off corner is a conventional
  codec, so the anchor is the thing it degrades to, and a learned codec is an
  alternative transform rather than an alternative to the decomposition. That is
  a defensible position, but only if it is written down; silence reads as
  oversight.

Either way, refresh the learned-coding paragraph past 2019 and say where
PointStream sits relative to it.

## C. Related work currency and the bound the introduction quotes

`sections/introduction.tex` `NOTE(sec:intro-bound)` and
`sections/related_work.tex` `NOTE(subsec:rw-s2d-bound)` both lean on
**Sparse2Dense's 74.54% BD-rate against VVC**, and the second says in as many
words that *"a PointStream BD-rate far beyond 74.5% on harder content"* would
need explaining.

PointStream currently measures **+90.97%** — the wrong sign, not a smaller
saving. Whatever branch of `plans/FORK-bp31.md` the paper ends on, those two
notes need re-reading against the real number, and the comparison needs a
sentence saying what is and is not comparable: Sparse2Dense codes talking-head
video against VVC; this codes 4K broadcast sport against av1. Different content,
different anchor, different task.

Also sweep the 2026 literature once for anything that has appeared since
2026-08-22, when `ref.bib` was last touched.

## D. Reproducibility, which the abstract already promises

The abstract commits: *"The modular implementation and dataset will be made
available upon acceptance."* Today there is no artefact anyone could release.

An appendix under `appendices/` — the convention is one file per appendix,
`\input` from the `\appendix` block — covering:

- **Exact tool versions.** `AGENTS.md` requires external tools resolved by path
  and version, "this host has carried two builds of the same encoder with
  different capabilities", and `plans/DEFERRED.md` D3 turns on an exact ffmpeg
  build string. Collect them in one table: SvtAv1EncApp, vvencapp, kvazaar,
  x264/x265 via ffmpeg, VMAF model, LPIPS weights.
- **The environment.** `pyproject.toml` pins dependencies; record the Python
  version, the torch build, and the fact that CI runs CPU torch while the runs
  are GPU.
- **How a rung is reproduced**: one command per published number, keyed to the
  `CLAIM` lines.
- **The dataset.** What the tennis corpus is, how scenes are selected, what the
  BP21 cached windows contain, and what can actually be redistributed — the
  source is broadcast footage, so the honest release is probably the annotations,
  the scene index and the code, not the video.

## Done when

- Unreferenced figures are archived with an index; the four live figures are
  generated by committed scripts and carry `CLAIM` lines.
- The learned-codec question is answered in the manuscript, one way or the other.
- The Sparse2Dense bound notes are reconciled with the measured number.
- A reproducibility appendix exists with real version strings in it.
