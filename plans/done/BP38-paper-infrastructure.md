# BP38 — Paper infrastructure: figures, baselines a referee expects, reproducibility

Current status (2026-09-03): the separate `tex` conda environment builds the
PDF. Follow `PAPER-NEXT.md` for remaining writing and page-budget work; do not
repeat toolchain setup. Learned-codec comparison remains gated on first-domain evidence.

**Gate update (2026-09-02):** build repair, claim cleanup, figure inventory and
reproducibility packaging may proceed now. Add the DCVC-RT learned-video-codec
baseline only after Gate B in `plans/ROADMAP.md`; DCVC-UF is a stretch goal.

**Everything the manuscript needs that does not depend on a result.** Doing it
now means that when BP31 and BP41 land, the paper is a writing job rather than a
build job.

**Owns:** the paper repo `67a9ea6275d3d9785ce57026/` — `figures/`,
`appendices/`, `sections/related_work.tex`, `ref.bib` — plus
`experiments/figures/**` (new) in the code repo. **Commit in the paper repo, not
the parent.** Its `AGENTS.md` and marker convention govern there.

**Read first:** the paper repo's `AGENTS.md` and `sections/README.md` ·
`appendices/README.md` · `plans/ROADMAP.md` §6.

**Mixed dependency.** Paper/build infrastructure has none; the learned baseline
follows Gate B.

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

## B. Add a DCVC-class anchor — decided 2026-09-02

There is no learned-codec baseline anywhere: not in `src/`, not in
`experiments/`, not in the anchors, and Related Work's only learned-video-coding
citation is `lu2019dvc` — DVC, 2019. For a 2026 TOMM submission positioned in
semantic and generative coding, that is the most predictable referee objection in
the paper.

**The decision is taken: add one.** Not a paragraph explaining why the
conventional ladder suffices — an actual anchor, on the same clips, at the same
rungs, in the same tables.

**Which one, and the shape of the work.** Two candidates, and they answer
different questions:

- **A learned *image* codec on the plate.** The plate is 88–91% of the payload
  and is a single still, so this is where a learned codec has the clearest shot
  at beating av1 intra, and it drops straight into the existing sidecar interface
  beside `jpeg`/`av1`/`vvc` (`src/components/background/sidecar.py`). Cheapest to
  wire, and it strengthens the component the whole paper turns on.
- **A learned *video* codec as a whole-clip anchor**, DCVC-family, beside
  AVC/HEVC/AV1/VVC. This is the one a referee is actually asking for: it says
  where PointStream sits against modern learned coding, not just against
  conventional coding.

**Do both if time allows; do the video anchor first if not**, because it is the
one the objection is about. The sidecar arm is a component result; the whole-clip
arm is a positioning result.

**Constraints that will bite, so plan for them.**

- **Do not install into the `pointstream` env.** Host rules are explicit and
  several forked models here are version-sensitive. A DCVC reference
  implementation will want a newer torch. This is the same second-env problem as
  `DEFERRED.md` D2 (SAM3), and solving it once serves both — that is an argument
  for doing it properly rather than twice.
- **Resolve it by path and version** like every other encoder here, and record
  the checkpoint. A learned codec's weights are its version.
- **It must be swept, not run at one point.** The currency is BD-rate; a single
  operating point compares nothing. And it reports **encode and decode time**
  beside rate and quality — learned codecs are slow, that is part of the honest
  comparison, and `AGENTS.md` now requires all three dimensions on every result.
- **Bound it before believing it.** Published DCVC-class results claim large
  BD-rate gains over conventional anchors on standard test sets at 1080p. On 4K
  broadcast sport, on this hardware, expect **[−40%, +30%]** against the av1
  preset-10 anchor. A number far outside that is more likely a resolution or
  colour-space mismatch than a result — the same class of trap as
  `plans/done/BP24-findings.md` §12 (RGB-PSNR against a 4:2:0 codec).

## C. Related work currency## C. Related work currency and the bound the introduction quotes

`sections/introduction.tex` `NOTE(sec:intro-bound)` and
`sections/related_work.tex` `NOTE(subsec:rw-s2d-bound)` both lean on
**Sparse2Dense's 74.54% BD-rate against VVC**, and the second says in as many
words that *"a PointStream BD-rate far beyond 74.5% on harder content"* would
need explaining.

PointStream currently measures **+90.97%** — the wrong sign, not a smaller
saving. Whatever branch of `plans/done/FORK-bp31.md` the paper ends on, those two
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
- A DCVC-class anchor is swept on the same clips and rungs as the conventional
  ladder, with rate, quality and time in one table, and Related Work's
  learned-coding paragraph is current past 2019.
- The Sparse2Dense bound notes are reconciled with the measured number.
- A reproducibility appendix exists with real version strings in it.
