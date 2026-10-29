# Card: H5p 4d, a cheaper foreground mask

Effort: high (two new oracles and their rules). Paste everything below the
line.

---

You are an **item session** for PointStream (repo
`emanuele-artioli/PointStream`), running on the server checkout, so the
fleet and `~/pointstream-data` are reachable. Report in your item file and
your last reply.

## Read first
AGENTS.md (Experiments, Evidence, Fleet), `docs/sessions.md`, PLAN.md's
"Where PointStream stands" (egocentric bullet), item H5 (H5p, H5q) and the
ledger's "H5p 4d" paragraph, and `docs/experiments/h5m.md`,
`docs/experiments/h5p.md` (entries 2, 4a, 4b, 4c). Code:
`src/pointstream/` (`mask.py`, `bitstream.py`, `composite.py`),
`experiments/visor/h5.py`.

Branch: `codex/h5p-4d-mask` from `origin/main`. Append entries to
`docs/experiments/h5p.md`.

## Background (for the item file and the user)
PointStream codes the foreground (hands and handled objects, region F) and
the background in separate streams; the client composites them with F. On
egocentric video F is a median 24% of the frame. Every way of sending F
tried so far costs 36–116 kbps on the 4 pilot items: lossless coding of
downscaled frames XORed with the previous one (lzma at 1/8 and 1/16
resolution, H5m and H5p 4a), a video codec on F's signed distance (H5p 2),
and a key colour inside the foreground stream (H5p 4c). Billed, any of
them removes the split's win. The ceiling is H5p 4b: with the exact F at
zero bits the split wins weighted DISTS (−15%, 4 of 4) and still loses
weighted LPIPS (+16%); a post hoc reading puts the mask's budget at about
2 kbps for DISTS to keep its win. So the target is a mask that costs about
2 kbps, or none.

## Goal
Decide, on the 4 pilot items against H5p 4b's ceiling, whether either of
two untested routes brings F under its budget:
- (a) **Binary shape coding with motion prediction**: context-based
  arithmetic coding of the binary mask, intra and inter (MPEG-4 Part 2's
  CAE is the reference design; JBIG2-style contexts are the intra case),
  with the foreground stream's own motion where it helps. CPU only: rate
  at lzma's accuracy (IoU, boundary distance), against lzma's 36–116 kbps.
- (b) **F at the client**: the client re-derives F from what it already
  decodes (the segmenter that made F, run on the decoded foreground and
  background streams), so F costs no bits, or only a small correction
  stream. Oracle: F̃'s accuracy, and the split's BD-rate with F̃ (as H5p 4b
  scores the exact F), against the venue-adapted DCVC-UF at equal
  adaptation (H5q's anchor).
Write each rule, hypothesis, competing explanation and budget in
`docs/experiments/h5p.md` before its run. Run (a) first: it needs no GPU.

## Scope and budget
In: the two routes above, focused tests. Out: the 34-item runs, venue model
changes, tennis (item T1), editing other items' text in PLAN.md. GPU: none
for (a); ≤ 2 GPU-hours for (b) (the segmenter on decoded frames); ask the
user before going over. Every stage ≤ 45 minutes. Stop when both rules
decide, the budget is spent, a decision belongs to the user, or you run out
of ideas.

## Ledger
PLAN.md's dispatch ledger is the single record of open work. When you start,
set your row's status to `running <date>` and push that commit at once. When
you stop, set it to `reported <date>: <one-line outcome>; <GPU-hours used> of
<budget>`, and update your item's bullet in "Where PointStream stands" in one
or two lines. Edit only your own row and bullet.

## Report
Run `ruff check`, `mypy --config-file pyproject.toml`, `python -m pytest`.
Commit, push, open the pull request with auto-merge. Put the report (format
in `docs/sessions.md`, "The report") at the end of your entry and at the end
of your last reply. Then tell the user you are ready to be archived once the
pull request has merged.
