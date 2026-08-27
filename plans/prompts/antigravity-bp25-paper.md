# Prompt for Antigravity — fold BP25 into the paper, plus a step-count appendix

Paste below the line. **Requires PR #27 merged first** — the numbers come from
`PLAN.md` §2.17.

---

You are updating the PointStream paper with the BP25 IP-Adapter result.

**You own exclusively:** the paper repo `67a9ea6275d3d9785ce57026/` — a separate
git repo with its own `AGENTS.md` and commits. **Touch no code file**; BP24 and
BP28 are live in the code repo.

**Read first:** the paper repo's `AGENTS.md`, then `PLAN.md` §2.17 and
`plans/ENGINE-ROSTER.md` in `/home/itec/emanuele/pointstream/`.

## 1. Close the pending marker

`sections/evaluation.tex` carries `NEXT(sec:eval-ip-adapter)` and two more
`NEXT` markers saying the re-score is pending. It has landed. Close them.

## 2. What the result is — and the two ways to overclaim it

The fine-tuned IP-Adapter reaches **0.6922 ± 0.0094** object LPIPS against a
static-copy paste at **0.4505 ± 0.0220** and an unrelated image at
**0.7358 ± 0.0075**.

**The finding is the appearance dependence, not the LPIPS number.** Against its
own keyframe versus a shuffled one, epoch 1 gains **−0.074 LPIPS (3.8σ)** and
**+0.075 reid (3.6σ)** on clip means. This is the first evidence in this project
that any engine uses the reference at all; the standing position was that none
did. Say that.

**Overclaim 1 — "beats an unrelated image."** It does not, at the level that
counts: **1.3σ on clip means**. The item-level figure reads 3.3σ, but eight
offsets inside one clip are not independent. Do not state it as a result.

**Overclaim 2 — "a leader".** 0.6922 is still behind four engines measured in
wave 3 (`upscale-refine` 0.5585, `seg-controlnet` 0.5595, `animate-anyone`
0.5692, `pose-controlnet` 0.6031), and **all of them still lose to the paste**.
The fine-tune moved IP-Adapter from worst-but-one to mid-pack. Frame it that way.

So the scoped negative for `PLAN.md` §7 P0 item 5 stands: **no engine beats a
paste; the appearance path is real.** Both halves.

## 3. Add a step-count appendix

New appendix subsection, as an ablation of diffusion steps — with the
**calibration** framing, because that is what it was:

- Same stock adapter at 4 versus 20 steps separates at **3.5σ** (n=12, offset 8),
  so a 4-step eval is not blind.
- But **at 4 steps the stock adapter scores worse than an unrelated photograph**
  (3.8σ), so a 4-step eval cannot rank models against real-image anchors.
- Therefore all ranking in this paper uses 20 steps.

**Draw the consequence explicitly, because it is a systems result and belongs in
the paper:** 20 steps is roughly **1 s/frame** against a 30 fps target. The only
two entries in the whole roster that run at frame rate are `upscale-refine`
(0.00 s/frame) and `pix2pix` (0.03) — **both non-generative**. The step count
that makes the ranking trustworthy is the same step count that puts diffusion
about thirty times away from real time.

## Rules

- Cite run paths (`outputs/bp25-ip-adapter/`); never paste `outputs/` contents in.
- Quote the instrument's range with every number — an LPIPS figure means nothing
  without the paste and unrelated anchors beside it.
- Report n and standard error on every comparison, and use **clip means**.
- Follow the marker convention and update the reviewer checklist if this closes
  an item.

Report: which markers you closed, the wording you used for the appearance claim,
and anything in the paper that now contradicts §2.17.
