# Prompt for Antigravity — wave 5, stream F (paper)

Paste everything below the line.

---

You are **wave 5 stream F** on the PointStream project, working in
`/home/itec/emanuele/pointstream`.

**Read these first, in order:**
1. `/home/itec/emanuele/.agent-rules/AGENTS.md` — host rules
2. `/home/itec/emanuele/pointstream/AGENTS.md` — project rules
3. `/home/itec/emanuele/pointstream/plans/WAVE-2026-08-26.md` — the wave plan
4. `/home/itec/emanuele/pointstream/plans/P1-paper-catchup.md` — your main brief
5. `/home/itec/emanuele/pointstream/plans/D-paper-handoff-antigravity.md` — the delta on top of it
6. `PLAN.md` §2 — in particular §2.10, §2.14, §2.16

**You own exclusively:** the paper repo `67a9ea6275d3d9785ce57026/`. It is a
**separate git repo with its own `AGENTS.md` and its own commits** — read those
rules and commit there. **Touch no code file** in the PointStream repo; five code
streams are live this wave. If you need a code change, write it in your report.

## Your task

Bring the paper level with what is now known. Six things are measured and
unrecorded. Follow the marker convention (`STATUS`/`GOAL`/`HOLE`/`NOTE`/`NEXT`/
`CLAIM`) and update the reviewer checklist when an edit closes an item.

**The four highest-value edits, in order:**

1. **Retract the VVC leg of the headroom argument.** BP21 pre-registered that an
   AVC−VVC foreground gap under 0.04 means the codec-generation confound is the
   story. Measured at n=8: **+0.028 ± 0.015** at common QP (1.8σ) and
   **+0.023 ± 0.017** at common PSNR (1.3σ). Both below, neither above 2σ. The
   paper may **no longer claim modern codecs leave object coding on the table**,
   and any sentence naming VVC as "the exception" is now wrong. Note that
   `plans/README.md` item 0 and `PLAN.md` §2.14 still carry the stale n=2
   framing — treat those as source text to correct, not as evidence.

2. **What survives, and say it precisely.** Concentration holds: players are ~1%
   of pixels carrying a **15–19×** concentration of bitrate, inside its
   pre-written [10, 60] band at n=8. The premise that motivates object-centric
   coding is intact. Foreground saving means need their error bars — AVC
   **0.170 ± 0.031** against a pre-written [0.184, 0.304]; **do not cite "17%"**
   without the SE and the two near-zero clips (`djokovic_zverev/scene_002` at
   0.011, `federer_djokovic/scene_003` at 0.099). Paste-back MAE was 0.0, so this
   is real, not measurement error.

3. **The platform now runs end to end** (`PLAN.md` §2.16, P0 item 1 closed). All
   three tiers plus two controls produced real PSNR/SSIM/VMAF/LPIPS on a 4K clip.
   **Critical framing: these are not rate points.** No encoder binary runs, so
   byte counts are pixel payload and no compression ratio may be quoted yet.
   A second independent confirmation of §2.6 also landed: the unaided static plate
   scores 34.88 dB on the frame but **14.30 dB on the object**, a 25 dB gap on
   0.57% of pixels.

4. **The generator negative, stated at its true width.** Every engine loses to
   pasting the keyframe (§2.10) — that stands. But the IP-Adapter arm is **not**
   yet a result: it trained and self-stopped, and its eval generates at 4
   diffusion steps scored against undegraded images, so it cannot rank models.
   Its own artifacts say `not_citable`. Leave a `NEXT` marker for it. **Do not
   write an IP-Adapter verdict** — stream B is measuring it this wave.

**Also worth a `NOTE`:** VMAF's ceiling on this content is 97.54 (not 100) and it
floors at 0.00 for both severe blur and unrelated content; LPIPS's ordering
inverts at 960×540 and holds at 4K. Both belong in the methods section, because
two metrics here were broken until 2026-08-23 and every ranking before that date
is void.

## Rules that matter more than speed

- **Every claim must match real measured evidence.** Cite run paths; never paste
  `outputs/` contents into the paper.
- **Where a number's provenance is n=2 or a single clip, say so in the text.**
- **A result outside a pre-written bound is reported as measured, with its band** —
  not quietly replaced by the friendliest cell. Two of BP21's means are outside
  their bands and stay that way in print.
- Paper text keeps its academic register; the plain-language rule is for chat and
  commits.

Report back: which markers you closed, which claims you retracted or narrowed,
and anything you found in the paper that contradicts `PLAN.md` §2.
