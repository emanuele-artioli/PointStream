# P1 — Bring the paper level with what we know

**Parallel with everything.** Touches only the paper repo
(`67a9ea6275d3d9785ce57026/`), so it contends with no code stream.

**Owns exclusively:** that repo. **Read first:** its own `AGENTS.md` — separate
git repo, own rules, own commits — then `plans/done/RESEARCH-HISTORY.md` §2.

## Where the paper stands

Related Work is current: `BP6` merged on 2026-08-22 and the Introduction's
novelty claim is repaired around the lattice-corner argument. **Everything since
Wave 1 is missing.** Six things are known and unrecorded, and one of them is a
headline result.

## What to add

### 1. The generative engines do not use appearance — the big one

`plans/done/RESEARCH-HISTORY.md` §2.6. Measured on the coding task, 12 clips, seed 42:

| Arm | Object PSNR |
|---|---|
| static copy — paste the keyframe, no model | **11.82 dB** |
| seg-controlnet | 11.01 dB |
| pose-controlnet | 11.20 dB |

Cause: `scripts/train_controlnet.py` trains on condition + a fixed caption, with
**no reference image**, so the checkpoints synthesise *a* player, never *this*
player.

This is what `plans/done/RESEARCH-HISTORY.md` §7 P0 item 5 called *"a working generative engine, or an
honest scoped negative result."* It is currently the second. Update
`HOLE(sec:evaluation)`, which still says the blocker is that *"no generative
engine has been selected on evidence... best probe scored 15.8 VMAF"* — that is
stale and understates what is now known. **Do not write it as a failure of the
architecture.** It is a property of these checkpoints, and the distinction is the
difference between a scoped negative result and a dead paper.

### 2. Quality is measured region-scoped

The Evaluation section says nothing about it, and it is now architectural
(`plans/done/RESEARCH-HISTORY.md` §6.4): a frame-level score hides a broken object, so object generation
is scored on the object and background on the background, with whole-frame
reported alongside rather than instead. Add it to `subsec:eval-metrics`.

### 3. The static-copy floor belongs in the methodology

Any generative arm must beat pasting the keyframe forward. Cheap, and it is what
exposed item 1. State it as a reported baseline, not as an internal check.

### 4. AVC region control is a no-op under QP

`NOTE(sec:evaluation)` item (c) commits to giving every baseline region control
*"wherever the implementation supports it"*. Verified: ffmpeg `addroi` under
`-qp` produces **byte-identical bitstreams**; it is CRF-only. Say so, and say the
AVC arm therefore uses the pixel-domain path. Appendix~\ref{app:roi} carries the
detail.

### 5. Animate-Anyone has seen the held-out videos

`plans/done/RESEARCH-HISTORY.md` §2.5: its fine-tuning set covers all 7 videos including both held-out
ones. Any AA number is in-domain. `subsec:eval-general` exists precisely to
separate fine-tuned from pretrained, so this constraint must be stated there
rather than discovered by a reviewer.

### 6. Two System Design claims are now verified

`C1`: the all-off corner reduces to the source video, and residual absent versus
lossless changes the payload. `C2`: 16 lattice corners build, and a disabled
detector costs zero. The System Design section asserts both as properties of the
architecture; they now have evidence and can carry `CLAIM` lines.

## Traps

**A number enters the paper only with a `CLAIM` line naming a real `outputs/`
path**, and the `HOLE` it answers is deleted in the same edit. The probe numbers
above are **triage, explicitly not citable** — they belong in `NOTE`/`HOLE`
prose describing the state of the work, **not** in a results table.

**Do not soften item 1 into "further tuning is needed".** Tuning is ruled out on
evidence: no parameter search adds an input the model never had.

**Do not let item 1 leak into claims about the lattice or the residual.** Those
are unaffected and now better evidenced (item 6).

## Done when

- All six land, with markers updated in the same edits.
- `HOLE(sec:evaluation)` describes the current blocker, not July's.
- Nothing triage-grade appears as a result.
- Committed in the paper repo.
