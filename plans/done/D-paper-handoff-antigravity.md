# Stream D — paper catch-up, for Antigravity

**Wave 4, stream D.** Paper writing is offloaded to Antigravity by standing
practice. This file is the delta to add on top of `plans/done/P1-paper-catchup.md`,
which is still accurate and still the main brief — read it first.

**Owns exclusively:** the paper repo `67a9ea6275d3d9785ce57026/`. It is a
**separate git repo with its own `AGENTS.md` and its own commits** — read those
rules and commit there. Touch no code file in the PointStream repo; three code
streams are live this wave.

## What P1 does not yet know

### 1. The VVC headroom gap did not survive — this is the important one

`BP21` widened the real-4K headroom measurement from n=2 to n=8 and
**pre-registered** the decision rule: if the AVC−VVC foreground gap fell below
0.04, the codec-generation confound is the story rather than a real
object-coding gap.

Measured at n=8:

| Slice | AVC−VVC | n | significance |
|---|---|---|---|
| common QP (32/40/46) | **+0.028 ± 0.015** | 8 | 1.8σ |
| common PSNR (AVC/HEVC/VVC) | **+0.023 ± 0.017** | 8 | 1.3σ |

Both below the pre-registered threshold, neither above 2σ. **The paper may no
longer claim that modern codecs leave object coding on the table**, and any
sentence naming VVC as "the exception" is now wrong. `plans/done/README.md` item 0
and `plans/done/RESEARCH-HISTORY.md` §2.14 both still carry the old n=2 framing — treat those as stale
source text, not as evidence.

What *does* survive: the **concentration** result. Players are ~1% of pixels and
carry a 15–19× concentration of the bitrate, inside its pre-written [10, 60]
band at n=8. The premise that motivates object-centric coding still holds. It is
the *codec-comparison* leg of the argument that does not.

### 2. Foreground-saving means need their error bars and their two clips

AVC foreground plate saving is **0.170 ± 0.031 (n=8)**, below its pre-written
[0.184, 0.304] band; AV1 is **0.154 ± 0.028**, below [0.169, 0.289]. This is not
a measurement error — paste-back MAE was 0.0. Two clips sit near zero
(`djokovic_zverev/scene_002` at 0.011, `federer_djokovic/scene_003` at 0.099)
because on those the original and plate bitstreams are nearly the same size: the
player is 0.54% of pixels and the rest is high-rate content a still plate does
not cheapen.

**Do not cite "17%" as the opening argument without the standard error and those
two clips.** The old n=2 figure quoted only the two highest-saving clips.

Full write-up with every cell: `plans/done/BP21-headroom-widen.md`.

### 3. Background saving came in above its band

AV1 background **0.780 ± 0.056** and VVC **0.761 ± 0.039**, both above the
pre-written [0.25, 0.75]. Recorded as an alarm, not retconned: a still plate
against a 4K intercoded background can save more than the n=2 band allowed.
Report it as measured, with the band it exceeded.

### 4. The generator negative is unchanged, and one arm is still open

Every engine in the roster loses to pasting the keyframe (`plans/done/RESEARCH-HISTORY.md` §2.10). That
is still the honest scoped negative for §7 P0 item 5. **An IP-Adapter arm is
training right now** (wave 4 stream A) and its bounds are pre-written; its
declared ceiling is a *semantic* appearance match, not identity. Leave a
`NEXT` marker for it rather than writing the result — it does not exist yet.

### 5. Housekeeping the Evaluation section should reflect

The pre-rewrite code tree is half retired (213 of 433 legacy tests gone,
`src/encoder` and `src/main.py` deleted). This is not a paper claim; it matters
only if the paper describes the codebase's structure anywhere.

## How to work

Follow the paper repo's marker convention (`STATUS`/`GOAL`/`HOLE`/`NOTE`/
`NEXT`/`CLAIM`). Every claim must match real measured evidence — cite the run
paths, never paste `outputs/` contents into the paper. Where a number's
provenance is a single clip or n=2, say so in the text.

**The standing rule for this project applies hardest here:** a result outside a
pre-written bound is an alarm, and the paper reports it as measured with its
band, rather than quietly quoting the friendliest cell.
