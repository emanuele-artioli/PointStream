# K5c: independent image judge for the ball — proposed bounded pilot

**Status:** user approved this bounded independent-judge pilot on 2026-11-01:
900 GPU-seconds and 1,800 workload CPU-seconds, conditional on source/CPU
checks, a 16-frame smoke and a feasible inclusive projection. No job submitted
yet. Do not resume the stopped K5b second part or spend W8/W9 balances.

**Goal:** establish whether a source-qualified, independently trained WASB
image judge can evaluate the billed streak component, and measure the cost of
that exact path before proposing a larger confirmation. This is a component
correctness/cost pilot, not a complete-codec or unseen-study result.

**Scope:** `experiments.tennis.k5c_independent_ball`, one development clip
`tracknet/game8/Clip6`, registered frames 0–47. Proposed smoke uses its first
16 frames; the pilot uses all 48 only after its correctness and inclusive cost
projection pass. Freeze DCVC-UF HT-L QP 0/21/42 and SVT-AV1 preset 4 CRF
48/55/62, one native point at a time, baseline and the original per-frame
streak only. No color-quantization change, training, model substitution,
additional clip, failed-point retry or automatic larger run. Preserve all
failed artifacts and stop rather than silently dropping a point.

**Approved ceiling:** 900 inclusive GPU-seconds and 1,800 workload
CPU-seconds. Approval does not establish a feasible runtime; all gates below
still apply. The historical 247-second 48-frame smoke used another
parallel configuration and does not forecast this sequential worker.

**Pre-admission work:** audit frozen source, native tools, staged input/frame
and label identities, WASB/BallTrack checkpoint/code identities and the
training-split receipt. The pinned authors' WASB configuration lists games 1–7
for training and 8–10 for testing; this is author-reported provenance, not an
independent training-log audit. The pilot input must match the reported model
and split. Game8/Clip6 has been used in historical development, so it is not an
unseen confirmation sample. Qualify the independent judge on source images
before encoding; fail on fewer than 80% of visible labels within 10 pixels.
Record the actual visible/negative/occluded counts; zero negative examples
cannot establish a false-positive rate.

**What runs:** persist the actual native streams and framed BALL packets;
independently reread them and compare the second native decode with the
encoder-side decode. Render from packet-decoded symbols, not cached fitted
symbols. WASB step 1 receives reconstructed images only; labels and transmitted
coordinates are used exclusively by the later scorer. BallTrack is a secondary
reported judge and remains the encoder position source. Preserve every
registered frame, missed predictions, 5/10-pixel matches, negative-frame false
detections, actual physical byte counts and original-image PSNR. Report crop
LPIPS/DISTS only if the production metrics and their cost are explicitly
qualified; the preparation worker does not currently implement those metrics.
No fresh source-hidden complete receiver or perceptual-realism claim follows.

**Cost and lifecycle gate:** before admission, establish a reviewed sequential
worker, inherited single-CPU affinity for every native/helper child, no
CPU-affinity widening or detached children, and canonical supervisor teardown.
Use `single_cpu_campaign: true`, `cpu_threads: 1` and an inclusive monotonic
campaign limit at most 720 seconds, including staging and validation. Reserve
at most 840 seconds for measured allocation-to-terminal lifetime plus 60 for
release/termination within the 900-second GPU ceiling. The approved CPU
envelope is at most 300 seconds preparation, 840 single-core lifecycle,
180 independent validation, 120 monitoring, and 360 contingency = 1,800.
These bounds are conditional on verified containment/lifetime and must not be
presented as measured aggregate CPU use. Affinity is a cooperative operational
guard, not an adversarial security boundary. Per-process RLIMIT alone is
insufficient. If the descendant/lifecycle bounds cannot be established before
GPU use, stop with a CPU-only preflight finding and revise the proposal.

Smoke must end within 240 seconds; the 48-frame stage within 420 seconds, with
60 seconds reserved inside the campaign for validation/publication. Promotion
requires all substantive checks and a conservative forecast (at least 1.25
margin on the measured variable work, with startup and remaining staging,
judging, independent validation and cleanup charged) that fits both remaining
ceilings. A failed projection is a stop, not authority to enlarge the budget.
Inspect existing jobs and every eligible host before a single submission;
select only previously qualified A6000/Ada model/runtime profiles after a fresh
device/path smoke. Persist job/source/spec/input identities and inclusive charge
before a single bounded canonical fleet wait. Never replay an uncertain submit.

**Decision and stops:** success means usable independent image predictions,
correct physical packet/native replay, complete denominators and a feasible
measured pilot path. A favorable quality point cannot pass the original 6/8
rule; its +1% frame-PSNR BD-rate guard and prior failures remain unchanged.
Use PCHIP over supported quality overlap for any reported BD-rate and the
achievable anchor envelope; otherwise report observed points and unresolved
coverage. Stop for invalid identity, unqualified judge/source, failed native
replay, wrong device/provider, missing frame, invalid metric, packet mismatch,
budget uncertainty, timeout or failed projection. No scale/confirmation follows
without a new reviewed proposal and authority.

**Branch/report:** `codex/paper-strengthening-2026-11-01`; report in
`docs/experiments/k5c-independent-ball.md` and root PLAN ledger. Models/data and
completed outputs remain read-only; any new outputs would use a new Datasets
run directory and fleet-owned scratch, never overwrite prior evidence.
