# Experiments: learned player receiver witness design

### 2026-11-16 — reuse, complete contract and bounded pilot

**Design/reuse audit only; zero GPU authority.** The earlier automatic window
expired. No inference, training, scoring, transfer of large assets, or remote
submission was performed. This proposes a smaller development witness, not a
retroactive completion of W8 or confirmation on an independent match. No racket
gate or numerical usefulness floor is added. Current research scope is player
body/clothing plus ball, with equipment harms retained in full-frame review.

## Reuse and exact scope

| Existing component | Reuse | Boundary |
|---|---|---|
| `experiments/tennis/k6b_generate.py:Renderer` | Checked model loading, scheduler, pose/reference conditioning, 512×784 generation, seeded context rendering | Preserve exact preprocessing, CFG, context and checkpoint joins; profile adapters are separate from fixed W8 orchestration. |
| `w8_payload.py`, `w8_inputs.py` | Canonical bounded sections, hashes, compact header, KPS/reference/match reconstruction | Existing `decode_inputs` is explicitly two-player/240-frame/match5, not a general endpoint. Use new version/profile; retain historical decoder unchanged. |
| `w8_receiver_generate.py` | Packet-only preparation, tensor fingerprints, weight SHA inventory; qualification renders first whole 24-frame context | Fixed nad000, match5, 30 steps, CFG2, two units. Full path generates 240 frames but keeps only 80. Do not call its saved ticks an all-frame reconstruction. |
| `w8_receiver_finish.py` | Native conversion, generated-crop SAM, projection/composition, sealing | Fixed native 1920×1080 and sampled ticks; new every-frame compositor is required. SAM runs on generated pixels, not hidden source masks. |
| `w8_witness_namespace.py`, `w8_witness_profile.py`, strict workers/bootstrap | PID reaping, fresh exec, resource/namespace receipts, Landlock and protected-input denials, post-seal assessors | Requalify changed code/resource grants. Synthetic and CPU-only receipts do not qualify changed learned inference. |
| `src/pointstream/codec.py`, `tracked_patch.py` | Actual native stream, 64px tile/32px placement, framing, inspection and byte counts | PTP1 pastes onto its decoded native base; factor/reuse patch decoding and placement for a player-composited frame, rather than serialize a second base. |
| `ptp_isolated_receiver.py` | Actual-source filesystem probes and native inherited restrictions | Explicitly proves neither network nor MCP isolation, nor independent native-child births. Do not inherit a stronger claim. |

Known frozen assets in `w8_input_job.py`: nad000 window
`nadal_kyrgios/000@120`, packet `eff1aca8…bbf4`, profile `bec98d67…1107`,
prepared archive `94a5d631…2bfa`, match fits `cceb49b3…7584`, fit member
`525830aa…f570`, tennis base archive `075b001a…3285`, source archive
`b7d887c0…3f1d`, masks `4eb91e79…49b`, saved generations `cce89567…4985`.
These are input identities, not proof every remote asset is presently readable.
Original weight hashes are enumerated in `w8_receiver_generate.PINNED_WEIGHTS`.
The common tennis checkpoint is 48,878,567 bytes, SHA256
`85e3c65c9777b799f9ed0df3a5c8da567fe8a92e4ac83d115719527acb739eb4`.

Known state from retained records: `20261101T093351Z-24914a4d` passed exact
CPU input equivalence (480 tensors); `20261101T151030Z-df638d2f` failed CUDA
initialization before model construction. Later strict attempts progressed:
`20261105T075639Z-7a7d7655` constructed Renderer but failed source origin;
`20261105T082120Z-df001de2` passed constructor/origin then failed profiler
clocksource enumeration. `20261105T222814Z-8eeacee2` later passed one-step model/denial checks
but failed cost admission. More decisively, `20261106T005057Z-77c276fa`
completed both 1/3-step real stages on exclusive gpu5 Ada GPU1; independently
audited sampling wall was34.289/71.640s, with actual kernels and480 tensors.
This resolves a blanket claim that W8 never passed CUDA, but does not qualify
30-step reconstruction. The two-leaf clock policy and source-runtime identity
are the right reuse lead; 30-step context controller remains new. `20261110T172634Z-0afe4e35`
completed the versioned CPU/lifecycle proof (144.010s created-to-terminal,
not aggregate CPU). Final `20261110T183845Z-4c240783` expired unassigned,
accepted November12, without execution/claim. No active compute release is
inferred. The nine-stage original W8 route remains unqualified; its illustrative
first-six defaults alone were 7,460 CPU/6,900 GPU seconds before assessments.

## Smallest prospective source-independent contract

Use the first 24 native frames of the already exposed nad000 window, both
players, mechanically chosen before new outcomes. Its frozen profile declares
**50/1 fps**, 1920×1080 and context [24,4]; the interval is 0.48s. Preserve
source frame identities and original start mapping. This has directly reusable
player inputs and source footage from which a same-cadence ball channel can be
made. No verified matching saved ball stream was found; the confirmed
RacketVision streams are different sources/cadences and cannot substitute.
Do not assert useful visible-ball coverage before source-only inspection.
If absent/unknown throughout, retain it as an honest correctness case and
mechanically select the next 24-frame block with source-only visible evidence
for the quality screen, recording selection and any changed context inputs.

Packet contains exactly one complete native base; paid pose/keypoint tracks,
reference crops, crop boxes or their deterministic paid reconstruction rule;
player IDs/order; generation and output schedules; profile/resource hashes;
seed/scheduler/steps/CFG/context/color rule; session adapter/guider updates;
optional ball IVF and one placement/null per native frame; all framing/checksums.
Decoder accepts no arbitrary imports or executable paths from the packet.
Installed-resource resolution uses an externally pinned manifest.

Common output timeline is native 50Hz with 24 frames. Cheapest historical-like
mode generates both complete 24-frame contexts and composites player ticks
0,3,…,21; all other frames use the paid native base. This fallback is explicit
and receives all-frame/temporal evaluation; it is not native-rate player
enhancement. Also support an explicit all-frame enhancement schedule without
claiming its quality in advance. Pose uncertainty, missing player crop/mask or
declared generation omission retains native pixels; malformed/corrupt packet or
resource mismatch is an error, never silent success. Record per-frame fallback
reason/count. Native and other controls use identical 50Hz source conversion;
sampled judgments are separately named.

Compose base → players in paid deterministic order → ball rectangle last.
Ball last preserves transmitted ball pixels even where player masks overlap;
it may overwrite player pixels within the square, so score actual combined RGB.
Null ball placement retains the player-composited frame; it does not assert
physical absence. Do not reuse source masks as decoder masks unless serialized
and billed. Primary reference is blinded source-only player union plus visible
ball definition, separate absent/unknown ledger; racket-inclusive results remain
secondary. Actual foreground/complement 70/30 DISTS, LPIPS corroboration and
object/full/temporal reports require a later authorized measurement.

Three profiles are released `original`, original plus the common tennis LoRA,
and original plus common tennis and existing match adapter/guider. The new
`tennis_only` profile uses checkpoint85e3c65c, distinct from the pre-reset
full-weight profile named `finetuned_tennis`. Freeze
profile/checkpoint/guider joins and identical reference/pose inputs. Existing
CFG3.5 stock/base evidence and CFG2 match evidence do not isolate weights;
use common CFG3.5 for the first paired screen, preserving the released stock
default and qualified Ada numeric setting. Label the match profile as changed
from historical CFG2. The parent implementation saves all 24 generated frames,
while the endpoint may still explicitly select sparse enhancement/fallback. No new fitting. No generation need be repeated across
three base rates if its packet conditioning is identical by hash.

Installed frozen generic weights/runtime, including the common tennis resource,
are reported by footprint and cold initialization for all codecs consistently.
Session-chosen/fitted adapters, guider changes, banks, references, poses,
placements, masks and refreshes are paid session state. Historical Al/Med
adapter+guider bills (1,488,252/1,495,592 bytes) are not nad000's measured bill.
Report actual serialized bytes, a cold 0.48s packet rate, and separately
`8*(session_bytes + window_bytes*N)/(N*0.48)` for declared reuse N.
Do not replace full references with modeled twelve-crop reuse or assume 7200s.
Measure cold setup, steady generation, update/load latency and first output
separately; no online/causal adaptation claim follows.

## Avoiding the old runtime trap without reopening every qualification

Stage the exact packed MAIN/JCV environments, source and weights to verified
separate local disk; no per-file NFS runtime, downloads, root scratch or hidden
network dependency. Materialize the VAE/CLIP folders as checked regular files
before restrictive entry: the earlier symlink helper failed `MAKE_SYM`.
Pin the actual AnimateAnyone module tree and validate origin before constructor;
the duplicate site-packages/opt trees caused an earlier false rejection.
Reuse the reviewed public system-leaf inventory and profiler clocksource fix,
not broad `/home`, `/proc`, `/sys` or dataset grants. Before loading expensive
weights, one exact-runtime isolated CUDA init + tiny kernel records provider,
UUID, allocations and actual path errors; fail immediately with preserved trace.
Then the one-step representative model/kernel probe and full 24-frame/30-step
context use the *same* isolation and final runtime. These are future GPU stages,
not permitted now. Reuse historical model-class qualifications for unchanged
numeric components; freshly qualify only changed controller/policy/source joins.

W8's user/network namespace plus restricted filesystem and PID reaper is a
stronger boundary than the public CPU filesystem witness, but it still needs
fresh receipts for this new endpoint. Deny original source/archive aliases,
training/masks/annotations, cached generations and encoder state; exercise
real existing paths and mutation attempts. Clear environment/FDs, prohibit
network and omit MCP credentials/tools from the worker. Host agent MCP access
is not evidence of receiver access; document the actual process boundary.
Run assessment with source only after output sealing, in a separate process.
Check complete native and generated frame coverage, fresh output hashes,
overlap/absent/null/fallback cases, packet tampering/resource substitution,
native children, timeout/kill descendant teardown and claim release.

## Cost forecast and proposed bounded maximum (not authority)

Measured historical leads: Ada 30-step CFG3.5 240-frame unit 661–665s
(2.76s/frame, shared GPU); A6000 CFG2 unit 863s (3.6s/frame, shared GPU).
For 3 profiles × 2 players × 24 frames = 144 generated frames, simple steady
extrapolation is **397s Ada / 518s A6000**, or **596/778s at 1.5 margin**.
The later exclusive W8 240-frame 1/3-step sample slope gives an illustrative
30-step sampling projection `34.289 +29*(71.640-34.289)/2 =575.879s` per
240-frame unit, excluding all fixed/lifecycle costs; it cannot certify a24-frame
context, CFG3.5 or cold profile initialization. These are historical
extrapolations, not measured new runtime,
and do not include cold loads or nonlinear context effects. Ada al028's two
players took 1323.9s full +92.2s smoke; separate SAM full/smoke331/103.6s
and judge283.3/33.8s are additional historical stage timings, not endpoint
lifecycle measurements. Do not linearly shrink the cold portions.

The concrete paired-profile request is now **5,400 GPU seconds and7,200 CPU
seconds**, specified by the [implementation/release card](player-requalification-20261116.md).
It covers three24frame-context profile jobs with representative smoke, complete
input checks, publication and independent audit. Existing cache metadata and
actual frozen input hashes are recorded there. This design's earlier whole-
witness sketch was not a measured forecast; it is superseded by that smaller,
reviewable release. A complete combined witness still needs a new per-stage
forecast after measuring the first controller job. No allocation or automatic
retry follows from this design note.

## Can finish with zero GPU

Implement/version the bounded packet and inspect byte accounting; deterministic
resource/profile schema; native-cadence composition and every-frame fallback;
null/overlap/malformed tests; isolated worker/receipt/config plumbing; cost and
actual-source selection manifests. Use synthetic arrays and existing native
tool checks only. No saved composite may masquerade as fresh learned output.
Real CUDA qualification, three-profile replay, ball sender inference when hints
are absent, SAM, perceptual scoring and the complete learned witness remain
blocked on a concrete new compute release. Source-only human annotation may
also become essential for the scientific screen; infrastructure correctness
can be completed before that input.

Sources: `RESEARCH.md`; PLAN stages A–D; `docs/codec-endpoint.md`;
`submission-combined-foreground-gap-audit-20261115.md`;
`submission-player-model-history-audit-20261115.md`; `w8-wire-audit.md`;
`w8-inputs/receiver-20261101/w8-exact-dispatch-static-plan-20261113.md`;
`w8-inputs/linux-input-20261101/expected-profile.json`;
`docs/fleet.md` model–GPU table; `docs/fleet-recovery-2026-11-12.md`;
the source modules named above. No runtime measurements were made in this audit.

## Concrete paired-worker update and retained small-input addresses

The parent is implementing `player_requalification.py` with common CFG3.5,
512×784, 30 steps, context24/overlap4, all 24 output crops and unchanged Renderer.
Each of three per-profile jobs renders one 24-frame smoke unit plus two full
24-frame units. Thus **216** generated frames including duplicate smoke, not144:
596s Ada /778s A6000 historical steady extrapolation; 1.5 margin894/1167s.
A probe of8frames×1step per profile adds24 one-step frames total, but no valid
measured linear step-cost conversion exists; reserve60 GPU-s per job for probe
and init/teardown rather than call it a measured estimate. Per-profile proposed
**1800 GPU-s /2400 CPU-s**, ≤2400s inclusive lifecycle+caller wall deadline,
smoke≤600s (including cold/probe/first context), full two-context stage≤900s,
validator/publication/teardown/callers≤900s. Three jobs maximum5400GPU/7200CPU
(1.5h/2h), as detailed in the implementation/release card. A broader combined
receiver, ball/SAM/anchor and scoring campaign is **not included** in this
request. These are conservative hard ceilings, **not proven completion
forecasts**. Independent whole-command measurements inside the first job gate
further jobs. Stop after one failed costly gate rather than consume reserve on
retries.

Retained durable small packet:
`/home/itec/emanuele/Datasets/pointstream-data/tennis/w8-linux-input-20261101-eff1aca8/packet-eff1aca8.psw8`
(2,289,836bytes, eff1aca8…bbf4); sibling `profile-bec98d67.json`
(4,869bytes,bec98d67…1107), independently publication-readbacked per
`w8-inputs/linux-input-20261101/publication-receipt.json`. These paths are
retained evidence, not freshly remote-reverified in this audit.

The exact CPU-oracle output publication is
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261101T093351Z-24914a4d/full/published.tar`,
SHA256 `3b80afe8c159b6d572acf24e879e4c924b4c81ced06d33185a83f77b6649fe10`.
Its two documented members are
`publish/nadal_kyrgios__000_at120__person_000000000000.npz` and
`publish/nadal_kyrgios__000_at120__person_000000000001.npz`
(exact flattened layout enforced by `w8_input_job.recover_fixed_publication`;
archive and total regular-member contents are capped at20MiB). Exact file
hashes are `387544a3fc344685f7c07632879affbdafcfd54fd6e1c00e0d9d6c06ca98995d`
and `74be8c9e2f8eb65ca67e6339ef9a2a2f8238df36c60a4b5a07a45fc68d3712a7`.
The independently recovered audit directory is
`/local/users/emanuele/pointstream/audits/w8-input-24914a4d-independent-20261101`;
its report records those hashes and480 tensors. Do not assume its mutable local
scratch remains installed or input NPZ member names without checking.

Common tennis base archive/member: fleet `20261023T194934Z-b0dcbb19/full/published.tar`
under the same inbox, SHA075b001a…3285, member
`publish/tennis-base@fit-100.pt`, SHA85e3c65c…9eb4. Match archive is
`20261031T150848Z-a582ecc0/full/published.tar`, SHAcceb49b3…7584, member
`publish/nk-C@fit-100-fp16.pt`, SHA525830aa…f570. Original weights' declared
root is `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-2026-10-22/inputs/aa/`;
new production configs should resolve authenticated current Models storage
without renaming those historical paths as freshly verified. Full path/hash
inventory is `w8-inputs/receiver-20261101/asset-manifest.json`.
