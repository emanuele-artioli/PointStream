# PointStream and demo unification

Living record. Started 4 October 2026. It supersedes the 30 September
submission schedule in `docs/roadmap.md` and `PLAN.md`, and the rule that the
second domain waits for a claimable tennis point.

## Decisions (user, 4 October 2026)

- **No fixed submission date.** TOMM accepts submissions continuously. The
  internal date is set at the Phase 3 gate, from the research questions that
  have evidence by then.
- **Two co-equal domains:** broadcast tennis and egocentric hands
  (Egocentric-10K). Both use one pipeline, one byte ledger and one evaluation
  contract.
- **In-flight work is redirected into the unified pipeline now:**
  - the Gate-A native VVC/AV1 comparison;
  - the demo's HNeRV/DCVC background readiness work.

## Audit summary

| Axis | Main (`src/`) | Demo (`demo/`) | Unified direction |
|---|---|---|---|
| Structure | Contracts, registry, stage DAG | About 52 standalone scripts | Port reusable parts into `src/` components; `demo/` keeps the pitch site and teleop story |
| Domain | `TENNIS` (PTZ camera), `GENERAL` | Egocentric, free-moving camera | `EGOCENTRIC` profile with the `hand-21` schema (done) |
| Pose | COCO-WholeBody-133 canonical | Pretrained RTMW-l, chosen visually, no labels | `rtmw` pose backend; projection from WholeBody hand banks (done) |
| Metadata | Raw float16 keypoints, JSON boxes | PK, DWB2, PSFG v2 | `objectstream.v1` (done, see below) |
| Background | Panorama plate, sidecars, AV1 stream, VVC residual | Downscaled SVT-AV1 ladder; DCVC-UF and HNeRV unranked | `stream-ladder` plus `hnerv`/`dcvc-uf` backends for both domains |
| Generation | Off by default; no generator beats paste | RGBA hand UNet/SPADE; flawed alpha objective | One trainer, alpha-supervised RGBA head, domain dataset adapters |
| Evaluation | Weighted PSNR, VMAF, BD-rate guards, byte-only receiver | Task metrics (detection + MPJPE + PCK); fallback to original landmarks | One contract: common metrics plus a domain-primary metric; byte-only receiver only |

The demo trained hand *generators*, not pose models. The tennis analogue is a
racket-aware player generator.

## Done

### Egocentric domain (`788c311`)

- **Schema:** `hand-21` in `src/contracts/keypoints.py`, side-agnostic, with
  20 edges. `hand_from_wholebody(side)` projects losslessly from the
  WholeBody-133 hand banks.
- **Profile:** `EGOCENTRIC` in `src/contracts/domain.py` (free-moving camera,
  so panorama backgrounds are rejected). Registered as domain backend
  `egocentric`.
- **Manifest:** `src/components/domain/datasets/egocentric.yaml`.
  - Clips are identified by raw source video, not by the demo's `curated/`
    names. On gpu5 those curated files have the same byte sizes as the raw
    videos. `curated/manifest.json` later reused the same names for other
    factories, which is the wrong-clip risk recorded in `912b0d7`.
  - **Development hold-out:** the last 10 s of each development video. These
    are the exact windows in `pointstream-demo/holdouts/holdouts.json`, which
    are already exposed.
  - **Validation:** the preceding 10 s.
  - **Fit:** the rest of each video.
  - **Final hold-out:** factory_003. It is referenced by no branch before this
    manifest.
- **Guard:** `iter_dataset` never yields final-hold-out clips unless
  `include_final_holdout=True`.
- **Caveat:** checkpoints trained before this manifest sampled whole videos.
  They are not governed by these windows.

### `objectstream.v1` (`a281a9d`)

`src/components/transport/objectstream_codec.py` separates two layers:

- **Quantizer.** Grids are frame, pixel (`2**bits` per px) or box-relative.
  These reproduce PK/DWB2 u8 and PSFG 1/16 px precision.
- **Lossless entropy layer.**
  - Each track opens a segment with an absolute record.
  - Later records choose absolute or delta per record.
  - Prediction is closed-loop, with optional box-motion compensation.
  - Residuals use Exp-Golomb-k, with k chosen per segment for boxes and for
    joints.
  - An unchanged present set costs 1 bit.
  - CRC32; segments decode on their own.

Measured on retained data. These are local CPU checks, not paper evidence.

| Input | Reference bytes | objectstream bytes | Note |
|---|---:|---:|---|
| Egocentric clip_01 DWB2 (`/private/tmp/pointstream-pose-packing-inputs/dwpose.bin`), 300 frames, hands/face/body codes | 16,876 | 16,618 | Lossless on codes, DWB2 slot matching |
| Same, with causal box-distance track association | 16,876 | 16,574 | Association adds under 1% |
| Same, box re-anchored joint prediction | 16,876 | 18,152 | Measured negative; off by default |
| Synthetic rigid-motion hands at 1/16 px (test fixture) | PSFG delta+zlib 2,359 | 2,042 with box-motion prediction (3,382 without) | Synthetic; not evidence |

Interpretation:

- The win over DWB2 is small (1.5–1.8%) at identical codes.
- The real rate lever is quantization precision. On the same data, 1 px
  pixel-grid joints cost about 21 kB with at most 0.5 px joint error. DWB2's
  grids have a box step of about 7.5 × 4.2 px and a median joint step of
  about 2 px.
- Context-adaptive arithmetic coding is the next entropy-layer candidate. It
  is unmeasured.

Still to do for metadata:
- PSR1 masks from the Gate-A branch as a section in the same container.
- Quantized homographies.
- Ledger mapping in `src/runner/accounting.py`.
- Client envelope integration.

### Background provenance blocker: diagnosis (4 October 2026)

Cause of the 30 s `dcvc-git-diff` timeout in job `20261004T195542Z-f8714ffc`.
All checks were direct, bounded, read-only Git commands on gpu5 against
`pointstream-data/jobs/neural-bg/src/DCVC`: 342 tracked files, 10.9 MB tracked,
1.7 GB tree with untracked `third_party/cutlass/` and checkpoints. No GPU was
claimed and no fleet job was submitted.

| Command (gpu5, shared NFS checkout) | Seconds |
|---|---:|
| `git --no-optional-locks diff --quiet HEAD`, first run / repeat | 0.078 / 0.010 |
| `git --no-optional-locks status --porcelain -uno` | 0.011 |
| `git --no-optional-locks status --porcelain` (with untracked) | 0.256 |
| Plain `diff` / `status` under the inbox worker's exact environment | 1.22 / 1.26 |
| `git --no-optional-locks diff-index -p HEAD --`, minutes later | 9.27 |

Findings:
- **Not the cause:** the inbox worker carries no `GIT_*` variables and uses
  the same `/usr/bin/git` 2.34.1. The repository is not inherently slow.
- **What varies:** the cost of the same read-only Git operation ranged from
  0.01 s to 9.3 s within minutes. It depends on the NFS stat-cache state of
  an index last written from another host.
- **What the job's commands do:** plain `status`/`diff` also rewrite the
  shared index, which provenance must never do.
- **Side effect of this diagnosis:** one plain run under the worker
  environment refreshed the DCVC index stat cache. Content was unchanged;
  the tracked patch SHA-256 before and after is
  `aa4bee51...6e3dc01e`, identical between `diff-index` and porcelain `diff`.
- **Status:** the 30 s stall itself was not reproduced. The most plausible
  mechanism is the same stat-driven rehash or index write over loaded NFS.

Changes made:
- **Read-only Git:** `src/utils/git_readonly.py`, used by the background
  smoke environment probe and the SAM 3.1 source check.
  - It always passes `--no-optional-locks` and `GIT_OPTIONAL_LOCKS=0`.
  - It produces patches with plumbing `diff-index -p`, which gives the same
    bytes as before.
  - This stops the mutation but does not bound the latency.

Required fix (next):
- Snapshot each external checkout once per host as an immutable archive of
  `HEAD` plus the tracked patch, with SHA-256s and a receipt, outside timed
  smoke parts.
- Jobs extract it to local scratch and verify the archive hash. NFS Git
  leaves the job path.
- This is the environment-snapshot item of the plan. It precedes any further
  B1 attempt.

## Next (Phase 0 continuation)

1. Gate-A: merge the reusable code from `codex/gate-a-local-confirmation`
   (`src/runner/{packet_packing,mask_rle}.py`, `experiments/packet_study/`,
   native anchor v2) through a reviewed branch. Inspect the existing
   QP52f48 job without replaying it. VVC QP52f96 and the AV1 overlap ladder
   become the first jobs of the shared anchor service.
2. Background: the timing check is done (above). Implement the per-host
   environment snapshot. Then run
   B2/B3/B4 through the new backends within the remaining budget
   (1,520 GPU-s / 1,035 CPU-s). Training needs explicit approval.
3. Review the unpushed Cursor branches; preserve the dirty Desktop checkout.

Later phases are as in the approved plan:
- Phase 1/2: components.
- Phase 3: unified measurement; the date is set here.
- Phase 4: training, with approval.
- Phase 5: paper.

## Local test environment

The Mac miniconda base lacks PyYAML, msgpack, torch and diffusers. Focused
tests ran in a scratch venv with PyYAML and msgpack added. Torch-only modules
and the local ffmpeg ROI test cannot run here. Two canvas tests also fail on
base commit `9ac6b5b`.
