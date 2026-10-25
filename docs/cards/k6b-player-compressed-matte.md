# Card: CPU-first compressed player mattes

Goal: measure compressed-matte rates, artifact coverage and actual saved-render quality where feasible, following the new preregistered entry in docs/experiments/k6b-players.md. Preserve the frozen12 cohort, original R/S regions and frame sets, native saved crops/background and all earlier evidence.

Scope: CPU only, 7200 aggregate CPU seconds/3 h wall, stages<=45 min, per-window checkpoints. Start H5 area-majority full-frame8/16/32 with LZMA temporal-XOR and COCO-RLE/Zstd of identical grids; bill all 240 frames and player identity/geometry/format metadata. Keep PNG/2 h and real keypoints primary; JPEG accounting conditional. No fitting, generation, new inference, GPU, new chats, agents or monitors.

Stop: complete frozen18/24 rule decides, resource bound expires, artifacts cannot support valid reconstruction, or full neural judging cannot fit CPU. In the latter case finish CPU measurements and prepare a concrete smallest GPU judging request; do not dispatch. Keep source/replay uncertainty, familiar two-match scope, granted matte/background history and historical step5 failure.

Branch: codex/k6b-compressed-matte. Report directly to planning chat 01a16cfc-1a88-7013-bd79-8489fe7c1eb2. Previous PR236 merged after explicit user auto-merge approval; previous compute ledgers stay closed.

State: preregistered; no new measurements/jobs yet.

CPU job 20261025T095256Z-b7692db7 submitted; source `c2f8e670`. No GPU. Preserve and inspect this ID before acting.

## Proposed follow-up — not authorized or dispatched

If the CPU projection is confirmed, complete only saved-generation judging on a newly approved maximum 0.75 GPU-hour ledger. No fitting, new generation or lossy-reference conditioning. Four resumable parts of three frozen windows, each total GPU occupancy <=675 s, correctness smoke<=60 s and full stage<=540 s plus validation/overhead. RTX6000 Ada compatible pool gpu5/gpu6 after rechecking all eligible hosts/existing jobs. Actual frozen LPIPS/DISTS backbones and packed main environment already support this class; assert CUDA/provider/device and memory in the first smoke.

Inputs: all 12 immutable generation archives and clips, four original anchor/mask/pose archives, original judging controls, actual keypoint/PNG-bank packets and new decoder-validated H5/RLE grids, each with SHA256. Recompute all 956 fixed judged frames with original R/S hashes, native source conversion, exact original Telea fill, far-to-near crop order and saved 512×784 RGB rectangles. Keep native clipping/fallback and report matte pixels outside available crop RGB. Score exact-control plus8/16/32 approximations; no quality-based scale selection by window. Validate old-control per-frame absolute differences<=1e-4, every variant finite, all expected frames/players, byte identities, image hashes and ledger entries.

Apply the original >=18/24 strict bracketed rule to each complete scale with actual keypoints,7200 s PNG bank and adapter/guider; common base shipped primary,48,878,567-byte base/7200 s separate. Same-window fixed anchors. Stop if no scale can reach18 even with every remaining cell winning, any validity gate fails, or budget exhausted. Report per match and all cells, including unresolved brackets.

Estimate: old four native three-window judge batches total1077.35 s for three variants; linear four-versus-three metric scaling gives1436.47 s (~0.40 GPU-h), with uncertainty from scene sizes and oracle/source processing. Proposed0.75 h hard cap reserves setup, representative smokes and validation. Every stage saves each complete window. Native completion wakeup remains unverified: attended one bounded fleet wait per batch, no recurring monitor. Source/replay and granted-background/end-to-end limitations persist.

Reference compression is a separate extension: CPU prices support testing half-size JPEG70 (fixed bicubic restoration), native lossy image coding and same-player inter-frame updates; JPEG AI requires a pinned implementation/dependency/cost audit. Actual regenerated quality, true reference lifetimes and a separately authorized generation budget are necessary before any reference-route claim. Existing CPU smoke crops cannot stand in for generation conditioned on these decoded images.

## CPU outcome

Complete native CPU job 20261025T095755Z-f5de5e93; earlier failed 20261025T095256Z-b7692db7 preserved and valid first-window work reused. All12 mask transports measured; independent77 packet and 120 statistical/coverage checks pass. Matte rates1/8:9.995–25.493,1/16:4.531–10.490,1/32:1.341–3.620 kbps. Actual two-frame neural smoke changes quality, with mixed1/8 effects and worse1/16–1/32 on both metrics. Full956-frame CPU neural projection23365–35623 CPU-s exceeds7200; full >=18/24 decision unresolved. Budget closed within cap,0 GPU, no active jobs/monitors. See experiment entry for exact native CPU ledger, all rates/coverage/background limitations and preserved evidence. Proposed0.75 GPU-hour judging stage remains unapproved and unlaunched. Lossy references remain separately untested for generated quality despite roughly90% half-size-JPEG bank byte savings.

Published: https://github.com/emanuele-artioli/PointStream/pull/239, merge-commit auto-merge enabled. CPU evidence and terminal events verified; full judging still requires separate budget approval. Planning report delivered directly.
