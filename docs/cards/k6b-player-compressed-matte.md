# Card: CPU-first compressed player mattes

Goal: measure compressed-matte rates, artifact coverage and actual saved-render quality where feasible, following the new preregistered entry in docs/experiments/k6b-players.md. Preserve the frozen12 cohort, original R/S regions and frame sets, native saved crops/background and all earlier evidence.

Scope: CPU only,7200 aggregate CPU seconds/3 h wall, stages<=45 min, per-window checkpoints. Start H5 area-majority full-frame8/16/32 with LZMA temporal-XOR and COCO-RLE/Zstd of identical grids; bill all240 frames and player identity/geometry/format metadata. Keep PNG/2 h and real keypoints primary; JPEG accounting conditional. No fitting, generation, new inference, GPU, new chats, agents or monitors.

Stop: complete frozen18/24 rule decides, resource bound expires, artifacts cannot support valid reconstruction, or full neural judging cannot fit CPU. In the latter case finish CPU measurements and prepare a concrete smallest GPU judging request; do not dispatch. Keep source/replay uncertainty, familiar two-match scope, granted matte/background history and historical step5 failure.

Branch: codex/k6b-compressed-matte. Report directly to planning chat01a16cfc-1a88-7013-bd79-8489fe7c1eb2. Previous PR236 merged after explicit user auto-merge approval; previous compute ledgers stay closed.

State: preregistered; no new measurements/jobs yet.

CPU job20261025T095256Z-b7692db7 submitted; sourcec2f8e670. No GPU. Preserve and inspect this ID before acting.
