---
paths:
  - "src/**"
---

<!-- GENERATED — DO NOT EDIT. Source: AGENTS.md via tools/sync_agent_rules.py
     The 'Architecture rules' section. Scoped so it costs no context until
     Claude reads a file it actually governs. -->

## Architecture rules

- **Strict scaffold** — no monolithic scripts: `src/main.py`, `src/shared/`
  (schemas, `SynthesisEngine`, interfaces, geometry), `src/encoder/`
  (extractors, DAG orchestrator, residual), `src/decoder/` (renderer, GenAI
  compositor/engines), `src/transport/`, `scripts/`, `tests/`, `assets/`,
  `outputs/`.
- **Symmetric synthesis:** never fork `SynthesisEngine` behavior between
  encoder and decoder — the Residual Guarantee breaks the moment server and
  client can disagree. Seeded determinism (`seed` config key) matters for the
  same reason.
- Operate on discrete `VideoChunk`s (~2 s clips); no infinite-stream code
  paths.
- Cross-module data uses the Pydantic models in `src/shared/schemas.py`
  (`VideoChunk`, `ActorPacket`, `FrameState`, keyframe/interpolate/static
  events, `EncodedChunkPayload`, …) — never raw dicts. Every transmitted
  semantic event carries `frame_id` and `object_id`.
- **Every tensor statement carries a shape-hint comment**, e.g.
  `# Shape: [Batch, Frames, Keypoints, Coords]`.
- **Mock-first:** a new extractor/neural module first returns deterministic
  dummy tensors of the exact target shape (see `MockActorExtractor`) so the
  plumbing is proven before heavy weights load.
- DAG execution via `InlineExecutionPool`/`TaggedMultiprocessPool`; tag nodes
  `@cpu_bound` (I/O, FFmpeg) or `@gpu_bound` (PyTorch) from
  `src/shared/tags.py`; pass tensors between processes via
  `torch.multiprocessing` shared memory (`make_shared_cpu_tensor`), not
  pickling.
- All transmission goes behind `BaseTransport` (`.send(payload)` /
  `.receive()`, `src/shared/interfaces.py`); currently only `DiskTransport`.
- Scene-classification routing stays modular: static "Interludes" (crowd
  shots) → traditional fallback codec; active "Exchanges" → semantic pipeline.
- Device-agnostic CUDA: fall back to single `cuda:0` with
  `torch.cuda.is_available()` checks; never hardcode `cuda:1`/multi-GPU in
  library code (multi-GPU tuning lives in scripts/config only). **The GPU is
  shared** (48 GB, other processes present) — SPADE at batch 16 / 512 px OOMs.
