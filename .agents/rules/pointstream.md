---
trigger: model_decision
description: When working on Pointstream: Object-Centric Semantic Neural Codec
---

# Pointstream System Development

## 1. Execution Blueprint
* **Conda Context:** Run all scripts and terminal pipelines exclusively within the `pointstream` conda environment.
* **Concurrent Session Commit Discipline:** Multiple agent sessions often work in this repo at once, sharing this working directory rather than an isolated worktree. Commit a fix as soon as it passes fast checks (ruff/mypy/unit tests) — never leave it uncommitted while running a slow verification (a multi-minute integration test, a real GPU/pipeline run) or while moving to unrelated work; another session's read-modify-write cycle can silently overwrite it. A surprising empty `git diff --stat` on a file you just edited is the tell that this happened — re-apply and commit immediately rather than diagnosing why.
* **Verification Paths:** The standard evaluation video target is `/home/itec/emanuele/pointstream/assets/real_tennis.mp4`.
* **Real Experiment Mandate:** End-to-end testing must explicitly define `--input /home/itec/emanuele/pointstream/assets/real_tennis.mp4` to bypass mock stream generation fallbacks.
* **Ablation Benchmarks:** For baseline-vs-variant Residual-Guarantee comparisons, use `python -m scripts.benchmark_matrix run config/benchmarks/<spec>.yaml` (matrix spec = base config + per-variant overrides) instead of hand-running configs; it writes the pays-for-itself report under `outputs/benchmarks/`.
* **Scaffold Integrity:** You must maintain the exact architectural layout. Do not dump monolithic scripts. File additions must strictly sit in: `src/main.py`, `src/shared/`, `src/encoder/`, `src/decoder/`, `src/transport/`, `scripts/`, `assets/`, `tests/`, `outputs/`.

## 2. Weight Management & Prototyping
* **Model Mocking Rule:** When introducing a new extractor or neural module, you must follow a mock-first design pattern. Return deterministic dummy PyTorch tensors matching precise target shapes via a `MockActorExtractor` to validate pipeline plumbing prior to loading heavy model weights.
* **Weights Resolution:** Search for native model weights within `/home/itec/emanuele/Models` and symlink them directly to `assets/weights/`.
* **Documentation Sanitization:** Never reference or expose absolute local file paths (e.g., `/home/itec/emanuele/...`) in user-facing documentation or the project `README.md`. Instruct external users to place weights manually in `assets/weights/`.

## 3. Architecture & Code Standards
* **Streaming Window Constraints:** System operates on discrete `VideoChunks` (e.g., 2-second clips). Infinite streaming implementations are prohibited.
* **Paradigm Choice:** Rely on functional programming for stateless data steps, and OOP patterns for stateful engines and core Base Interfaces. Communication structures must use Pydantic models or dataclasses—never raw Python dicts.
* **Tensor Shape Audits:** Every single PyTorch tensor assignment or manipulation statement must include an explicit shape hint comment (e.g., `# Shape: [Batch, Frames, Keypoints, Coords]`).
* **Sparsity & Tracking:** Enforce Event-Driven Sparsity (Non-Uniform Keyframing). Every transmitted semantic block must map explicit tracking metadata via `frame_id` and `object_id`.
* **Symmetric Synthesis:** Ensure the server executes the exact identical standalone `SynthesisEngine` class as the client to predict hallucinated frames and output true generative residuals.
* **Routing Strategy:** Support modular scene classification routing: steer static "Interludes" (e.g., crowd shots) to traditional fallback codecs, and direct active "Exchanges" into the main semantic pipeline.
* **Ablation Protocol:** Benchmark new additions explicitly against the "Whole-Frame Residual Baseline". Measure the specific impact an added extractor has on reducing residual payload size.
* **Multiprocessing Infrastructure:** Use a Directed Acyclic Graph (DAG) using `InlineExecutionPool` or `TaggedMultiprocessPool`. Use `torch.multiprocessing` shared memory (`make_shared_cpu_tensor`) to ship tensors across processes without PCIe or Pickling bottlenecks. Use `@cpu_bound` (FFmpeg/IO) and `@gpu_bound` (PyTorch) decorators.
* **Transport Abstraction:** Network transmission details must be wrapped entirely behind a unified `BaseTransport` interface implementing `.send(payload)` and `.receive()`.
* **Housekeeping:** Keep metadata synchronized by automatically updating `pyproject.toml` and the primary `README.md` file whenever project dependencies or modular structural patterns shift.

## 4. MCP & Structural Reasoning Rules
* **Sequential Thinking Directive:** Before implementing or debugging code related to `Event-Driven Sparsity` or `Non-Uniform Keyframing`, you must run a sequential thinking protocol. Explicitly reason through how keyframe omissions affect the downstream synthesis engine's generative residuals before altering the code.
* **Tensor Shape Auditing:** Use sequential thinking to explicitly deduce the expected vs. actual tensor shapes across your multi-process pipelines (`InlineExecutionPool`/`TaggedMultiprocessPool`) before writing plumbing logic.

## 5. Hardware Allocation & Agnosticism
* **Device Independence:** All PyTorch code must dynamically fallback to a single available CUDA device (`cuda:0`). Do not hardcode multi-GPU requirements (like explicitly assigning `cuda:1`) into user-facing library logic. Use `torch.cuda.is_available()` checks comprehensively.
* **Dual-GPU Target Optimization:** If the host environment detects multiple available GPUs, isolate background worker processes or the decoupled `SynthesisEngine` explicitly onto other GPUs, leaving `cuda:0` unencumbered for frontend client rendering simulation.
* **Documentation Constraints:** The `README.md` must list a standard single-GPU CUDA environment as the base requirement. Keep server-specific multi-GPU tuning configurations strictly inside specialized deployment scripts (`scripts/run_dual_gpu.sh`) or internal agent-facing config modules.