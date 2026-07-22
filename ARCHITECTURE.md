# POINTSTREAM architecture

A map of what lives where, and how a chunk flows through it. CI checks that
every module under `src/` appears here, so this cannot quietly go stale.

## Dataflow

```
  input video ──► scene_classification ──┬── Interlude (crowd shot) ──► fallback codec
                                         │
                                         └── Exchange ──► semantic pipeline
                                                              │
   ┌──────────────────────────── ENCODER ─────────────────────┴──────────────┐
   │  actor extraction (detect ─► track ─► segment ─► pose)                   │
   │  ball extraction                                                         │
   │  background modelling ──► panorama                                       │
   │  SynthesisEngine  ──► the server's own reconstruction                    │
   │  residual_calculator  ──► original − reconstruction                      │
   └──────────────────────────────────┬───────────────────────────────────────┘
                                      │  EncodedChunkPayload (Pydantic)
                                      ▼   over BaseTransport (DiskTransport)
   ┌──────────────────────────── DECODER ─────────────────────────────────────┐
   │  SynthesisEngine  ──► the *identical* reconstruction                     │
   │  genai compositing (ControlNet / AnimateAnyone / SPADE / pix2pix)        │
   │  + residual  ──► output frames                                           │
   └──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         run_summary.json + invariant_failures
```

The Residual Guarantee rests on the two `SynthesisEngine` calls being
byte-identical. Any behaviour that can differ between encoder and decoder
breaks it.

## Modules

### Entry and orchestration
| Module | Responsibility |
|---|---|
| `main.py` | The only CLI (`--input`, `--config`); builds and runs the pipeline. |
| `experiment_evaluation.py` | Metrics and summary writing for a completed run. |
| `encoder/orchestrator.py` | Drives one chunk through the encoder stages. |
| `encoder/match_orchestrator.py` | Drives a whole match: scene routing, chunking. |
| `encoder/dag.py` | Stage graph with `@cpu_bound` / `@gpu_bound` tags. |
| `encoder/execution_pool.py` | Inline and tagged-multiprocess execution pools. |
| `encoder/pipeline_builders.py` | Assembles components from config. |

### Encoder stages
| Module | Responsibility |
|---|---|
| `encoder/actors/weights.py` | Weight resolution and bbox geometry, used by all the actor backends. |
| `encoder/actors/detection.py` | YOLO backends that find people and rackets. |
| `encoder/actors/heuristics.py` | Deciding which detections are the players, not the ball kids or crowd. |
| `encoder/actors/segmentation.py` | Masks that drive compositing. |
| `encoder/actors/pose.py` | Keypoints used as generative conditioning. |
| `encoder/actors/payload.py` | Packing actors into the transmitted payload. |
| `encoder/actors/builder.py` | Assembling an actor pipeline from config (backends selected by name). |
| `encoder/actor_components.py` | Shim re-exporting the actors package; `actor_pipeline` and a test both bind to this path. |
| `encoder/actor_pipeline.py` | Runs actor extraction over a chunk. |
| `encoder/ball_extractor.py`, `encoder/segmentation_ball_extractor.py` | Ball localisation. |
| `encoder/reference_extractor.py` | Reference crops for the generative backends. |
| `encoder/background_modeler.py` | Background estimation feeding the panorama. |
| `encoder/residual_calculator.py` | Importance mapping and residual computation. |
| `encoder/residual.py` | Residual encode/decode. |
| `encoder/anchor_cache.py` | Caches anchor encodes across runs. |
| `encoder/video_io.py` | Frame reading and writing. |

### Decoder stages
| Module | Responsibility |
|---|---|
| `decoder/decoder_renderer.py` | Reconstructs frames from the payload. |
| `decoder/compositor.py` | Composites actors onto the background. |
| `decoder/compositing/weights.py` | Weight resolution for the generative backends. |
| `decoder/compositing/pose_render.py` | Pose tensors → the conditioning images every engine consumes. |
| `decoder/compositing/strategies.py` | The generative strategies a run selects by config name. |
| `decoder/compositing/compositor.py` | `BaseCompositor` and `DiffusersCompositor` — the decoder half of the Residual Guarantee. |
| `decoder/genai_compositor.py` | Shim re-exporting the compositing package; the engine modules bind to this path. |
| `decoder/controlnet_engine.py` | ControlNet inference with temporal conditioning. |
| `decoder/animate_anyone_runtime.py` | Animate-Anyone backend. |
| `decoder/pix2pix_engine.py`, `decoder/spade4tennis_engine.py` | GAN backends. |
| `decoder/attention_injection.py` | Cross-frame attention for temporal coherence. |

### Shared
| Module | Responsibility |
|---|---|
| `shared/synthesis_engine.py` | The deterministic reconstruction both sides run. **Never fork its behaviour.** |
| `shared/schemas.py` | Pydantic models for everything crossing a module boundary. |
| `shared/interfaces.py` | `BaseTransport` and the component protocols. |
| `shared/config.py` | Config loading and validation. |
| `shared/geometry.py` | Coordinate and box maths. |
| `shared/tags.py` | `@cpu_bound` / `@gpu_bound` markers. |
| `shared/mask_codec.py` | Mask serialisation for the payload. |
| `shared/track_id.py` | Stable `object_id` assignment across frames. |
| `shared/scene_classification.py` | Interlude vs Exchange routing. |
| `shared/player_extraction.py`, `shared/racket_heuristic.py` | Player and racket heuristics. |
| `shared/dwpose_draw.py` | Pose rendering for the conditioning images. |
| `shared/hnerv_arch.py`, `shared/spade4tennis_arch.py` | Learned-codec and GAN architectures. |
| `shared/tennis_dataset.py` | Training dataset assembly. |
| `shared/fvd.py`, `shared/lpips_metric.py` | Perceptual metrics. |
| `shared/invariants.py` | The methodology rules as code: real input, quality actually measured, payload accounting, the Residual Guarantee. Writes `invariant_failures` into each run summary. |
| `shared/profiling.py`, `shared/genai_debug.py` | Timing and debug dumps. |
| `shared/torch_dtype.py` | Device and dtype helpers. |

### Transport
| Module | Responsibility |
|---|---|
| `transport/disk.py` | `DiskTransport`, the only `BaseTransport` implementation. |
| `transport/panorama_encoder.py` | Panorama coding and the background-layer ladder. |

## Contracts worth knowing before changing anything

- **Encoder and decoder must run the identical `SynthesisEngine`.** The moment
  they can disagree, the residual computed at the server stops restoring the
  client's output and the Residual Guarantee is gone. Seeded determinism
  matters for the same reason.
- **A component earns its place only if it shrinks the payload by more than the
  metadata it adds.** Benchmark every addition against the Whole-Frame Residual
  Baseline.
- **Cross-module data is Pydantic, never raw dicts**, and every transmitted
  semantic event carries `frame_id` and `object_id`.
- **Generative backends are selected by name string from config**, so a
  factory's string→class mapping is part of the config contract: renaming a key
  silently changes which backend runs.
- **A run with a non-empty `invariant_failures` is not citable.**
