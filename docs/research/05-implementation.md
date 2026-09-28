# Implementation

This chapter records the runnable contracts behind the method in [04-method.md](04-method.md). It separates implementation from evaluated evidence: code paths establish what can run and what is counted, while only a reproducible run with source identity, configuration, codec versions, and delivered-frame scores can establish empirical performance. See the [evidence ledger](08-evidence-ledger.md) for the evidence state and [experiment plan](07-experiment-plan.md) for planned measurements. In the descriptions below, implemented means present in source; measured means supported by a run artifact; proposed means an experiment or paper claim that still needs that evidence.

## Encoder, payload, and client path

`src/runner/run.py` validates the configuration, binds components lazily, builds one shared stage context, and runs the configured stage graph chunk by chunk. The default runtime can receive supplied `ObjectRequest` subjects or obtain subjects through registered perception components. Disabled stages are not implicitly replaced with their intended behavior. The runner then builds a client envelope, reconstructs through that envelope, scores the reconstructed base and delivered frames, and reconciles the byte ledger ([runner](../../src/runner/run.py#L181), [stage bindings](../../src/runner/stages.py#L251)).

The client envelope is schema-versioned JSON metadata plus named NumPy arrays in an NPZ byte stream. It carries coded background packets and geometry headers, object-keyed appearance references, placements, optional masks and generation conditions, residual bitstream or explicitly raw residual data, and generator identity when generation is enabled. The serialized decoder validates the envelope and reconstructs without source pixels or encoder-side objects. It decodes the reference image and background payloads before composing the scene. A request requiring generation also needs a resolvable generator identity/checkpoint. This source-free path is why scoring a server-side intermediate alone is insufficient ([serializer](../../src/runner/client.py#L190), [decoder](../../src/runner/client.py#L449)).

The implemented run path can be summarized as:

```text
for each source chunk:
    observations = supplied subjects or configured perception stages
    base_prediction = reconstruct background + decoded references/conditions
    signed_residual = source - base_prediction
    residual_payload = configured gate/downscale/representation
    transmitted_residual = native encode, then decode the produced bitstream
    request = serialize background + references + placements + residual + metadata
    delivered = independent_client_decode(request)  # no source pixels
    score source against base_prediction and delivered
    reconcile named byte costs against the complete request
```

The loop describes the contract rather than claiming every stage is enabled in every experiment. In particular, the low-rate sweep disables generation, and an unencoded residual is reported as raw data.

The accounting charges all transmitted elements. Panorama bytes come from the actual background payload, actor-reference bytes from the transmitted appearance representation, residual bytes from the transmitted payload, and metadata includes the residual envelope remainder after those payload charges. The subledger further identifies masks, pose/motion arrays, placement headers, generator metadata, and envelope overhead. The ledger marks raw array components, preventing a mixed raw/coded total from being presented as a codec rate ([accounting types](../../src/runner/accounting.py#L84), [stage cost collection](../../src/runner/stages.py#L1275)).

For coded lossy residuals, `compute_residual` first forms the signed source-minus-prediction difference, then applies configured block gating, background downscaling, and optional chroma subsampling before representation encoding. In clipped mode, with offset $o=128$, each component is encoded as

\[
 u=\operatorname{uint8}(\operatorname{clip}(r+o,0,255)),\qquad \tilde r=u-o.
\]

This representation is not lossless over the full residual range: differences outside $[-128,127]$ saturate. Integer values inside that interval survive this mapping exactly before video coding. Full-range mode maps $[-255,255]$ to 256 levels, retaining zero and both extrema but necessarily quantizing. After this representation step, the native codec path writes an intermediate lossless file, invokes the requested encoder, retains the produced bitstream, decodes that same bitstream, converts the decoded uint8 values back to signed differences, and applies those differences to the predictor. The decoder therefore measures the codec roundtrip that is actually transmitted ([residual representation](../../src/pipeline/residual/lossy.py#L56), [residual computation](../../src/pipeline/residual/signal.py#L80), [native residual codec](../../src/pipeline/residual/codec.py#L73)).

The exact int16 residual is available as an uncompressed calibration path; it is not the lossy native-codec path. If the lossy residual encoder is unavailable or rejects an input, the runtime returns an uncompressed residual representation. That run may still be useful for debugging, but its `raw_parts` make `SizesBytes.is_rate` false. Likewise, a source-fallback artifact cannot earn a valid quality score. There is no rule that a configured semantic path always has an automatic, quality-preserving fallback ([codec stage](../../src/runner/stages.py#L937), [rate guard](../../src/runner/accounting.py#L116)).

## Shared tennis perception and data contracts

The shipped tennis dataset manifest is intentionally minimal: a short video smoke clip and several named player tracks, each identified as a video path or a sorted sequence of frames. The loader resolves declared roots, tags each sample with its domain and clip ID, and either reports missing files or skips them according to explicit policy; it does not download data. The general manifest lists DAVIS clips separately. These manifests support reproducible discovery and bounded loading, but do not certify annotation quality or establish that any checkpoint was trained on them ([tennis manifest](../../src/components/domain/datasets/tennis.yaml), [manifest loader](../../src/components/domain/datasets/catalog.py)).

Runtime observations share explicit coordinate and identity contracts. `ObjectRequest` carries an object identity, class, frame index, box, appearance crop, mask, and optional conditions into reconstruction; duplicate identity/frame pairs are removed before serialization. Crop transforms record source size, half-open crop bounds, target and resized sizes, padding, and scale. Their forward and inverse point transforms keep pose and mask coordinates aligned when model inputs use cropped canvases ([coordinate transform](../../src/components/perception/coordinates.py), [reconstruction request](../../src/pipeline/reconstruction/reconstruct.py#L43)).

The registered SAM3.1 path pins the local checkpoint hash and source revision when loading real artifacts, and rejects missing, dirty, or mismatched inputs rather than silently downloading them. It offers sequence sessions, role-separated prompts, tracked mask observations, and forward-only propagation under the causal runtime policy. Its registry adapter handles a single frame through a temporary session and currently recognizes people and rackets; other classes return no mask. The adapter and sequence APIs are implementation contracts, not a tennis segmentation quality result ([SAM3.1 implementation](../../src/components/segmentation/sam31.py#L116), [segmenter registry](../../src/components/segmentation/__init__.py)).

DWPose loads local detector and pose ONNX weights, crops around a person detection, optionally applies the corresponding mask, runs inference, and maps confident keypoint coordinates back into source-frame coordinates. Its output is WholeBody-133 with confidence and visibility state. `to_wire` projects that canonical pose to the consumer schema (for example OpenPose-18) and preserves absent-joint semantics. Consumer projection is part of the interface; it does not imply that an existing generative checkpoint accepts every new racket or pose condition ([DWPose estimator](../../src/components/pose/dwpose.py#L22), [wire schema](../../src/components/pose/wire.py#L80)).

Model adapters expose shared player-mask, racket-mask, racket-axis, and racket-width channels on a common crop canvas. A racket cross is eligible for cross-conditioned training; a hull fallback is explicitly ineligible. The adapter source says that new racket-aware SPADE or ControlNet checkpoints must be trained and validated against these maps. Existing checkpoint registration or architecture code is not that training evidence ([model adapter views](../../src/components/perception/model_adapters.py)).

## Generation, metrics, and evidence boundaries

The generation registry is lazy and capability-based: a declared backend specifies what appearance or motion inputs it consumes and what conditions it requires. Checkpoint resolution and compatibility are separate runtime concerns. In particular, registry presence must not be described as ready model weights. `experiments/tier/low_rate_sweep.py` binds a generator function that raises if called, because the sweep keeps generation off. `experiments/modular/neural_benchmark.py` evaluates numeric candidate fields read from a JSON specification; it does not load or execute the listed neural models. Its verdict is threshold logic on supplied numbers, not a benchmark run ([registry mechanism](../../src/contracts/registry.py), [sweep setup](../../experiments/tier/low_rate_sweep.py#L137), [benchmark evaluator](../../experiments/modular/neural_benchmark.py#L54)).

The repository's metric registry exposes reference and sequence metrics such as PSNR, SSIM, VMAF, LPIPS, pose OKS, mask IoU, and identity-related scores. A repository search of `src/`, `experiments/`, `config/`, and `manifests/` found no integrated MTTF, GLC, GVC-RT, or S2VC implementation. The retired generation note also marked MTTF unintegrated. `DISTS-pytorch` is pinned in `pyproject.toml`, but DISTS is not registered in the core metric registry. These are search results scoped to the checked repository paths, not claims about the external literature or packages ([metric registry](../../src/components/metrics/__init__.py), [generation status note](https://github.com/emanuele-artioli/PointStream/blob/086ae9035bc0ff7b552c05eb04904224f9abb32a/docs/areas/generation.md), [dependencies](../../pyproject.toml)).

## Focused verification map

The existing focused tests correspond to the most consequential contracts:

- `tests/runner/test_run.py` checks delivered-frame access, per-chunk concatenation, code-stage ordering, residual-only scoring, and ledger fit.
- `tests/runner/test_residual_transport.py` checks signed arithmetic, saturation, native codec roundtrip, fresh-process source-free decoding, corrupted payload rejection, and all-byte reconciliation.
- `tests/components/test_rigid.py` checks that rackets are hull/cross shapes and balls use blob strategies rather than skeletons; `tests/components/test_model_adapters.py` checks projected conditions and fallback eligibility.
- `tests/components/test_sam31_segmenter.py`, `tests/components/test_dwpose_estimator.py`, and `tests/components/test_domain_datasets.py` cover the stated sequence, coordinate, visibility, and manifest contracts.

These tests verify implementation behavior. They do not replace the paper's matched-rate tennis evaluation, cross-match generalization, perceptual/semantic evaluation, or audit of the actual model and codec artifacts. Those belong in [06-evaluation.md](06-evaluation.md), [07-experiment-plan.md](07-experiment-plan.md), and [08-evidence-ledger.md](08-evidence-ledger.md).
