# Method

PointStream is studied here as an object-centric codec for broadcast tennis. The camera often leaves the court background nearly static within a shot while players move substantially and the racket and ball occupy few pixels. The method therefore represents the scene as a reusable background plus tracked foreground objects and a corrective residual. The intended submission claims and paper-level motivation are outlined in [01-introduction.md](01-introduction.md) and [02-related-work.md](02-related-work.md); this chapter describes the implemented method and its limits. “Implemented” below refers to code paths and contracts; “measured” requires a cited run artifact, and “planned” describes work that remains to be evaluated. The code description alone establishes no final tennis quality or generalization result.

## Scene factorization

Let an input clip be $X=(x_0,\ldots,x_{T-1})$, with RGB frames $x_t\in[0,255]^{H\times W\times3}$. For a shot with a usable background model, the encoder transmits a plate $P$ and frame-to-plate maps $H_t$. The client obtains a background prediction

\[
 b_t = \mathcal{W}(P,H_t^{-1}),
\]

where \(\mathcal{W}\) is perspective warping. The code treats the recorded homographies as frame-to-plate maps and inverts them for reconstruction. A static, same-size plate with identity maps is copied without interpolation. If the background stage is disabled or deferred, its prediction is zero and the residual must account for the omitted pixels. A delta plate is relative to the previous decoded plate for the same scene; a scene must begin with a full plate. These are explicit runtime rules, not a claim that every tennis shot satisfies a single-plane camera model ([background resolver](../../src/pipeline/reconstruction/background.py), [background stage](../../src/runner/stages.py#L679)).

For each selected object $i$ on frame $t$, the encoder sends an appearance reference $a_i$, a bounding box $q_{i,t}$, a stable identity, and, when available, a segmentation mask $m_{i,t}$ or structured motion condition. The client decodes the reference, resizes it to the box, and blends it into the predicted background. With segmentation, the blend is restricted to the mask; without it, the fallback is the full box, including surrounding background. Denote this assembled prediction by $p_t=\mathcal{C}(b_t,\{a_i,q_{i,t},m_{i,t}\}_i)$. The compositor performs placement and blending; pixel generation is a separate optional operation ([compositor](../../src/pipeline/reconstruction/compositor.py)).

The corrective signal is the signed difference $r_t=x_t-p_t$, computed in signed 16-bit arithmetic. The delivered reconstruction is

\[
 \hat{x}_t = \operatorname{clip}(p_t + \tilde r_t,0,255),
\]

where $\tilde r_t$ is the decoded, possibly lossy and video-coded residual. This residual path corrects prediction error; it does not make semantic prediction lossless once quantized or coded. The no-residual point is the unaided prediction. The all-off runner corner copies raw source RGB frames and uses their array byte count when no payload exists. It is an identity/raw-rate control, not an encoded AV1/VVC anchor and not evidence of semantic reconstruction quality.

## Tennis object representations

Human pose is stored internally in a canonical COCO WholeBody-133 schema with per-joint confidence, presence, and visibility. A model consumer receives a projection into its declared schema; absent joints remain absent, and derived joints such as an OpenPose neck are formed only when their parent joints are present. Thus a smaller consumer schema can be sent without treating zero-filled missing joints as detections ([keypoint schema](../../src/contracts/keypoints.py), [pose wire projection](../../src/components/pose/wire.py)).

Rackets are rigid objects, not articulated skeletons. The implementation extracts a convex hull from a mask, or uses the box corners when a mask is unavailable. An optional racket-cross representation contains the associated player wrist, the farthest hull vertex as tip, and the hull intersections with a transverse line at a configurable fraction of the wrist-to-tip axis. The ordered four points describe racket axis and width. If wrist association or geometry fails, the result is explicitly typed as a hull fallback with a reason; it is not silently relabelled as a valid cross. A wrist is borrowed from a same-frame player pose and must be confident and close to the hull ([racket geometry](../../src/components/rigid/racket.py)).

The ball has no skeleton either. Implemented representations are a centroid and equivalent-area radius from an observed mask or box, and a difference-blob estimate formed by thresholding absolute grayscale difference from the plate after excluding player and racket regions. The difference method returns the largest connected component above its area threshold; it returns no estimate when the background is absent or no component qualifies. This rule is an inexpensive heuristic, not a validated ball tracker, and it can confuse other moving court pixels with the ball ([ball geometry](../../src/components/rigid/ball.py)).

## Rate and distortion protocol

At an operating point, the accounting ledger names panorama bytes $B$, actor-reference bytes $F$, residual bytes $R$, and aggregate metadata/envelope bytes $M_{code}$. The code-level transport total is $T=B+F+R+M_{code}$, where $M_{code}$ denotes the aggregate metadata/envelope field. In the evaluation chapter this aggregate is split into motion/mask/geometry $M$ and remaining envelope $H_{env}$, so $M_{code}=M+H_{env}$. The all-off corner instead uses raw source RGB bytes; conventional encoded anchors are separate experiments. The metadata subledger records masks, pose/motion, placement headers, generator identity, and envelope overhead. Each named component must fit within the measured transport total. If any component is still a raw array size rather than a coded bitstream, the run is marked `is_rate=false` and the source ratio is withheld ([byte accounting](../../src/runner/accounting.py#L84), [serialized-envelope reconciliation](../../src/runner/client.py#L395)).

Distortion is evaluated against the pixels reconstructed from the client payload. In particular, rate-quality pairs use the result after residual encode/decode; the runner exposes this as `delivered_frames` and scores it as `delivered_quality`. `RunResult.frames` is a distinct pre-codec residual-stage reconstruction and must not be paired with the coded total. The experiment chapters should compare operating points only when both the delivered pixels and complete wire cost refer to the same run ([run result contract](../../src/runner/run.py#L134), [client reconstruction and score](../../src/runner/run.py#L701)).

This is a measured rate-distortion protocol, not an implemented optimizer of $D+\lambda R$. The stage lattice and experiment scripts select configurations; they do not solve a learned joint rate-distortion objective. The paper should report quality-versus-measured-rate results, the selected metric and region, and the codec/environment for each point. See [06-evaluation.md](06-evaluation.md), [07-experiment-plan.md](07-experiment-plan.md), and [08-evidence-ledger.md](08-evidence-ledger.md).

## Design alternatives and failure boundaries

The implemented static/reference-paste route provides a deterministic pixel carrier with measured appearance bytes and avoids generative hallucination. Neural generation is an optional backend axis, not a necessary part of the codec. Registry entries declare names, capabilities, conditioning requirements, and lazy construction targets; they do not prove that checkpoint files are present, compatible, trained on the claimed input contract, or ready for inference. In particular, the low-rate tennis sweep injects no generator and disables generation. The separate neural-benchmark script reads candidate metrics from JSON and applies thresholds; its code does not invoke candidate models. It cannot serve as model-inference evidence ([backend registry](../../src/contracts/registry.py), [low-rate sweep](../../experiments/tier/low_rate_sweep.py#L137), [threshold evaluator](../../experiments/modular/neural_benchmark.py#L54)).

The main representation trade-offs are observable in the implementation:

- A per-shot full plate avoids dependence on an earlier scene state. A delta plate can reduce repeated background payload, but requires ordered decoding and a prior plate for that scene. Streamed background coding is a separate temporal option, and its first keyframe remains part of the run cost.
- A segmentation mask can constrain pasted pixels more closely than a box. Missing masks deliberately degrade to box compositing and can introduce seams or overwrite nearby court pixels. Missing or poor detections also propagate into object placement and motion conditions.
- A racket hull preserves observed shape, while a racket cross is a compact, model-oriented summary. The cross can be unavailable for a hidden wrist, a distant association, or degenerate hull geometry; the typed fallback supports exclusion during cross-conditioned training.
- A lossy residual reduces payload only insofar as its actual bitstream does. Clipped uint8 residual coding saturates differences outside $[-128,127]$ at the default offset; full-range mapping preserves a wider range with quantization. Chroma subsampling and the selected video codec add further loss. If native encoding fails, the fallback is raw and cannot be reported as a compression rate.

The egocentric manufacturing demo is secondary to the tennis method. Its freely moving camera violates the static-plate assumption more often, so it needs separate camera-motion treatment and separate evidence. It is not evidence that the tennis method generalizes; planned scope and motivation belong in [03-motivating-example.md](03-motivating-example.md) and the evaluation chapter. This repository also contains a general DAVIS manifest for handheld clips, but that manifest is an index, not proof of a completed generalization study ([general dataset manifest](../../src/components/domain/datasets/general.yaml)).
