from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch

from src.encoder.background_modeler import BackgroundModeler
from src.encoder.ball_extractor import BallExtractor
from src.encoder.dag import DAGNode, DAGOrchestrator
from src.encoder.execution_pool import BaseExecutionPool
from src.encoder.actor_pipeline import (
    ActorExtractionResult,
    ActorExtractor,
)
from src.encoder.reference_extractor import ReferenceExtractor
from src.encoder.residual_calculator import ResidualCalculator
from src.encoder.video_io import decode_video_to_tensor, probe_video_metadata
from src.shared.schemas import EncodedChunkPayload, FrameState, PanoramaPacket, ResidualPacket, VideoChunk
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import cpu_bound, gpu_bound
from src.transport.panorama_encoder import (
    build_panorama_encoder,
    round_trip_panorama,
    round_trip_panorama_delta,
)


class _StreamingActorBundle:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame_states: list[FrameState] = []
        self._result: ActorExtractionResult | None = None
        self._error: Exception | None = None
        self._done = False

    @property
    def frame_states(self) -> list[FrameState]:
        return self.finalize().frame_states

    @property
    def actor_packets(self) -> list[Any]:
        return self.finalize().actor_packets

    def append_state(self, state: FrameState) -> None:
        with self._cond:
            self._frame_states.append(state)
            self._cond.notify_all()

    def complete(self, result: ActorExtractionResult) -> None:
        with self._cond:
            self._result = result
            self._done = True
            self._cond.notify_all()

    def fail(self, exc: Exception) -> None:
        with self._cond:
            self._error = exc
            self._done = True
            self._cond.notify_all()

    def wait_for_frame_state(self, frame_idx: int) -> FrameState | None:
        with self._cond:
            while len(self._frame_states) <= frame_idx and not self._done and self._error is None:
                self._cond.wait()
            if self._error is not None:
                raise self._error
            if len(self._frame_states) > frame_idx:
                return self._frame_states[frame_idx]
            return None

    def finalize(self) -> ActorExtractionResult:
        with self._cond:
            while not self._done and self._error is None:
                self._cond.wait()
            if self._error is not None:
                raise self._error
            if self._result is None:
                return ActorExtractionResult(frame_states=list(self._frame_states), actor_packets=[])
            return self._result


class EncoderPipeline:
    def __init__(
        self,
        config: Any, # We use Any to avoid circular import or we can import PointstreamConfig
        execution_pool: BaseExecutionPool | None = None,
        actor_extractor: Any | None = None,
        ball_extractor: Any | None = None,
        reference_extractor: Any | None = None,
        object_tracker: Any | None = None,
        residual_calculator: ResidualCalculator | None = None,
    ) -> None:
        self.config = config
        self._dag = DAGOrchestrator(execution_pool=execution_pool)
        self._background_modeler = BackgroundModeler(config=self.config)
        self._panorama_encoder = build_panorama_encoder(config=self.config)
        self._actor_extractor = actor_extractor or ActorExtractor(config=self.config)
        self._object_tracker = object_tracker
        self._ball_extractor = ball_extractor or BallExtractor()
        self._reference_extractor = reference_extractor or ReferenceExtractor()
        self._residual_calculator = residual_calculator or ResidualCalculator(config=self.config, synthesis_engine=SynthesisEngine(config=self.config))
        # Background-layer ladder rung 2 ("panorama+delta", report 10 Phase
        # 5.3): last reconstructed (decoded) panorama + its codec_id per
        # VideoChunk.scene_id, so a later sub-chunk of the same scene can be
        # sent as a delta against it instead of a fresh full panorama. Keyed
        # by scene_id, not chunk_id, and grows for the lifetime of this
        # EncoderPipeline instance (one per match in match_orchestrator).
        self._scene_panorama_state: dict[str, tuple[np.ndarray, str]] = {}
        self._register_nodes()
        self._last_dag_profile: dict[str, float] = {}
        self._last_actor_profile: dict[str, float] = {}
        self._last_residual_profile: dict[str, float] = {}
        # Report 10 Phase 5.1(e): per-scene panorama cache. `None` (the
        # default) disables caching entirely -- callers that never opt in via
        # `set_scene_context()` keep today's per-chunk recompute behavior.
        self._panorama_scene_key: Any | None = None
        self._panorama_scene_cache: PanoramaPacket | None = None

    def set_scene_context(self, scene_key: Any | None) -> None:
        """Mark the start of a new scene for panorama caching.

        Background is near-static within one tennis point/scene (this
        project's constrained domain, see CLAUDE.md), so recomputing the
        panorama via `BackgroundModeler` for every sub-chunk of the same
        scene is wasted work (measured 0.46 fps stage in report 10 Phase 5).
        `src.encoder.match_orchestrator` calls this once per scene, before
        that scene's first sub-chunk `encode_chunk()`, with a key that is
        stable within the scene and changes across scenes (e.g. the scene
        index) -- changing the key invalidates the cache and the next
        `encode_chunk()` recomputes via `BackgroundModeler` as before.
        Passing `None` disables caching (the safe default for any caller
        that does not opt in, e.g. the single-chunk `run_pipeline` path).

        Residual Guarantee note: the cached object is exactly the
        post-codec-round-trip `PanoramaPacket` this pipeline transmits (see
        `build_panorama_node` below) -- reusing it verbatim for later
        sub-chunks in the same scene never diverges from what the client
        actually receives, because the cache *is* the transmitted bytes,
        not a separate copy that could go stale. Reused sub-chunks may have
        a different `num_frames`/`start_frame_id` than the sub-chunk that
        produced the cache; `SynthesisEngine._reconstruct_background_frames`
        already pads `homography_matrices` with identity transforms (or
        truncates) to match the current chunk's own frame count, so this is
        the same defensive path already exercised whenever a real payload's
        homography count and chunk frame count merely happen to differ.
        """
        if scene_key != self._panorama_scene_key:
            self._panorama_scene_key = scene_key
            self._panorama_scene_cache = None

    def get_detailed_profile(self) -> dict[str, float]:
        profile = dict(self._last_dag_profile)
        for k, v in self._last_actor_profile.items():
            profile[k] = v
        for k, v in self._last_residual_profile.items():
            profile[f"residual_{k}"] = v
        return profile

    @staticmethod
    def _make_chunk_node(process_method: Any):
        def node_func(context, deps):
            return process_method(deps["chunk"])

        setattr(
            node_func,
            "_execution_tag",
            getattr(process_method, "_execution_tag", "cpu"),
        )
        return node_func

    def _register_nodes(self) -> None:
        @cpu_bound
        def load_chunk(context, deps):
            return context["chunk"]

        self._dag.add_node(DAGNode(name="chunk", func=load_chunk))

        actor_bundle_func = self._make_chunk_node(self._process_actor_bundle)
        self._dag.add_node(
            DAGNode(
                name="actor_bundle",
                func=actor_bundle_func,
                dependencies=("chunk",),
            )
        )

        def select_actor_packets(context, deps):
            return deps["actor_bundle"].actor_packets

        setattr(select_actor_packets, "_execution_tag", getattr(actor_bundle_func, "_execution_tag", "gpu"))

        self._dag.add_node(
            DAGNode(
                name="actors",
                func=select_actor_packets,
                dependencies=("actor_bundle", "ball"),
            )
        )

        def select_frame_states(context, deps):
            return deps["actor_bundle"].frame_states

        setattr(select_frame_states, "_execution_tag", getattr(actor_bundle_func, "_execution_tag", "gpu"))

        self._dag.add_node(
            DAGNode(
                name="frame_states",
                func=select_frame_states,
                dependencies=("actor_bundle", "ball"),
            )
        )

        def build_panorama_node(context, deps):
            chunk = deps["chunk"]

            background_layer = str(
                getattr(self.config, "background_layer", "panorama-static") or "panorama-static"
            ).strip().lower()

            # Report 10 Phase 5.1(e) vs 5.3 reconciliation (2026-07-12): the
            # full-packet scene cache below only applies to rungs where every
            # sub-chunk of a scene transmits the identical packet
            # (panorama-static, roi-video). panorama-delta (rung 2) must
            # always fall through to BackgroundModeler.process(), because it
            # needs each sub-chunk's true current panorama to diff against
            # the scene's previous one -- reusing a verbatim cached packet
            # here would silently make the delta rung a permanent no-op
            # (every sub-chunk resending the scene's first packet instead of
            # a delta). See `set_scene_context()`'s docstring for why this
            # cache is safe to reuse verbatim for the rungs it does apply to.
            if (
                background_layer != "panorama-delta"
                and self._panorama_scene_key is not None
                and self._panorama_scene_cache is not None
            ):
                return self._panorama_scene_cache.model_copy(update={"chunk_id": chunk.chunk_id})

            panorama_packet = self._background_modeler.process(
                chunk=chunk,
                decoded_video_tensor=context.get("decoded_video_tensor"),
                frame_states=None,
            )
            # Residual Guarantee: synthesize against the codec-decoded panorama the
            # client will actually reconstruct from, not the raw pre-codec pixels.
            panorama_np = np.asarray(panorama_packet.panorama_image, dtype=np.uint8)

            scene_id = chunk.scene_id
            previous_state = self._scene_panorama_state.get(scene_id) if scene_id else None

            if background_layer == "panorama-delta" and scene_id is not None and previous_state is not None:
                # Not the scene's first sub-chunk: send a delta against the last
                # panorama sent for this scene instead of a fresh full one.
                previous_np, _previous_codec_id = previous_state
                encoded_bytes, decoded_np = round_trip_panorama_delta(
                    current_bgr=panorama_np,
                    previous_bgr=previous_np,
                    encoder=self._panorama_encoder,
                )
                codec_id = f"{self._panorama_encoder.codec_id}+delta"
                panorama_mode = "delta"
            else:
                # Rung 1 (panorama-static), rung 3 (roi-video), or this scene's
                # first sub-chunk under panorama-delta: send the full panorama,
                # same as today.
                encoded_bytes, decoded_np = round_trip_panorama(panorama_np, self._panorama_encoder)
                codec_id = self._panorama_encoder.codec_id
                panorama_mode = "full"

            if scene_id is not None:
                self._scene_panorama_state[scene_id] = (decoded_np, codec_id)

            result = panorama_packet.model_copy(
                update={
                    "panorama_image": decoded_np.tolist(),
                    "panorama_codec_bytes": encoded_bytes,
                    "panorama_codec_id": codec_id,
                    "panorama_mode": panorama_mode,
                }
            )
            if self._panorama_scene_key is not None and True:
                self._panorama_scene_cache = result
            return result

        setattr(
            build_panorama_node,
            "_execution_tag",
            getattr(self._background_modeler.process, "_execution_tag", "cpu"),
        )
        self._dag.add_node(
            DAGNode(
                name="panorama",
                func=build_panorama_node,
                dependencies=("chunk",),
            )
        )
        def build_actor_references_node(context, deps):
            return self._reference_extractor.process(
                chunk=deps["chunk"],
                frame_states=deps["frame_states"],
            )

        setattr(
            build_actor_references_node,
            "_execution_tag",
            getattr(self._reference_extractor.process, "_execution_tag", "cpu"),
        )
        self._dag.add_node(
            DAGNode(
                name="actor_references",
                func=build_actor_references_node,
                dependencies=("chunk", "frame_states"),
            )
        )
        def build_rigid_objects_node(context, deps):
            if hasattr(self, "_object_tracker") and self._object_tracker is not None:
                return self._object_tracker.process(deps["chunk"])
            return []
        tracker = getattr(self, '_object_tracker', None)
        tag = getattr(tracker.process, '_execution_tag', 'cpu') if tracker is not None else 'cpu'
        setattr(build_rigid_objects_node, '_execution_tag', tag)

        self._dag.add_node(
            DAGNode(
                name="rigid_objects",
                func=build_rigid_objects_node,
                dependencies=("chunk",),
            )
        )
        def build_ball_node(context, deps):
            actor_bundle = deps["actor_bundle"]
            if hasattr(self._ball_extractor, "process_shifted") and hasattr(actor_bundle, "wait_for_frame_state"):
                return self._ball_extractor.process_shifted(
                    chunk=deps["chunk"],
                    panorama=deps["panorama"],
                    actor_bundle=actor_bundle,
                )
            return self._ball_extractor.process(
                chunk=deps["chunk"],
                panorama=deps["panorama"],
                frame_states=actor_bundle.frame_states,
            )

        setattr(
            build_ball_node,
            "_execution_tag",
            getattr(self._ball_extractor.process, "_execution_tag", "gpu"),
        )
        self._dag.add_node(
            DAGNode(
                name="ball",
                func=build_ball_node,
                dependencies=("chunk", "panorama", "actor_bundle"),
            )
        )
        def build_residual_node(context, deps):
            placeholder_residual = ResidualPacket(
                chunk_id=deps["chunk"].chunk_id,
                codec="hevc-placeholder",
                residual_video_uri=f"memory://residual/{deps['chunk'].chunk_id}.mp4",
            )
            payload = EncodedChunkPayload(
                chunk=deps["chunk"],
                panorama=deps["panorama"],
                actors=deps["actors"],
                actor_references=deps["actor_references"],
                rigid_objects=deps["rigid_objects"],
                ball=deps["ball"],
                residual=placeholder_residual,
            )
            return self._residual_calculator.process(
                chunk=deps["chunk"],
                payload=payload,
                frame_states=deps["frame_states"],
            )

        setattr(
            build_residual_node,
            "_execution_tag",
            getattr(self._residual_calculator.process, "_execution_tag", "cpu"),
        )
        self._dag.add_node(
            DAGNode(
                name="residual",
                func=build_residual_node,
                dependencies=("chunk", "panorama", "actors", "actor_references", "rigid_objects", "ball", "frame_states"),
            )
        )

    @gpu_bound
    def _process_actor_bundle(self, chunk: VideoChunk) -> Any:
        shifted_enabled = getattr(self.config, "enable_shifted_ball", False)
        supports_streaming = hasattr(self._actor_extractor, "process_with_states_streaming")
        supports_shifted_ball = hasattr(self._ball_extractor, "process_shifted")

        if shifted_enabled and supports_streaming and supports_shifted_ball:
            bundle = _StreamingActorBundle()

            def _run_stream() -> None:
                try:
                    result = self._actor_extractor.process_with_states_streaming(
                        chunk,
                        on_frame_state=bundle.append_state,
                    )
                    bundle.complete(result)
                except Exception as exc:  # pragma: no cover - guarded runtime path
                    bundle.fail(exc)

            thread = threading.Thread(target=_run_stream, name="pointstream-actor-stream", daemon=True)
            thread.start()
            return bundle

        if hasattr(self._actor_extractor, "process_with_states"):
            return self._actor_extractor.process_with_states(chunk)
        actor_packets = self._actor_extractor.process(chunk)
        return ActorExtractionResult(frame_states=[], actor_packets=actor_packets)

    def encode_chunk(self, chunk: VideoChunk) -> EncodedChunkPayload:
        context = self._dag.run(initial_context={"chunk": chunk})
        # capture DAG profiling produced by the orchestrator
        self._last_dag_profile = context.get("dag_profile", {})
        actor_bundle = context.get("actor_bundle")
        if actor_bundle is not None and hasattr(actor_bundle, "finalize"):
            self._last_actor_profile = actor_bundle.finalize().profile
        elif actor_bundle is not None and hasattr(actor_bundle, "profile"):
            self._last_actor_profile = actor_bundle.profile
        else:
            self._last_actor_profile = {}
            
        if hasattr(self._residual_calculator, "get_detailed_profile"):
            self._last_residual_profile = self._residual_calculator.get_detailed_profile()
        else:
            self._last_residual_profile = {}
            
        return EncodedChunkPayload(
            chunk=context["chunk"],
            panorama=context["panorama"],
            actors=context["actors"],
            actor_references=context["actor_references"],
            rigid_objects=context["rigid_objects"],
            ball=context["ball"],
            residual=context["residual"],
        )

    def encode_video_file(
        self,
        video_path: str | Path,
        chunk_id: str,
        start_frame_id: int = 0,
        max_frames: int | None = None,
    ) -> tuple[EncodedChunkPayload, torch.Tensor]:
        metadata = probe_video_metadata(video_path)
        effective_num_frames = metadata.num_frames if max_frames is None else min(metadata.num_frames, max_frames)

        chunk = VideoChunk(
            chunk_id=chunk_id,
            source_uri=str(video_path),
            start_frame_id=start_frame_id,
            fps=metadata.fps,
            num_frames=effective_num_frames,
            width=metadata.width,
            height=metadata.height,
        )

        context = self._dag.run(initial_context={"chunk": chunk})
        # capture DAG profiling produced by the orchestrator
        self._last_dag_profile = context.get("dag_profile", {})
        actor_bundle = context.get("actor_bundle")
        if actor_bundle is not None and hasattr(actor_bundle, "finalize"):
            self._last_actor_profile = actor_bundle.finalize().profile
        elif actor_bundle is not None and hasattr(actor_bundle, "profile"):
            self._last_actor_profile = actor_bundle.profile
        else:
            self._last_actor_profile = {}
            
        if hasattr(self._residual_calculator, "get_detailed_profile"):
            self._last_residual_profile = self._residual_calculator.get_detailed_profile()
        else:
            self._last_residual_profile = {}

        payload = EncodedChunkPayload(
            chunk=context["chunk"],
            panorama=context["panorama"],
            actors=context["actors"],
            actor_references=context["actor_references"],
            rigid_objects=context["rigid_objects"],
            ball=context["ball"],
            residual=context["residual"],
        )
        decoded_video = decode_video_to_tensor(video_path)
        decoded_tensor = decoded_video.tensor
        if max_frames is not None:
            decoded_tensor = decoded_tensor[:max_frames]
        return payload, decoded_tensor

    def encode_video_file_with_states(
        self,
        video_path: str | Path,
        chunk_id: str,
        start_frame_id: int = 0,
        max_frames: int | None = None,
    ) -> tuple[EncodedChunkPayload, torch.Tensor, list[FrameState]]:
        metadata = probe_video_metadata(video_path)
        effective_num_frames = metadata.num_frames if max_frames is None else min(metadata.num_frames, max_frames)

        chunk = VideoChunk(
            chunk_id=chunk_id,
            source_uri=str(video_path),
            start_frame_id=start_frame_id,
            fps=metadata.fps,
            num_frames=effective_num_frames,
            width=metadata.width,
            height=metadata.height,
        )

        context = self._dag.run(initial_context={"chunk": chunk})
        # capture DAG profiling produced by the orchestrator
        self._last_dag_profile = context.get("dag_profile", {})
        actor_bundle = context.get("actor_bundle")
        if actor_bundle is not None and hasattr(actor_bundle, "finalize"):
            self._last_actor_profile = actor_bundle.finalize().profile
        elif actor_bundle is not None and hasattr(actor_bundle, "profile"):
            self._last_actor_profile = actor_bundle.profile
        else:
            self._last_actor_profile = {}
            
        if hasattr(self._residual_calculator, "get_detailed_profile"):
            self._last_residual_profile = self._residual_calculator.get_detailed_profile()
        else:
            self._last_residual_profile = {}

        payload = EncodedChunkPayload(
            chunk=context["chunk"],
            panorama=context["panorama"],
            actors=context["actors"],
            actor_references=context["actor_references"],
            rigid_objects=context["rigid_objects"],
            ball=context["ball"],
            residual=context["residual"],
        )
        decoded_video = decode_video_to_tensor(video_path)
        decoded_tensor = decoded_video.tensor
        if max_frames is not None:
            decoded_tensor = decoded_tensor[:max_frames]
        return payload, decoded_tensor, context["frame_states"]

    def shutdown(self) -> None:
        self._dag.shutdown()
