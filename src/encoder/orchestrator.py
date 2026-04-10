from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.encoder.background_modeler import BackgroundModeler
from src.encoder.ball_extractor import BallExtractor
from src.encoder.dag import DAGNode, DAGOrchestrator
from src.encoder.execution_pool import BaseExecutionPool
from src.encoder.mock_extractors import (
    ActorExtractionResult,
    ActorExtractor,
    ObjectTracker,
)
from src.encoder.reference_extractor import ReferenceExtractor
from src.encoder.residual_calculator import ResidualCalculator
from src.encoder.video_io import decode_video_to_tensor, probe_video_metadata
from src.shared.schemas import EncodedChunkPayload, FrameState, ResidualPacket, VideoChunk
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import cpu_bound, gpu_bound


class EncoderPipeline:
    def __init__(
        self,
        execution_pool: BaseExecutionPool | None = None,
        actor_extractor: Any | None = None,
        ball_extractor: Any | None = None,
        reference_extractor: Any | None = None,
    ) -> None:
        self._dag = DAGOrchestrator(execution_pool=execution_pool)
        self._background_modeler = BackgroundModeler()
        self._actor_extractor = actor_extractor or ActorExtractor()
        self._object_tracker = ObjectTracker()
        self._ball_extractor = ball_extractor or BallExtractor()
        self._reference_extractor = reference_extractor or ReferenceExtractor()
        self._residual_calculator = ResidualCalculator(SynthesisEngine())
        self._register_nodes()

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

        def build_panorama_node(context, deps):
            return self._background_modeler.process(
                chunk=deps["chunk"],
                decoded_video_tensor=context.get("decoded_video_tensor"),
            )

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
                dependencies=("actor_bundle",),
            )
        )

        def select_frame_states(context, deps):
            return deps["actor_bundle"].frame_states

        setattr(select_frame_states, "_execution_tag", getattr(actor_bundle_func, "_execution_tag", "gpu"))

        self._dag.add_node(
            DAGNode(
                name="frame_states",
                func=select_frame_states,
                dependencies=("actor_bundle",),
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
        self._dag.add_node(
            DAGNode(
                name="rigid_objects",
                func=self._make_chunk_node(self._object_tracker.process),
                dependencies=("chunk",),
            )
        )
        def build_ball_node(context, deps):
            return self._ball_extractor.process(
                chunk=deps["chunk"],
                panorama=deps["panorama"],
                frame_states=deps["frame_states"],
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
                dependencies=("chunk", "panorama", "frame_states"),
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
    def _process_actor_bundle(self, chunk: VideoChunk) -> ActorExtractionResult:
        if hasattr(self._actor_extractor, "process_with_states"):
            return self._actor_extractor.process_with_states(chunk)
        actor_packets = self._actor_extractor.process(chunk)
        return ActorExtractionResult(frame_states=[], actor_packets=actor_packets)

    def encode_chunk(self, chunk: VideoChunk) -> EncodedChunkPayload:
        context = self._dag.run(initial_context={"chunk": chunk})
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
