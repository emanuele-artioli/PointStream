from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.encoder.background_modeler import BackgroundModeler
from src.encoder.dag import DAGNode, DAGOrchestrator
from src.encoder.execution_pool import BaseExecutionPool
from src.encoder.mock_extractors import (
    ActorExtractor,
    BallTracker,
    ObjectTracker,
)
from src.encoder.residual import ResidualCalculator
from src.encoder.video_io import decode_video_to_tensor, probe_video_metadata
from src.shared.schemas import EncodedChunkPayload, VideoChunk
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import cpu_bound


class EncoderPipeline:
    def __init__(self, execution_pool: BaseExecutionPool | None = None) -> None:
        self._dag = DAGOrchestrator(execution_pool=execution_pool)
        self._background_modeler = BackgroundModeler()
        self._actor_extractor = ActorExtractor()
        self._object_tracker = ObjectTracker()
        self._ball_tracker = BallTracker()
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
        self._dag.add_node(
            DAGNode(
                name="actors",
                func=self._make_chunk_node(self._actor_extractor.process),
                dependencies=("chunk",),
            )
        )
        self._dag.add_node(
            DAGNode(
                name="rigid_objects",
                func=self._make_chunk_node(self._object_tracker.process),
                dependencies=("chunk",),
            )
        )
        self._dag.add_node(
            DAGNode(
                name="ball",
                func=self._make_chunk_node(self._ball_tracker.process),
                dependencies=("chunk",),
            )
        )
        self._dag.add_node(
            DAGNode(
                name="residual",
                func=self._make_chunk_node(self._residual_calculator.process),
                dependencies=("chunk", "panorama", "actors", "rigid_objects", "ball"),
            )
        )

    def encode_chunk(self, chunk: VideoChunk) -> EncodedChunkPayload:
        context = self._dag.run(initial_context={"chunk": chunk})
        return EncodedChunkPayload(
            chunk=context["chunk"],
            panorama=context["panorama"],
            actors=context["actors"],
            rigid_objects=context["rigid_objects"],
            ball=context["ball"],
            residual=context["residual"],
        )

    def encode_video_file(
        self,
        video_path: str | Path,
        chunk_id: str,
        start_frame_id: int = 0,
    ) -> tuple[EncodedChunkPayload, torch.Tensor]:
        metadata = probe_video_metadata(video_path)

        chunk = VideoChunk(
            chunk_id=chunk_id,
            source_uri=str(video_path),
            start_frame_id=start_frame_id,
            fps=metadata.fps,
            num_frames=metadata.num_frames,
            width=metadata.width,
            height=metadata.height,
        )

        context = self._dag.run(initial_context={"chunk": chunk})
        payload = EncodedChunkPayload(
            chunk=context["chunk"],
            panorama=context["panorama"],
            actors=context["actors"],
            rigid_objects=context["rigid_objects"],
            ball=context["ball"],
            residual=context["residual"],
        )
        decoded_video = decode_video_to_tensor(video_path)
        return payload, decoded_video.tensor

    def shutdown(self) -> None:
        self._dag.shutdown()
