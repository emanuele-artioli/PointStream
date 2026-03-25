from __future__ import annotations

from typing import Any

from src.encoder.dag import DAGNode, DAGOrchestrator
from src.encoder.execution_pool import BaseExecutionPool
from src.encoder.mock_extractors import (
    ActorExtractor,
    BackgroundModeler,
    BallTracker,
    ObjectTracker,
)
from src.encoder.residual import ResidualCalculator
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
        self._dag.add_node(
            DAGNode(
                name="panorama",
                func=self._make_chunk_node(self._background_modeler.process),
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

    def shutdown(self) -> None:
        self._dag.shutdown()
