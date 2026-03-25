from __future__ import annotations

from src.encoder.dag import DAGNode, DAGOrchestrator
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
    def __init__(self) -> None:
        self._dag = DAGOrchestrator()
        self._background_modeler = BackgroundModeler()
        self._actor_extractor = ActorExtractor()
        self._object_tracker = ObjectTracker()
        self._ball_tracker = BallTracker()
        self._residual_calculator = ResidualCalculator(SynthesisEngine())
        self._register_nodes()

    def _register_nodes(self) -> None:
        @cpu_bound
        def load_chunk(context, deps):
            return context["chunk"]

        self._dag.add_node(DAGNode(name="chunk", func=load_chunk))
        self._dag.add_node(
            DAGNode(
                name="panorama",
                func=lambda context, deps: self._background_modeler.process(deps["chunk"]),
                dependencies=("chunk",),
            )
        )
        self._dag.add_node(
            DAGNode(
                name="actors",
                func=lambda context, deps: self._actor_extractor.process(deps["chunk"]),
                dependencies=("chunk",),
            )
        )
        self._dag.add_node(
            DAGNode(
                name="rigid_objects",
                func=lambda context, deps: self._object_tracker.process(deps["chunk"]),
                dependencies=("chunk",),
            )
        )
        self._dag.add_node(
            DAGNode(
                name="ball",
                func=lambda context, deps: self._ball_tracker.process(deps["chunk"]),
                dependencies=("chunk",),
            )
        )
        self._dag.add_node(
            DAGNode(
                name="residual",
                func=lambda context, deps: self._residual_calculator.process(deps["chunk"]),
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
