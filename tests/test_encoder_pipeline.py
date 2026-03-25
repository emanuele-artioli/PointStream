from __future__ import annotations

import unittest

from src.encoder.orchestrator import EncoderPipeline
from src.shared.schemas import VideoChunk


class TestEncoderPipeline(unittest.TestCase):
    def test_encode_chunk_contract_and_events(self) -> None:
        chunk = VideoChunk(
            chunk_id="enc001",
            source_uri="assets/test_chunks/enc001.mp4",
            start_frame_id=10,
            fps=25.0,
            num_frames=12,
            width=960,
            height=540,
        )

        pipeline = EncoderPipeline()
        payload = pipeline.encode_chunk(chunk)

        self.assertEqual(payload.chunk.chunk_id, "enc001")
        self.assertEqual(payload.panorama.frame_width, 960)
        self.assertEqual(payload.panorama.frame_height, 540)
        self.assertEqual(len(payload.panorama.camera_poses), 12)
        self.assertEqual(len(payload.actors), 2)
        self.assertEqual(len(payload.rigid_objects), 1)
        self.assertEqual(payload.ball.object_id, "ball_0")
        self.assertEqual(payload.residual.codec, "hevc-placeholder")

        for actor in payload.actors:
            self.assertTrue(actor.events)
            for event in actor.events:
                self.assertIsNotNone(event.object_id)
                self.assertGreaterEqual(event.frame_id, 0)

    def test_execution_tags_are_propagated_to_dag_nodes(self) -> None:
        chunk = VideoChunk(
            chunk_id="enc002",
            source_uri="assets/test_chunks/enc002.mp4",
            start_frame_id=0,
            fps=30.0,
            num_frames=4,
            width=320,
            height=180,
        )

        pipeline = EncoderPipeline()
        context = pipeline._dag.run(initial_context={"chunk": chunk})

        self.assertEqual(context["chunk__tag"], "cpu")
        self.assertEqual(context["panorama__tag"], "cpu")
        self.assertEqual(context["actors__tag"], "gpu")
        self.assertEqual(context["rigid_objects__tag"], "gpu")
        self.assertEqual(context["ball__tag"], "gpu")
        self.assertEqual(context["residual__tag"], "gpu")


if __name__ == "__main__":
    unittest.main()
