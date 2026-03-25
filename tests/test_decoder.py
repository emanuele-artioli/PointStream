from __future__ import annotations

import unittest

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.orchestrator import EncoderPipeline
from src.shared.schemas import VideoChunk


class TestDecoderRenderer(unittest.TestCase):
    def test_decoder_output_matches_chunk_dimensions(self) -> None:
        chunk = VideoChunk(
            chunk_id="dec001",
            source_uri="assets/test_chunks/dec001.mp4",
            start_frame_id=0,
            fps=30.0,
            num_frames=6,
            width=640,
            height=360,
        )

        payload = EncoderPipeline().encode_chunk(chunk)
        decoded = DecoderRenderer().process(payload)

        self.assertEqual(decoded.chunk_id, "dec001")
        self.assertEqual(decoded.num_frames, 6)
        self.assertEqual(decoded.width, 640)
        self.assertEqual(decoded.height, 360)
        self.assertTrue(decoded.output_uri.endswith("dec001.mp4"))


if __name__ == "__main__":
    unittest.main()
