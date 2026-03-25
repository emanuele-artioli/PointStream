from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.encoder.orchestrator import EncoderPipeline
from src.shared.schemas import VideoChunk
from src.transport.disk import DiskTransport


class TestDiskTransport(unittest.TestCase):
    def test_roundtrip_payload(self) -> None:
        chunk = VideoChunk(
            chunk_id="rt001",
            source_uri="assets/test_chunks/rt001.mp4",
            start_frame_id=0,
            fps=30.0,
            num_frames=8,
            width=320,
            height=180,
        )

        payload = EncoderPipeline().encode_chunk(chunk)

        with tempfile.TemporaryDirectory() as tmp_dir:
            transport = DiskTransport(root_dir=tmp_dir)
            transport.send(payload)
            recovered = transport.receive("rt001")

            self.assertEqual(recovered.chunk.chunk_id, "rt001")
            self.assertEqual(len(recovered.actors), 2)

            chunk_dir = Path(tmp_dir) / "chunk_rt001"
            self.assertTrue((chunk_dir / "metadata.msgpack").exists())
            self.assertTrue((chunk_dir / "residual.mp4").exists())

    def test_receive_missing_payload_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            transport = DiskTransport(root_dir=tmp_dir)
            with self.assertRaises(FileNotFoundError):
                transport.receive("missing_chunk")


if __name__ == "__main__":
    unittest.main()
