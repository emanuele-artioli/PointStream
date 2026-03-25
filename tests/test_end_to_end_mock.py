from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.orchestrator import EncoderPipeline
from src.transport.disk import DiskTransport


class TestEndToEndMock(unittest.TestCase):
    def test_end_to_end_mp4_encode_transport_decode(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        assets_dir = project_root / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        video_path = assets_dir / "test_video.mp4"
        self._create_dummy_video(video_path, num_frames=30, width=96, height=64, fps=30.0)

        transport_root = project_root / ".pointstream"
        chunk_id = "e2e_mock_0001"
        chunk_dir = transport_root / f"chunk_{chunk_id}"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

        encoder = EncoderPipeline()
        try:
            payload, decoded_video_tensor = encoder.encode_video_file(
                video_path=video_path,
                chunk_id=chunk_id,
            )
        finally:
            encoder.shutdown()

        # Shape: [Frames, Channels, Height, Width]
        self.assertEqual(tuple(decoded_video_tensor.shape), (30, 3, 64, 96))

        transport = DiskTransport(root_dir=transport_root)
        transport.send(payload)

        self.assertTrue((chunk_dir / "metadata.msgpack").exists())
        self.assertTrue((chunk_dir / "residual.mp4").exists())

        recovered_payload = transport.receive(chunk_id)
        decoded_result = DecoderRenderer().process(recovered_payload)

        self.assertEqual(recovered_payload.chunk.chunk_id, chunk_id)
        self.assertEqual(recovered_payload.chunk.num_frames, 30)
        self.assertEqual(recovered_payload.chunk.width, 96)
        self.assertEqual(recovered_payload.chunk.height, 64)
        self.assertEqual(len(recovered_payload.actors), 2)
        self.assertEqual(len(recovered_payload.rigid_objects), 1)
        self.assertEqual(recovered_payload.ball.object_id, "ball_0")

        self.assertEqual(decoded_result.chunk_id, chunk_id)
        self.assertEqual(decoded_result.num_frames, 30)
        self.assertEqual(decoded_result.width, 96)
        self.assertEqual(decoded_result.height, 64)

    @staticmethod
    def _create_dummy_video(
        video_path: Path,
        num_frames: int,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_path),
            getattr(cv2, "VideoWriter_fourcc")(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create dummy video at: {video_path}")

        for frame_idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x0 = (frame_idx * 2) % max(1, width - 20)
            y0 = height // 3
            cv2.rectangle(frame, (x0, y0), (x0 + 18, y0 + 14), (0, 200, 50), thickness=-1)
            writer.write(frame)

        writer.release()


if __name__ == "__main__":
    unittest.main()
