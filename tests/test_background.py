from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.transport.disk import DiskTransport


class TestBackgroundModeler(unittest.TestCase):
    def test_real_video_background_stitching_and_transport(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        video_path = project_root / "assets" / "real_tennis.mp4"
        if not video_path.exists():
            self.skipTest("Expected test asset is missing: assets/real_tennis.mp4")

        transport_root = project_root / ".pointstream"
        chunk_id = "background_real_0001"
        chunk_dir = transport_root / f"chunk_{chunk_id}"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

        debug_panorama_path = project_root / "assets" / "debug_panorama.jpg"
        if debug_panorama_path.exists():
            debug_panorama_path.unlink()

        metadata = probe_video_metadata(video_path)
        streamed_preview = list(
            iter_video_frames_ffmpeg(
                video_path,
                width=metadata.width,
                height=metadata.height,
            )
        )
        self.assertTrue(streamed_preview)

        encoder = EncoderPipeline()
        try:
            payload, _decoded_video_tensor = encoder.encode_video_file(
                video_path=video_path,
                chunk_id=chunk_id,
            )
        finally:
            encoder.shutdown()

        panorama = payload.panorama
        self.assertEqual(len(panorama.homography_matrices), payload.chunk.num_frames)
        self.assertTrue(panorama.selected_frame_indices)
        self.assertEqual(panorama.selected_frame_indices[0], 0)
        self.assertLessEqual(len(panorama.selected_frame_indices), payload.chunk.num_frames)

        homography_0 = np.asarray(panorama.homography_matrices[0], dtype=np.float64)
        self.assertEqual(homography_0.shape, (3, 3))

        panorama_array = np.asarray(panorama.panorama_image, dtype=np.uint8)
        self.assertEqual(panorama_array.shape[0], payload.chunk.height)
        self.assertEqual(panorama_array.shape[1], payload.chunk.width)
        self.assertEqual(panorama_array.shape[2], 3)

        self.assertTrue(debug_panorama_path.exists())
        debug_img = cv2.imread(str(debug_panorama_path))
        self.assertIsNotNone(debug_img)
        self.assertGreater(int(np.max(debug_img)), 0)

        transport = DiskTransport(root_dir=transport_root)
        transport.send(payload)
        recovered = transport.receive(chunk_id)
        decoded = DecoderRenderer().process(recovered)

        self.assertEqual(recovered.chunk.chunk_id, chunk_id)
        self.assertEqual(len(recovered.panorama.homography_matrices), recovered.chunk.num_frames)
        self.assertTrue((chunk_dir / "metadata.msgpack").exists())
        self.assertTrue((chunk_dir / "residual.mp4").exists())

        self.assertEqual(decoded.chunk_id, chunk_id)
        self.assertEqual(decoded.width, recovered.chunk.width)
        self.assertEqual(decoded.height, recovered.chunk.height)


if __name__ == "__main__":
    unittest.main()
