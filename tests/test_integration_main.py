from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.encoder.execution_pool import TaggedMultiprocessPool, WorkerConfig
from src.main import run_mock_pipeline


class TestMainIntegration(unittest.TestCase):
    def test_run_mock_pipeline_emits_summary_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_mock_pipeline(transport_root=tmp_dir)

            self.assertEqual(summary["chunk_id"], "0001")
            self.assertEqual(summary["num_actor_packets"], 2)
            self.assertEqual(summary["num_rigid_object_packets"], 1)
            self.assertEqual(summary["ball_object_id"], "ball_0")

            chunk_dir = Path(tmp_dir) / "chunk_0001"
            self.assertTrue((chunk_dir / "metadata.msgpack").exists())
            self.assertTrue((chunk_dir / "residual.mp4").exists())

    def test_run_mock_pipeline_with_tagged_execution_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool = TaggedMultiprocessPool(config=WorkerConfig(cpu_workers=1, gpu_workers=1))
            summary = run_mock_pipeline(transport_root=tmp_dir, execution_pool=pool)

            self.assertEqual(summary["chunk_id"], "0001")
            self.assertGreaterEqual(pool.cpu_dispatch_count, 1)
            self.assertGreaterEqual(pool.gpu_dispatch_count, 1)

            chunk_dir = Path(tmp_dir) / "chunk_0001"
            self.assertTrue((chunk_dir / "metadata.msgpack").exists())


if __name__ == "__main__":
    unittest.main()
