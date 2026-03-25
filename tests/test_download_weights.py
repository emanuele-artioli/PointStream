from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.download_weights import ensure_weights


class TestDownloadWeights(unittest.TestCase):
    def test_ensure_weights_passes_when_all_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_dir = Path(tmp_dir)
            for name in ["a.pt", "b.pt"]:
                (weights_dir / name).write_bytes(b"x")

            ensure_weights(weights_dir, ["a.pt", "b.pt"])

    def test_ensure_weights_raises_with_missing_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_dir = Path(tmp_dir)
            (weights_dir / "a.pt").write_bytes(b"x")

            with self.assertRaises(FileNotFoundError) as ctx:
                ensure_weights(weights_dir, ["a.pt", "b.pt", "c.pt"])

            message = str(ctx.exception)
            self.assertIn("- b.pt", message)
            self.assertIn("- c.pt", message)


if __name__ == "__main__":
    unittest.main()
