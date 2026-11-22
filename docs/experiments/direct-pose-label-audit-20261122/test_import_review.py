"""Synthetic-only tests: never reuse generated marks as scientific labels."""

import copy
import hashlib
import json
import tempfile
import unittest

from PIL import Image
from pathlib import Path

import import_review


class ImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        (self.path / "frames").mkdir()
        image = self.path / "frames" / "010.png"
        Image.new("RGB", (1920, 1080)).save(image)
        self.frame_sha = hashlib.sha256(image.read_bytes()).hexdigest()
        self.spec = {
            "schema": "pointstream-body-joints-v1",
            "source_sha256": "synthetic",
            "width": 1920,
            "height": 1080,
            "fps": 50,
            "actors": ["p0", "p1"],
            "joints": ["nose", "left_eye"],
            "synthetic_fixture": True,
            "frames": [
                {
                    "tick": 10,
                    "timestamp_seconds": 0.2,
                    "sha256": self.frame_sha,
                    "file": "frames/010.png",
                }
            ],
        }
        self.manifest = self.path / "manifest.json"
        raw = json.dumps(self.spec, sort_keys=True, separators=(",", ":")).encode()
        self.manifest.write_bytes(raw)
        self.export = {
            "schema": self.spec["schema"],
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "source_sha256": "synthetic",
            "width": 1920,
            "height": 1080,
            "fps": 50,
            "frames": [
                {
                    "tick": 10,
                    "timestamp_seconds": 0.2,
                    "source_png_sha256": self.frame_sha,
                    "actors": [
                        {
                            "id": p,
                            "joints": [
                                {"name": "nose", "status": "visible", "x": 123, "y": 456},
                                {"name": "left_eye", "status": "unsure", "x": None, "y": None},
                            ],
                        }
                        for p in ["p0", "p1"]
                    ],
                }
            ],
        }
        self.file = self.path / "export.json"

    def tearDown(self):
        self.temp.cleanup()

    def check(self, value):
        self.file.write_text(json.dumps(value))
        return import_review.validate(self.manifest, self.file)

    def test_valid_fixture(self):
        x = self.check(self.export)
        self.assertTrue(x["synthetic_fixture"])
        self.assertEqual(x["status_counts"]["visible"], 2)
        self.assertEqual(x["status_counts"]["unsure"], 2)

    def test_rejects_identity_time_bounds_and_missing(self):
        for kind in ("actor", "time", "bounds", "missing", "false_coordinate", "digest"):
            x = copy.deepcopy(self.export)
            if kind == "actor":
                x["frames"][0]["actors"][0]["id"] = "other"
            if kind == "time":
                x["frames"][0]["timestamp_seconds"] = 0.22
            if kind == "bounds":
                x["frames"][0]["actors"][0]["joints"][0]["x"] = 1920
            if kind == "missing":
                x["frames"][0]["actors"][0]["joints"].pop()
            if kind == "false_coordinate":
                x["frames"][0]["actors"][0]["joints"][1]["x"] = 10
            if kind == "digest":
                x["manifest_sha256"] = "other"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.check(x)


if __name__ == "__main__":
    unittest.main()
