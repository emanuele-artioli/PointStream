"""Reproduce and authenticate the archived Mac-built 240-frame Shanghai raster."""

import argparse
import json
from pathlib import Path
import sys

import cv2

from src.pointstream import ball

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test-crop-joint-20261122"))
import build_joint as sender  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "features", "ball", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    inputs = parser.parse_args()

    track = json.loads((sender.HERE / "match102-result.json").read_text())
    original = json.loads((sender.HERE / "match102-sender.json").read_text())
    assert sender.sha(inputs.source) == track["source_sha256"]
    assert sender.sha(inputs.features) == track["prior_feature_archive_sha256"]
    assert sender.sha(inputs.ball) == "3ad726835c0e51c11daa80d9b54630a1e52494227680b434254e583950426918"
    assert cv2.__version__ == original["source_opencv"]

    inputs.out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    positions, _ = ball.decode_frames(inputs.ball.read_bytes(), 240, 960, 544)
    source = sender.raster_source(inputs)
    masks = sender.spatial_masks(inputs, track, positions)
    result = sender.donor_base(inputs, source, masks, smoke_only=False)
    for key in ("smoke", "donors", "holes", "painted_area", "mask_sha256", "clean_yuv_sha256"):
        assert result[key] == original["donor"][key], key
    (inputs.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Original clean YUV SHA-256", sender.sha(inputs.out / "clean.yuv"))


if __name__ == "__main__":
    main()
