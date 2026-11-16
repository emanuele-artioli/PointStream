"""CPU checks for the player generator's image and geometry boundaries."""

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.tennis import k6b, k6b_fit


def test_clip_reference_uses_full_asymmetric_rgb_crop():
    image = np.zeros((784, 512, 3), dtype=np.uint8)
    image[:196, :, 0] = 255
    image[588:, :, 2] = 255
    actual = k6b_fit.clip_reference_images(image[None])[0]
    # pose2vid_long's ref_image.resize((224, 224)); PIL's default is bicubic for RGB.
    expected = Image.fromarray(image).resize((224, 224))
    assert actual.mode == "RGB" and actual.size == (224, 224)
    assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert tuple(np.asarray(actual)[10, 20]) == (255, 0, 0)
    assert tuple(np.asarray(actual)[210, 20]) == (0, 0, 255)
    assert not np.array_equal(np.asarray(actual), np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.NEAREST)))


def test_crop_coordinates_follow_rounded_source_box_and_padding():
    image = np.zeros((8, 10, 3), np.uint8)
    image[2, 1] = (255, 70, 20)
    box = np.array([-1.2, 0.2, 8.2, 8.2])
    points = np.zeros((17, 3))
    points[0] = [1, 2, 0.8]
    mapped = k6b.to_crop(points, box)
    assert np.allclose(mapped[0], [128, 196, 0.8])
    crop = k6b.crop_image(image, box)
    assert tuple(crop[245, 160]) == (255, 70, 20)
    assert not crop[:, :64].any()


def test_pinned_upstream_body_renderer_if_available():
    path = os.environ.get("POINTSTREAM_AA_UTIL")
    if not path:
        pytest.skip("set POINTSTREAM_AA_UTIL to the pinned upstream util.py for external comparison")
    source = Path(path)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "6eb8effe3ac50d31ea3140c0d084b04f099c2a0fecc9da53c4f6cf27cd186f4f"
    spec = importlib.util.spec_from_file_location("pinned_dwpose_util", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kp = np.zeros((17, 3))
    kp[:, 0] = np.arange(17) * 19 + 37
    kp[:, 1] = np.arange(17) * 29 + 41
    kp[:, 2] = 1
    kp[8, 2] = 0.2
    points, visible = k6b.openpose18(kp)
    candidate = points / np.array([k6b.GEN_W, k6b.GEN_H])
    subset = np.where(visible, np.arange(18), -1)[None, :]
    expected = module.draw_bodypose(np.zeros((k6b.GEN_H, k6b.GEN_W, 3), np.uint8), candidate, subset)
    actual = k6b.pose_map(kp)
    # DWPose normalizes and multiplies coordinates before truncation; floating
    # roundoff can move a joint or limb edge by a pixel.
    changed = np.any(actual != expected, axis=2).sum()
    assert changed < 300
