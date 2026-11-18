import numpy as np
import pytest

from experiments.tennis import common_player_finish as finish


def test_missing_and_empty_masks_leave_decoded_base_and_keep_player_order():
    base = np.full((1080, 1920, 3), 11, np.uint8)
    far = np.full((784, 512, 3), 37, np.uint8)
    near = np.full_like(far, 89)
    box = np.array([20, 30, 40, 50])
    mask = np.ones((784, 512), bool)
    active = {"far": (far, box), "near": (near, box)}
    image, row = finish.compose_tick(base, ["far", "near"], active, {"far": mask, "near": mask})
    assert row["coverage_pixels"] == 40 * 50
    assert (image[30:80, 20:60] == 89).all()
    assert np.array_equal(image[:30], base[:30])
    missing, row = finish.compose_tick(
        base, ["far", "near"], {"far": active["far"]}, {"far": ~mask}
    )
    assert row["status"] == {"far": "empty_mask", "near": "missing_pose"}
    assert np.array_equal(missing, base)
    offscreen, row = finish.compose_tick(
        base, ["far"], {"far": (far, box + [2100, 0, 0, 0])}, {"far": mask}
    )
    assert row["status"] == {"far": "offscreen"} and np.array_equal(offscreen, base)
    absent, row = finish.compose_tick(base, ["far"], {"far": (far, box)}, {})
    assert row["status"] == {"far": "insufficient_prompt"} and np.array_equal(absent, base)
    assert finish.crop_prompt(np.zeros((17, 3)), box) is None


def test_generation_seal_cannot_substitute_boxes_or_drop_coverage(tmp_path):
    times = np.arange(2)
    boxes = np.ones((2, 4))
    crops = np.zeros((2, 784, 512, 3), np.uint8)
    geometry = (times, None, boxes)
    p = tmp_path / "crop.npz"
    np.savez(p, t=times, box=boxes, crops=crops)
    assert np.array_equal(finish.checked_crops("athlete", p, geometry, 2), crops)
    np.savez(p, t=times, box=boxes + 1, crops=crops)
    with pytest.raises(ValueError, match="geometry"):
        finish.checked_crops("athlete", p, geometry, 2)
    np.savez(p, t=times, box=boxes, crops=crops[:1])
    with pytest.raises(ValueError, match="coverage"):
        finish.checked_crops("athlete", p, geometry, 2)
