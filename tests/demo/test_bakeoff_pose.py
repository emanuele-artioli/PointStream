from __future__ import annotations

from pathlib import Path

import pytest

from demo.experiments.bakeoff_pose import (
    TOPOLOGY_NOTE,
    as_hand_poses,
    centroid_distances,
    main as bakeoff_main,
    n_hands,
)
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand


def _hand(x: float, y: float, n: int = 21) -> SingleHand:
    lms = [[x + i, y] for i in range(n)]
    return SingleHand(
        "Right",
        0.9,
        [int(x), int(y), int(x) + n, int(y) + 4],
        [[p[0] / 1920, p[1] / 1080, 0.0] for p in lms],
        lms,
    )


def test_centroid_when_both_fire() -> None:
    a = [FrameHandPose(0, [_hand(100, 100)]), FrameHandPose(1, [_hand(200, 200)])]
    b = [FrameHandPose(0, [_hand(110, 100)]), FrameHandPose(1, [])]
    stats = centroid_distances(a, b)
    assert stats["n_pairs"] == 1
    assert stats["mean"] == pytest.approx(10.0, abs=0.2)
    assert n_hands(a) == 2
    assert n_hands(b) == 1


def test_no_mpjpe_in_bakeoff_skip(tmp_path: Path) -> None:
    out = tmp_path / "maps"
    code = bakeoff_main(["--out", str(out)])
    assert code == 0
    text = (out / "bakeoff_pose.json").read_text()
    assert "skipped" in text
    assert '"mpjpe": null' in text
    assert "not the same" in TOPOLOGY_NOTE or "index-aligned" in TOPOLOGY_NOTE


def test_as_hand_poses_passthrough() -> None:
    poses = [FrameHandPose(0, [])]
    assert as_hand_poses(poses) is poses


def test_bakeoff_missing_clip_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "bakeoff_pose.json"
    code = bakeoff_main(["--clip", str(tmp_path / "missing.mp4"), "--out", str(out)])
    assert code == 0
    assert out.is_file()
    assert '"skipped": true' in out.read_text()


@pytest.mark.integration
def test_bakeoff_on_clip_integration() -> None:
    pytest.skip("optional --clip; run on gpu1 with a real file")
