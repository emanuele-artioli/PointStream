from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.segmentation import load_domain
from src.segmentation.evaluate import boundary_f, compare, region_scores, summarize
from src.segmentation.masks import ClipMasks
from src.segmentation.yoloe import TrackFilter, YoloeSegmenter, detections_from_result


def _square(size: int = 40, x: int = 10, y: int = 10, side: int = 12) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y:y + side, x:x + side] = True
    return mask


def test_region_scores_separate_missed_and_extra_foreground() -> None:
    ref = _square()
    pred = _square(x=16)  # half of the reference, plus as much outside it
    scores = region_scores(pred, ref)
    assert scores["iou"] == pytest.approx(72 / 216)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["precision"] == pytest.approx(0.5)
    empty = np.zeros_like(ref)
    assert region_scores(empty, empty)["iou"] == 1.0
    assert region_scores(empty, ref)["recall"] == 0.0
    assert region_scores(pred, empty)["precision"] == 0.0


def test_boundary_f_tolerates_small_shifts_only() -> None:
    ref = _square(size=200, x=50, y=50, side=60)
    assert boundary_f(ref, ref) == 1.0
    near = boundary_f(_square(size=200, x=51, y=50, side=60), ref)
    far = boundary_f(_square(size=200, x=80, y=50, side=60), ref)
    assert near > 0.9 > far
    assert boundary_f(np.zeros_like(ref), ref) == 0.0


def _clip(masks: list[np.ndarray], classes: tuple[str, ...] = ("player",)) -> ClipMasks:
    clip = ClipMasks(classes, *masks[0].shape, 25.0)
    clip.ensure_frames(len(masks))
    for index, mask in enumerate(masks):
        clip.add(index, classes[0], 1, mask)
    return clip


def test_compare_scores_identity_as_perfect_and_reports_flicker() -> None:
    steady = [_square(x=10 + t) for t in range(4)]
    flickering = [steady[0], np.zeros_like(steady[0]), steady[2], np.zeros_like(steady[0])]
    ref = _clip(steady)
    same = compare(_clip(steady), ref)
    fg = same["scopes"]["foreground"]
    assert fg["J"] == fg["F"] == fg["recall"] == 1.0
    assert same["scopes"]["player"]["J"] == 1.0
    noisy = compare(_clip(flickering), ref)["scopes"]["foreground"]
    assert noisy["recall"] == 0.5
    assert noisy["flicker"] > noisy["reference_flicker"]

    rows = [{"backend": "b", "frames": 4, "scopes": {"foreground": fg}, "ms_per_frame": 10.0, "fps": 100.0,
             "peak_gpu_mib": None}]
    table = summarize(rows)
    assert table[0]["J"] == 1.0 and table[0]["ms_per_frame"] == 10.0 and table[0]["peak_gpu_mib"] is None


def test_track_filter_needs_hits_holds_misses_and_gates_new_tracks() -> None:
    hand = _square()
    tracker = TrackFilter(new_track_conf=0.5, min_hits=2, hold_frames=1)
    assert tracker.update([("hand", hand, 0.9)]) == []  # one hit is not enough
    first = tracker.update([("hand", _square(x=11), 0.2)])  # a weak box may extend a track
    assert [(name, tid) for name, tid, _m, _s in first] == [("hand", 1)]
    held = tracker.update([])
    assert len(held) == 1  # held through one miss
    assert tracker.update([]) == []
    assert tracker.update([("hand", hand, 0.2)]) == []  # weak boxes cannot start a track


def test_track_filter_drops_small_masks_relative_to_the_largest_of_a_class() -> None:
    tracker = TrackFilter(min_area_ratio=0.5)
    big, small = _square(side=20), _square(x=0, y=0, side=4)
    out = tracker.update([("hand", big, 0.9), ("hand", small, 0.9), ("arm", small, 0.9)])
    assert sorted((name, int(mask.sum())) for name, _t, mask, _s in out) == [("arm", 16), ("hand", 400)]


def _result(masks: np.ndarray, cls: list[int], conf: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        masks=SimpleNamespace(data=masks.astype(np.float32)),
        boxes=SimpleNamespace(cls=np.asarray(cls), conf=np.asarray(conf)),
        orig_shape=masks.shape[1:],
    )


def test_detections_keep_the_largest_component_and_drop_unknown_classes() -> None:
    mask = _square()
    mask[0, 0] = True  # a stray pixel
    found = detections_from_result(_result(np.stack([mask, mask]), [1, 7], [0.4, 0.9]), ("arm", "hand"))
    assert [(name, score) for name, _m, score in found] == [("hand", pytest.approx(0.4))]
    assert not found[0][1][0, 0]


class _StubYoloe:
    def __init__(self) -> None:
        self.classes: list[str] = []

    def set_classes(self, classes: list[str]) -> None:
        self.classes = classes

    def predict(self, source: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        mask = np.zeros(source.shape[:2], dtype=bool)
        mask[2:6, 2:6] = True
        return [_result(mask[None], [1], [0.9])]


def test_yoloe_segment_writes_tracked_instances_for_every_frame(tmp_path: Path) -> None:
    import cv2

    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(7):
        cv2.imwrite(str(frames / f"{index:03d}.png"), np.zeros((8, 10, 3), np.uint8))
    model = _StubYoloe()
    segmenter = YoloeSegmenter("n", model=model, device="cpu")
    masks = segmenter.segment(frames, load_domain("egocentric"))
    assert model.classes == ["arm", "hand"]
    assert len(masks) == 7 and masks.classes == ("arm", "hand")
    # egocentric needs five hits before a hand is emitted
    assert [len(f) for f in masks.frames] == [0, 0, 0, 0, 1, 1, 1]
    assert masks.frames[4][0].class_name == "hand" and masks.frames[4][0].track_id == 1
    assert masks.meta["backend"] == "yoloe-26n" and masks.meta["timing"]["frames"] == 6
    with pytest.raises(TypeError, match="unknown YOLOE options"):
        YoloeSegmenter("n", bogus=1)
