from dataclasses import dataclass

import numpy as np
import pytest

from experiments.visor import h4c, k1
from src.segmentation import visor


@dataclass
class Inst:
    class_name: str
    label: str
    box: tuple[int, int, int, int]  # y0, y1, x0, x1

    def mask(self) -> np.ndarray:
        m = np.zeros(visor.FRAME_SIZE, bool)
        y0, y1, x0, x1 = self.box
        m[y0:y1, x0:x1] = True
        return m


class Clip:
    def __init__(self, frames: list[list[Inst]]):
        self.frames = frames

    def class_mask(self, t: int, name: str) -> np.ndarray:
        out = np.zeros(visor.FRAME_SIZE, bool)
        for inst in self.frames[t]:
            if inst.class_name == name:
                out |= inst.mask()
        return out


def test_handled_objects_exclude_hands_and_pixels_an_earlier_object_took():
    hand = Inst("left hand", "left hand", (0, 10, 0, 10))
    cup = Inst("active object", "cup", (0, 20, 0, 20))  # handled; under the hand at the top left
    board = Inst("active object", "board", (10, 30, 10, 30))  # not handled; painted after the cup
    jar = Inst("active object", "jar", (25, 35, 25, 35))  # handled; painted after the board, which took 25..30
    clip = Clip([[cup, board, hand, jar]])
    handled = {visor.object_key("active object", "cup"), visor.object_key("active object", "jar")}
    o = h4c.object_regions(clip, 1, handled)[0]
    expected = cup.mask() | (jar.mask() & ~board.mask())
    expected &= ~hand.mask()
    assert np.array_equal(o, expected)
    assert not (o & hand.mask()).any()
    assert o[15, 15] and not o[27, 27] and o[32, 32]


def test_point_names_cover_the_bar_and_mark_the_fitted_points():
    assert h4c.parse_point("svt@41") == ("b2", 41.0, 1)
    assert h4c.parse_point("svt@63") == ("low", 63.0, 1)
    assert h4c.parse_point("svt@55/8") == ("low", 55.0, 8)
    assert h4c.parse_point("dcvc@21/4") == ("dcvc", 21.0, 4)
    assert len(h4c.BAR) == 20 and len(set(h4c.BAR)) == 20
    assert h4c.FITTED == {f"dcvc@{q}/{s}" for q in (0, 21, 42) for s in (2, 4)} and h4c.FITTED <= set(h4c.BAR)
    with pytest.raises(SystemExit):
        h4c.parse_point("dcvc@7/4")


def verdict(v: str) -> dict:
    return {"verdict": {"verdict": v}}


@pytest.mark.parametrize("oracle, g, expected", [
    ("fails", "passes", "arms 2 and 3 not built; objects coded by DCVC-UF + G"),
    ("fails", "unclear", "arms 2 and 3 not built; objects coded by DCVC-UF off the shelf"),
    ("unclear", "fails", "stage 2: G_O and G_HO"),
    ("passes", "passes", "stage 2: G_O and G_HO"),
])
def test_stage_one(oracle, g, expected):
    assert h4c.stage1({"oracle": verdict(oracle), "G": verdict(g)})["next"].startswith(expected)
    assert h4c.stage1({"G": verdict(g)})["next"] == "incomplete"


def test_k1_maps_lpips_on_s_to_r_through_the_same_streams():
    pairs = [(0.1, 0.11), (0.2, 0.22), (0.3, 0.31)]
    assert k1.s_to_r(pairs, 0.15) == pytest.approx(0.165)
    assert k1.s_to_r(pairs, 0.05) is None  # outside the anchor's points
    assert k1.s_to_r([(0.1, 0.2), (0.2, 0.1)], 0.15) is None  # not monotone: no map


def test_k1_reads_the_hull_linearly_in_rate():
    hull = [(1.0, -0.3), (3.0, -0.1)]
    assert k1.at_rate(hull, 2.0) == pytest.approx(-0.2)
    assert k1.at_rate(hull, 0.5) is None
    assert k1.rate_for(hull, -0.2) == pytest.approx(2.0)
    assert k1.rate_for(hull, -0.05) is None


def test_segments_need_consecutive_frames_within_the_admitted_drift():
    alignment = {n: {"index": n - 1, "drift": 0} for n in range(1, 11)}
    alignment[6] = {"index": 5, "drift": 2}  # drifts too far
    assert h4c.segment_starts(alignment, 1, 10, 3, 1) == [1, 2, 3, 7, 8]
    alignment[6] = {"index": 6, "drift": 1}  # drift admitted, but frames 5 and 7 then share a decoded frame with it
    assert h4c.segment_starts(alignment, 1, 10, 3, 1) == [1, 2, 3, 7, 8]
    alignment[6] = {"index": 5, "drift": 1}
    assert h4c.segment_starts(alignment, 1, 10, 3, 1) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_contacts_name_the_objects_a_hand_touches():
    def entry(number, annotations):
        return {"image": {"name": f"P04_01_frame_{number:010d}.jpg"}, "annotations": annotations}

    doc = {"video_annotations": [entry(5, [
        {"id": "h", "name": "left hand", "class_id": 300, "segments": [], "in_contact_object": "c"},
        {"id": "c", "name": "cup", "class_id": 7, "segments": []},
        {"id": "r", "name": "right hand", "class_id": 301, "segments": [], "in_contact_object": "none-of-these"},
    ])]}
    assert h4c.contacts(doc) == {5: {"cup"}}


@pytest.mark.parametrize("o, r, versus, expected", [
    ({"G": "passes", "G_HO": "passes", "G_O": "passes"}, {}, {"G_HO>G:o": -0.05, "G_O>G:o": -0.05}, ("G", "G", 1)),
    ({"G": "passes", "G_HO": "passes", "G_O": "passes"}, {"G_HO": "passes"},
     {"G_HO>G:o": -0.2, "G_O>G_HO:o": -0.05, "G_HO>G:r": 0.03}, ("G_HO", "G_HO", 1)),
    ({"G": "unclear", "G_HO": "passes", "G_O": "passes"}, {"G_HO": "passes"},
     {"G_O>G_HO:o": -0.05, "G_HO>G:r": 0.25}, ("G_HO", "G", 2)),
    ({"G": "fails", "G_HO": "unclear", "G_O": "passes"}, {}, {}, ("G_O", "G", 2)),
    ({"G": "passes", "G_HO": "passes", "G_O": "passes"}, {"G_HO": "passes"},
     {"G_HO>G:o": -0.2, "G_O>G_HO:o": -0.15, "G_HO>G:r": 0.0}, ("G_O", "G", 2)),
    ({"G": "fails", "G_HO": "fails", "G_O": "unclear"}, {}, {}, ("DCVC-UF off the shelf", "G", 1)),
])
def test_stage_two(o, r, versus, expected):
    d = h4c.stage2(o, r, versus)
    assert (d["objects"].split(" (")[0], d["hands"], d["networks"]) == expected
