import numpy as np

from experiments.tennis import k4b_court as kc
from experiments.tennis.k4b_features import key
from src.segmentation.masks import ClipMasks

# 1 m = 40 px, court origin at (100, 50) px.
H = np.array([[40., 0, 100], [0, 40., 50], [0, 0, 1]])


def test_foot_is_the_lowest_row_at_the_centroid_x_in_1080p():
    m = np.zeros((540, 960), bool)
    m[10:20, 30:40] = True
    assert kc.foot(m) == (69., 40.)


def test_inside_allows_one_metre_at_the_sides_and_four_behind_the_baselines():
    px = lambda x, y: (100 + 40 * x, 50 + 40 * y)  # noqa: E731
    assert kc.inside(px(5, 12), H) and kc.inside(px(-0.9, 0), H) and kc.inside(px(11.9, 27.7), H)
    assert not kc.inside(px(-1.1, 12), H) and not kc.inside(px(5, -4.1), H) and not kc.inside(px(5, 27.9), H)
    behind = np.array([[1., 0, 0], [0, 1, 0], [0, 0, -1]])
    assert not kc.inside((0., 0.), behind)


def test_d_counts_registered_frames_only_and_merged_prompts_once():
    clip = ClipMasks(("person", "player"), 540, 960, 25.)
    for t in range(4):
        on = np.zeros((540, 960), bool)
        on[100:140, 300:310] = True  # foot (610, 280) px -> (12.75, 5.75) m: outside the side margin
        off = np.zeros_like(on)
        off[100:140, 100:110] = True  # foot (210, 280) -> (2.75, 5.75) m: inside
        clip.add(t, "person", 1, off, .9)
        clip.add(t, "player", 4, off, .9)
        clip.add(t, "person", 2, on, .9)
    aliases = {key("person", 1): key("person", 1), key("player", 4): key("person", 1), key("person", 2): key("person", 2)}
    court = kc.court_features(clip, aliases, 10, {10: H, 12: H, 13: H})
    assert court[key("person", 1)] == {"registered": 3, "inside": 3, "d": 1., "inside_most": True}
    assert court[key("person", 2)]["d"] == 0. and court[key("person", 2)]["registered"] == 3


def test_court_candidates_rank_inside_first():
    tracks = {"p:1": dict(presence=1., a=1., c=.9, racket_most=True), "p:2": dict(presence=1., a=9., c=0., racket_most=False),
              "p:3": dict(presence=1., a=2., c=.6, racket_most=True), "p:4": dict(presence=.2, a=50., c=1., racket_most=True)}
    court = {"p:1": dict(d=1., inside_most=True), "p:2": dict(d=.2, inside_most=False), "p:3": dict(d=.9, inside_most=True),
             "p:4": dict(d=1., inside_most=True)}
    picks = kc.choose_d(tracks, court)
    assert picks == {"d": ["p:1", "p:3"], "c+d": ["p:1", "p:3"], "a+d": ["p:3", "p:1"]}
    assert kc.choose_d(tracks, {"p:2": dict(d=1., inside_most=True)})["d"] == ["p:2"]
