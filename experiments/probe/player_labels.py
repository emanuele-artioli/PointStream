"""Who is in each track, labelled by eye, because the dataset does not say.

`BP18` needs one anchor that nothing else in this repo can supply: **two
different players in the same match**. Without it there is no way to tell an
identity metric that works from one that merely scores "person on a tennis
court" highly, and a metric like that produces perfectly ordered rankings while
measuring nothing — which is how the uncalibrated LPIPS shipped.

**Provenance.** Hand-labelled on 2026-08-23 by inspecting the first frame of
each track in a contact sheet, one sheet per video. Kit is the cue: within a
single broadcast the two players wear different colours, and officials wear
something different again. Three of the five training-split videos are labelled;
the other two are omitted rather than guessed at — in `alcaraz_ruud` every
sampled track wears the same rust shirt and white shorts, so no honest
different-player pair can be drawn from it.

**The circularity, stated because it is real.** Labels come from clothing, and
clothing is much of what a ReID embedding keys on. So a *cross-track* label is
partly the thing being tested. Two things keep the calibration honest:

* the primary same-player anchor is **the same track at a different frame**,
  which is ground truth — a track is one person by construction of the tracker,
  and needs no label from this file;
* the different-player pairs here are *conservative*, because both players share
  a court, a broadcast and a lighting setup, all of which push their embeddings
  together.

What this file therefore supports is the narrower, honest claim: within a match,
**can the metric tell the two players apart**. For this project that is the
question that matters — a reconstruction of the wrong player is the failure to
catch — but it is not a claim about identity in general.

`OFFICIAL` marks umpires and line judges. They are not players and are kept
separate rather than dropped: "player versus official" is a useful third anchor,
and a metric that scores them as similar as two players is telling you something.
"""

from __future__ import annotations

OFFICIAL = "official"

#: ``video -> {"scene/track": player label}``. Labels are per video; ``"A"`` in
#: one video has nothing to do with ``"A"`` in another.
PLAYER_LABELS: dict[str, dict[str, str]] = {
    "alcaraz_perricard": {
        # black adidas tracksuit, arms folded by the net
        "scene_003/track_0078": OFFICIAL,
        "scene_004/track_0089": OFFICIAL,
        # teal / mint shirt with print, black shorts
        "scene_005/track_0147": "A",
        "scene_007/track_0064": "A",
        "scene_002/track_0002": "A",
        # plain black shirt, white shorts
        "scene_006/track_0134": "B",
        "scene_006/track_0196": "B",
        "scene_007/track_0008": "B",
        "scene_010/track_0001": "B",
    },
    "federer_djokovic": {
        # grey / taupe shirt, white shorts, white cap
        "scene_007/track_0061": "A",
        "scene_013/track_0073": "A",
        "scene_015/track_0010": "A",
        "scene_017/track_0006": "A",
        # pink / red shirt, navy shorts
        "scene_013/track_0084": "B",
        "scene_015/track_0001": "B",
        "scene_017/track_0057": "B",
        "scene_019/track_0001": "B",
    },
    "sinner_alcaraz": {
        # maroon / burgundy shirt, teal shorts, cap
        "scene_001/track_0002": "A",
        "scene_006/track_0035": "A",
        "scene_008/track_0005": "A",
        "scene_012/track_0001": "A",
        # pink shirt, cream shorts
        "scene_001/track_0003": "B",
        "scene_002/track_0036": "B",
        "scene_014/track_0054": "B",
        "scene_018/track_0001": "B",
        "scene_021/track_0003": "B",
        "scene_012/track_0058": "B",
    },
}


def label_for(video: str, scene: str, track: str) -> str | None:
    """The player label for one track, or None when it was never labelled."""
    return PLAYER_LABELS.get(video, {}).get(f"{scene}/{track}")


def labelled_tracks(video: str) -> dict[str, str]:
    """Every labelled ``"scene/track"`` in one video."""
    return dict(PLAYER_LABELS.get(video, {}))


def same_player_pairs(video: str, *, include_officials: bool = False) -> list[tuple[str, str]]:
    """Distinct labelled tracks of the same person, within one video."""
    return _pairs(video, same=True, include_officials=include_officials)


def different_player_pairs(
    video: str, *, include_officials: bool = False
) -> list[tuple[str, str]]:
    """Distinct labelled tracks of different people, within one video."""
    return _pairs(video, same=False, include_officials=include_officials)


def _pairs(video: str, *, same: bool, include_officials: bool) -> list[tuple[str, str]]:
    items = [
        (key, label)
        for key, label in sorted(PLAYER_LABELS.get(video, {}).items())
        if include_officials or label != OFFICIAL
    ]
    out: list[tuple[str, str]] = []
    for index, (left, left_label) in enumerate(items):
        for right, right_label in items[index + 1 :]:
            if (left_label == right_label) is same:
                out.append((left, right))
    return out
