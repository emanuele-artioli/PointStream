"""Who is in each track — derived from the data first, labelled by eye only where it cannot be.

`BP18` needs one anchor nothing else supplies: **two different players in the
same match**. Without it there is no way to tell an identity metric that works
from one that merely scores "person on a tennis court" highly, and a metric like
that produces perfectly ordered rankings while measuring nothing.

**The good source is derivation, not labelling.** ``track_<id>_metadata.json``
carries a ``frame_id`` per entry, so two tracks in one scene whose frame ranges
**overlap are two people on court at the same instant** — necessarily different
individuals, with no judgement from anyone. `cooccurring_pairs` is that, and it
is what the calibration gate uses. Pairing them *at a shared frame* holds the
camera, the lighting and the moment fixed too.

Combined with **same track, different frame** — one person by construction of
the tracker — the decisive calibration needs no labels at all:

* same person  = same track, two frames
* different people = two tracks visible in the same frame

**The hand labels below are secondary** and no longer carry the gate. They were
written first (2026-08-23, from contact sheets, kit as the cue) and are kept for
two jobs derivation cannot do: marking **officials**, who are neither player,
and supplying *same-player-across-scenes* pairs, which no metadata implies. Both
are reported beside the derived anchors, never instead of them.

**The circularity, stated because it is real, and now confined.** Hand labels
come from clothing and clothing is much of what an embedding reads, so a
cross-track *label* is partly the thing being tested. That weakness no longer
touches the gate — it only affects the two secondary anchors, which are marked
"inferred" wherever they are reported.

Nothing here is training data. These pairs are read only when calibrating a
metric; no model is fitted to them.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

OFFICIAL = "official"

DATASET_ROOT = Path("assets") / "dataset"

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


# ---------------------------------------------------------------------------
# Derived from the data. No judgement, and what the calibration gate uses.


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scene_dir(video: str, scene: str, root: Path | None = None) -> Path:
    return (root or repo_root()) / DATASET_ROOT / video / "segmentations" / scene


@lru_cache(maxsize=None)
def track_frame_ids(video: str, scene: str, track: str) -> tuple[int, ...]:
    """Source frame ids this track appears in, in file order.

    The position of an id in this tuple is also the index of the matching crop
    PNG — the same positional pairing the rest of the probe uses, and the reason
    a filename is never rebuilt from an id.
    """
    path = _scene_dir(video, scene) / f"{track}_metadata.json"
    if not path.is_file():
        return ()
    records = json.loads(path.read_text())
    return tuple(int(record["frame_id"]) for record in records if "frame_id" in record)


def cooccurring_pairs(video: str, root: Path | None = None) -> list[tuple[str, str, int]]:
    """``(scene/trackA, scene/trackB, shared frame id)`` for tracks seen together.

    Two tracks visible in the same source frame are two different people. This
    is ground truth from the annotation, not an opinion about anybody's shirt.
    """
    base = (root or repo_root()) / DATASET_ROOT / video / "segmentations"
    if not base.is_dir():
        return []
    out: list[tuple[str, str, int]] = []
    for scene_dir in sorted(base.iterdir()):
        if not scene_dir.is_dir():
            continue
        tracks = sorted(
            item.name
            for item in scene_dir.iterdir()
            if item.is_dir() and not item.name.endswith(_DERIVED_SUFFIXES)
        )
        seen = {
            track: set(track_frame_ids(video, scene_dir.name, track)) for track in tracks
        }
        for index, left in enumerate(tracks):
            for right in tracks[index + 1 :]:
                shared = seen[left] & seen[right]
                if shared:
                    out.append(
                        (
                            f"{scene_dir.name}/{left}",
                            f"{scene_dir.name}/{right}",
                            min(shared),
                        )
                    )
    return out


def crop_index_of(video: str, key: str, frame_id: int) -> int | None:
    """Where ``frame_id`` sits in this track's file order, or None."""
    scene, track = key.split("/")
    ids = track_frame_ids(video, scene, track)
    try:
        return ids.index(frame_id)
    except ValueError:
        return None


_DERIVED_SUFFIXES = ("_skeleton", "_canny", "_pose_body", "_pose_racket")
