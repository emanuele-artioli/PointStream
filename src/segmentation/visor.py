"""EPIC-KITCHENS VISOR labels as `ClipMasks`, and the video frames they label.

VISOR releases polygons in two files per video (docs/resources.md#datasets):

* sparse ``GroundTruth-SparseAnnotations/annotations/<split>/<video>.json``:
  human-drawn, in 1920x1080 coordinates, about one frame every 78;
* dense ``Interpolations-DenseAnnotations/<split>/<video>_interpolations.zip``:
  one JSON of runs between two sparse frames, drawn in **854x480** coordinates
  (the file's own ``info``). ``type`` 1 marks a run's start and end frames, the
  sparse labels filtered to the entities present at both ends and redrawn at
  480p; ``type`` 0 marks the masks interpolated between them.

Every mask here records a provenance tier: ``human`` for the sparse file,
``interpolated`` for everything in the dense file (its keyframes too, because
they are 480p redraws). Polygons are scaled to the 1080p frame and filled there.

Classes are VISOR's own: ``left hand`` and ``right hand`` (each with the visible
forearm), and ``active object`` for every other mask, whose open-vocabulary name
is kept as the instance ``label``. A glove worn on a hand is that hand.

Frames are named ``<video>_frame_<n>``, numbered by VISOR's own frame
extraction, which is not the decoded video's numbering
(docs/resources.md#datasets). ``frame_mapping.json`` maps each *sparse* frame to
an EPIC-KITCHENS rgb frame, and an rgb frame maps to the video by time
(`epic_frame_to_video_index`). Dense frames have no mapping of their own:
`frame_alignment` places them between the keyframes of their run, and records
how far VISOR's numbering drifts from the video's there. Where it never drifts,
as on the EPIC-KITCHENS-100 (50 fps) videos, VISOR frame ``n`` is decoded frame
``n - 1`` (`visor_frame_to_video_index`).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.segmentation.masks import ClipMasks, Instance, encode_rle

CLASSES = ("left hand", "right hand", "active object")
HANDS = ("left hand", "right hand")
HUMAN = "human"
INTERPOLATED = "interpolated"
#: (height, width) of the frames and of the sparse polygons.
FRAME_SIZE = (1080, 1920)
#: (height, width) the dense polygons are drawn in.
DENSE_SIZE = (480, 854)
_FRAME_RE = re.compile(r"(?:^|_)frame_(\d+)\.\w+$")
_GLOVES = {"left glove": "left hand", "right glove": "right hand"}


def frame_number(name: str) -> int:
    """``n`` of a frame name: VISOR ``P01_01_frame_0000000965.jpg`` or EPIC ``frame_0000000966.jpg``."""
    match = _FRAME_RE.search(name)
    if match is None:
        raise ValueError(f"not a VISOR frame name: {name!r}")
    return int(match.group(1))


def visor_frame_to_video_index(number: int) -> int:
    """Decoded-frame index (0-based) of VISOR frame ``number`` where VISOR never drifts.

    True on the EPIC-KITCHENS-100 (50 fps) videos; elsewhere use `frame_alignment`.
    """
    if number < 1:
        raise ValueError(f"VISOR frames are 1-indexed, got {number}")
    return number - 1


def epic_frame_to_video_index(epic_frame: int, fps: float) -> int:
    """Decoded-frame index of EPIC-KITCHENS rgb frame ``epic_frame`` (1-indexed).

    The rgb frames were extracted at the nominal integer rate, so on a 59.94 fps
    video rgb frame k shows video time (k - 1) / 60 s; at 50 fps it is frame k - 1.
    """
    return int(round((epic_frame - 1) * fps / round(fps)))


def keyframe_anchors(doc: dict[str, Any], mapping: dict[str, str], fps: float) -> dict[int, int]:
    """Decoded-frame index of every labelled frame that ``frame_mapping.json`` maps.

    ``mapping`` is the release's entry for this video (VISOR name -> EPIC name).
    """
    epic = {frame_number(visor_name): frame_number(name) for visor_name, name in mapping.items()}
    return {
        number: epic_frame_to_video_index(epic[number], fps) for number in frames(doc) if number in epic
    }


def frame_alignment(doc: dict[str, Any], anchors: dict[int, int]) -> dict[int, dict[str, int]]:
    """Decoded-frame index of each labelled frame, placed between its run's anchors.

    Within a run, a frame between anchors ``a < b`` (VISOR numbers) gets
    ``anchors[a] + round((n - a) * span_video / span_visor)``; ``drift`` is
    ``span_video - span_visor``, the frames VISOR's numbering gained or lost on
    the video there. A frame is exact when it is an anchor or its drift is 0.
    Frames of a run outside its first and last anchor are left out.
    """
    out: dict[int, dict[str, int]] = {}
    for first, last in runs(frames(doc)):
        keys = sorted(n for n in anchors if first <= n <= last)
        for a, b in zip(keys, keys[1:]):
            video_span, visor_span = anchors[b] - anchors[a], b - a
            drift = video_span - visor_span
            for number in range(a, b + 1):
                index = anchors[a] + int(round((number - a) * video_span / visor_span))
                exact = drift == 0 or number in (a, b)
                previous = out.get(number)
                if previous is None or abs(drift) < abs(previous["drift"]):
                    out[number] = {"index": index, "drift": 0 if number in (a, b) else drift, "exact": int(exact)}
    return out


def native_class(annotation: dict[str, Any]) -> str:
    """VISOR class of one annotation: a hand (gloves on a hand included) or an active object."""
    name = str(annotation["name"])
    if name in HANDS:
        return name
    if name in _GLOVES:
        return _GLOVES[name]
    if "glove" in name and annotation.get("on_which_hand") in HANDS:
        return str(annotation["on_which_hand"])
    return "active object"


def load_annotations(source: Path | str | bytes) -> dict[str, Any]:
    """A VISOR annotation document from a ``.json``, a dense ``_interpolations.zip``, or bytes."""
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [n for n in archive.namelist() if n.endswith(".json")]
            if len(members) != 1:
                raise ValueError(f"expected one JSON in the interpolation zip, found {members}")
            data = archive.read(members[0])
    doc = json.loads(data)
    if "video_annotations" not in doc:
        raise ValueError("not a VISOR annotation document (no video_annotations)")
    return doc


def is_dense(doc: dict[str, Any]) -> bool:
    entries = doc["video_annotations"]
    return bool(entries) and "interpolation" in entries[0]["image"]


@dataclass(frozen=True)
class Frame:
    """The annotations of one VISOR frame, merged across the runs that share it."""

    number: int
    annotations: tuple[dict[str, Any], ...]
    interpolations: tuple[str, ...]

    @property
    def keyframe(self) -> bool:
        """A dense run's start or end (a sparse frame redrawn at 480p)."""
        return any(int(a.get("type", 1)) == 1 for a in self.annotations)


def frames(doc: dict[str, Any]) -> dict[int, Frame]:
    """Every labelled frame by VISOR number.

    Adjacent dense runs share their boundary frame and repeat its masks under new
    ids; a repeat (same name, same polygons) is kept once.
    """
    merged: dict[int, tuple[list[dict[str, Any]], list[str]]] = {}
    for entry in doc["video_annotations"]:
        number = frame_number(entry["image"]["name"])
        annotations, runs = merged.setdefault(number, ([], []))
        run = entry["image"].get("interpolation")
        if run is not None and run not in runs:
            runs.append(run)
        seen = {(a["name"], json.dumps(a["segments"])) for a in annotations}
        for annotation in entry["annotations"]:
            if (annotation["name"], json.dumps(annotation["segments"])) not in seen:
                annotations.append(annotation)
    return {
        number: Frame(number, tuple(annotations), tuple(runs))
        for number, (annotations, runs) in sorted(merged.items())
    }


def runs(numbers: Iterable[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive frame numbers, as inclusive ``(first, last)``."""
    out: list[tuple[int, int]] = []
    for number in sorted(set(numbers)):
        if out and number == out[-1][1] + 1:
            out[-1] = (out[-1][0], number)
        else:
            out.append((number, number))
    return out


def polygon_mask(
    segments: Sequence[Sequence[Sequence[float]]],
    *,
    source_size: tuple[int, int] = FRAME_SIZE,
    shape: tuple[int, int] = FRAME_SIZE,
) -> np.ndarray:
    """Fill VISOR polygons drawn on a ``source_size`` (h, w) canvas into a ``shape`` mask.

    Each polygon is scaled to ``shape``, rounded to pixel centres and filled with
    OpenCV (8-connected edges included); several polygons form their union.
    """
    import cv2

    scale = np.array([shape[1] / source_size[1], shape[0] / source_size[0]])
    canvas = np.zeros(shape, dtype=np.uint8)
    polygons = [
        np.round(np.asarray(polygon, dtype=np.float64) * scale).astype(np.int32)
        for polygon in segments
        if len(polygon) >= 3
    ]
    if polygons:
        cv2.fillPoly(canvas, polygons, (1,))
    return canvas > 0


def frame_instances(
    frame: Frame,
    *,
    dense: bool,
    tracks: dict[tuple[str, str, int], int],
    shape: tuple[int, int] = FRAME_SIZE,
) -> list[Instance]:
    """Instances of one frame. ``tracks`` maps (class, label, repeat) to a clip-stable id."""
    provenance = INTERPOLATED if dense else HUMAN
    source_size = DENSE_SIZE if dense else FRAME_SIZE
    repeats: dict[tuple[str, str], int] = {}
    out = []
    for annotation in frame.annotations:
        class_name, label = native_class(annotation), str(annotation["name"])
        repeat = repeats[(class_name, label)] = repeats.get((class_name, label), -1) + 1
        key = (class_name, label, repeat)
        track = tracks.setdefault(key, len(tracks))
        mask = polygon_mask(annotation["segments"], source_size=source_size, shape=shape)
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        out.append(Instance(class_name, track, 1.0, encode_rle(mask), bbox, provenance, label))
    return out


def clip_masks(
    doc: dict[str, Any],
    first: int,
    count: int,
    *,
    fps: float,
    alignment: dict[int, dict[str, int]] | None = None,
    shape: tuple[int, int] = FRAME_SIZE,
    meta: dict[str, Any] | None = None,
) -> ClipMasks:
    """Masks of VISOR frames ``first .. first + count - 1`` as one clip.

    Clip frame ``i`` is VISOR frame ``first + i``. With ``alignment``
    (`frame_alignment`), ``meta["video_indices"]`` names the decoded frame each
    clip frame labels (None where unknown) and ``meta["aligned_exactly"]`` says
    whether every frame is exact and the indices are consecutive. Frames the
    file does not label are flagged unlabelled and hold no instances.
    """
    if count <= 0:
        raise ValueError("a clip needs at least one frame")
    dense = is_dense(doc)
    labelled = frames(doc)
    clip = ClipMasks(CLASSES, shape[0], shape[1], float(fps), labelled=[])
    clip.ensure_frames(count)
    tracks: dict[tuple[str, str, int], int] = {}
    keyframes: list[int] = []
    interpolations: list[str] = []
    assert clip.labelled is not None
    for index in range(count):
        frame = labelled.get(first + index)
        if frame is None:
            continue
        clip.labelled[index] = True
        clip.frames[index] = frame_instances(frame, dense=dense, tracks=tracks, shape=shape)
        if dense and frame.keyframe:
            keyframes.append(index)
        interpolations.extend(run for run in frame.interpolations if run not in interpolations)
    video = doc["video_annotations"][0]["image"]["video"] if doc["video_annotations"] else None
    clip.meta.update(
        {
            "dataset": "EPIC-KITCHENS VISOR",
            "video": video,
            "source": "dense" if dense else "sparse",
            "polygon_canvas": list(DENSE_SIZE if dense else FRAME_SIZE),
            "first_visor_frame": first,
            "keyframes": keyframes,
            "interpolations": interpolations,
            "tracks": [
                {"track_id": track, "class_name": key[0], "label": key[1]}
                for key, track in sorted(tracks.items(), key=lambda item: item[1])
            ],
            **(meta or {}),
        }
    )
    if alignment is not None:
        placed = [alignment.get(first + index) for index in range(count)]
        indices = [None if p is None else p["index"] for p in placed]
        clip.meta["video_indices"] = indices
        clip.meta["max_abs_drift"] = max((abs(p["drift"]) for p in placed if p), default=None)
        clip.meta["aligned_exactly"] = all(p is not None and p["exact"] for p in placed) and all(
            b == a + 1 for a, b in zip(indices, indices[1:])  # type: ignore[operator]
        )
    return clip


def decode_frames(
    video: Path | str, indices: Iterable[int], *, margin: int = 0
) -> Iterator[tuple[int, np.ndarray]]:
    """``(index, RGB frame)`` for the decoded frames at ``indices`` (± ``margin``).

    The index of a decoded frame is its presentation time times the stream's
    nominal rate (``r_frame_rate``), so it counts frames from the first one on a
    constant-rate stream. Each contiguous group of wanted indices is reached by
    one seek to the preceding keyframe and decoded forward.
    """
    import av

    wanted = sorted({i + d for i in indices for d in range(-margin, margin + 1) if i + d >= 0})
    groups = runs(wanted)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.base_rate or stream.average_rate
        if rate is None or stream.time_base is None:
            raise RuntimeError(f"{video}: stream has no frame rate or time base")
        base, start = stream.time_base, stream.start_time or 0
        for first, last in groups:
            target = int(start + max(first - 2, 0) / rate / base)
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                at = int(round(float((frame.pts - start) * base * rate)))
                if at > last:
                    break
                if at >= first:
                    yield at, frame.to_ndarray(format="rgb24")


def video_info(video: Path | str) -> dict[str, Any]:
    """Codec, size and both frame rates of a video's first stream."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        return {
            "codec": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "r_frame_rate": str(stream.base_rate),
            "avg_frame_rate": str(stream.average_rate),
            "fps": float(stream.base_rate or stream.average_rate or 0.0),
            "frames_declared": stream.frames,
        }


__all__ = [
    "CLASSES",
    "DENSE_SIZE",
    "FRAME_SIZE",
    "HUMAN",
    "INTERPOLATED",
    "Frame",
    "clip_masks",
    "decode_frames",
    "epic_frame_to_video_index",
    "frame_alignment",
    "frame_instances",
    "frame_number",
    "frames",
    "is_dense",
    "keyframe_anchors",
    "load_annotations",
    "native_class",
    "polygon_mask",
    "runs",
    "video_info",
    "visor_frame_to_video_index",
]
