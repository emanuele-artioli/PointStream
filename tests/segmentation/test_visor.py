from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.visor import b1
from src.segmentation import visor
from src.segmentation.evaluate import compare
from src.segmentation.masks import ClipMasks

SHAPE = (108, 192)  # a tenth of the 1080p frame keeps the tests fast


def square(x: float, y: float, size: float) -> list[list[list[float]]]:
    return [[[x, y], [x + size, y], [x + size, y + size], [x, y + size]]]


def entry(number: int, annotations: list[dict[str, Any]], run: str | None = None) -> dict[str, Any]:
    image: dict[str, Any] = {"name": f"P99_01_frame_{number:010d}.png", "video": "P99_01"}
    if run is not None:
        image["interpolation"] = run
    return {"image": image, "annotations": annotations}


def mask(name: str, segments: list[Any], kind: int = 0) -> dict[str, Any]:
    return {"id": f"{name}-{kind}", "name": name, "type": kind, "segments": segments}


def dense_doc() -> dict[str, Any]:
    """Two abutting runs (10-12, 12-13) sharing frame 12, then a gap, then 20-21."""
    hand = square(100, 100, 200)  # 854x480 canvas
    cup = square(500, 300, 100)
    entries = [
        entry(10, [mask("left hand", hand, 1), mask("cup", cup, 1)], "r1"),
        entry(11, [mask("left hand", square(110, 100, 200)), mask("cup", cup)], "r1"),
        entry(12, [mask("left hand", hand, 1), mask("cup", cup, 1)], "r1"),
        entry(12, [mask("left hand", hand, 1), mask("cup", cup, 1)], "r2"),
        entry(13, [mask("right glove", square(600, 50, 120), 1)], "r2"),
        entry(20, [mask("knife", cup, 1)], "r3"),
        entry(21, [mask("knife", cup, 1)], "r3"),
    ]
    return {"info": {"details": "854x480"}, "video_annotations": entries}


def test_frame_numbering_rules() -> None:
    assert visor.frame_number("P01_01_frame_0000000965.jpg") == 965
    assert visor.visor_frame_to_video_index(1) == 0
    with pytest.raises(ValueError):
        visor.visor_frame_to_video_index(0)
    # EPIC rgb frames were extracted at the nominal integer rate: 60 for 59.94 fps.
    assert visor.epic_frame_to_video_index(1503, 60000 / 1001) == 1500
    assert visor.epic_frame_to_video_index(1503, 50.0) == 1502


def test_alignment_is_exact_without_drift_and_interpolated_with_it() -> None:
    doc = dense_doc()
    # Keyframes 10, 12, 13 and 20, 21 (VISOR) map to the video with no drift:
    exact = visor.frame_alignment(doc, {10: 9, 12: 11, 13: 12, 20: 19, 21: 20})
    assert [exact[n]["index"] for n in (10, 11, 12, 13)] == [9, 10, 11, 12]
    assert all(exact[n]["exact"] for n in exact)
    clip = visor.clip_masks(doc, 10, 4, fps=50.0, alignment=exact, shape=SHAPE)
    assert clip.meta["video_indices"] == [9, 10, 11, 12] and clip.meta["aligned_exactly"]

    # VISOR 10 -> 12 spans 2 frames but the video 4: frame 11 is placed halfway, inexactly.
    drifting = visor.frame_alignment(doc, {10: 9, 12: 13, 13: 14})
    assert drifting[11] == {"index": 11, "drift": 2, "exact": 0}
    assert drifting[12]["exact"] and 20 not in drifting  # the second run has no anchors
    clip = visor.clip_masks(doc, 10, 4, fps=50.0, alignment=drifting, shape=SHAPE)
    assert not clip.meta["aligned_exactly"] and clip.meta["max_abs_drift"] == 2
    assert clip.meta["drift"] == [0, 2, 0, 0] and clip.labelled_frames() == [0, 1, 2, 3]
    # For evaluation, the inexact frame keeps its masks but is never scored.
    scored = visor.clip_masks(doc, 10, 4, fps=50.0, alignment=drifting, exact_only=True, shape=SHAPE)
    assert scored.labelled_frames() == [0, 2, 3] and scored.frames[1]
    assert compare(scored, scored)["frames"] == 3

    mapping = {"P99_01_frame_0000000010.jpg": "frame_0000000010.jpg",
               "P99_01_frame_0000000012.jpg": "frame_0000000013.jpg"}
    assert visor.keyframe_anchors(doc, mapping, 60000 / 1001) == {10: 9, 12: 12}


@pytest.mark.parametrize("annotation,expected", [
    ({"name": "left hand"}, "left hand"),
    ({"name": "right glove"}, "right hand"),
    ({"name": "glove", "on_which_hand": "left hand"}, "left hand"),
    ({"name": "glove"}, "active object"),
    ({"name": "hand blender"}, "active object"),
])
def test_native_classes(annotation: dict[str, Any], expected: str) -> None:
    assert visor.native_class(annotation) == expected


def test_shared_boundary_frames_merge_and_runs_split_at_gaps() -> None:
    frames = visor.frames(dense_doc())
    assert sorted(frames) == [10, 11, 12, 13, 20, 21]
    assert len(frames[12].annotations) == 2 and frames[12].interpolations == ("r1", "r2")
    assert frames[12].keyframe and not frames[11].keyframe
    assert visor.runs(frames) == [(10, 13), (20, 21)]


def test_dense_polygons_are_scaled_from_their_480p_canvas() -> None:
    full = visor.polygon_mask(square(100, 100, 200), source_size=visor.DENSE_SIZE, shape=SHAPE)
    ys, xs = np.nonzero(full)
    # 100..300 on an 854x480 canvas is 22.5..67.4 by 22.5..67.5 on a 192x108 frame.
    assert abs(xs.min() - 22) <= 1 and abs(xs.max() - 67) <= 1
    assert abs(ys.min() - 22) <= 1 and abs(ys.max() - 68) <= 1


def test_clip_masks_flag_labelled_frames_and_record_provenance(tmp_path: Path) -> None:
    clip = visor.clip_masks(dense_doc(), 11, 11, fps=50.0, shape=SHAPE)  # frames 11..21
    assert clip.classes == visor.CLASSES and len(clip) == 11
    assert clip.labelled_frames() == [0, 1, 2, 9, 10]
    assert clip.frames[5] == []  # unlabelled, not empty: never scored
    assert clip.meta["keyframes"] == [1, 2, 9, 10] and clip.meta["first_visor_frame"] == 11
    assert "video_indices" not in clip.meta  # placed on the video only with an alignment
    instances = [inst for frame in clip.frames for inst in frame]
    assert {inst.provenance for inst in instances} == {visor.INTERPOLATED}
    glove = next(inst for inst in instances if inst.label == "right glove")
    assert glove.class_name == "right hand"
    cups = {inst.track_id for inst in instances if inst.label == "cup"}
    assert len(cups) == 1  # stable within the clip

    loaded = ClipMasks.load(clip.save(tmp_path / "item"))
    assert loaded.labelled == clip.labelled
    assert loaded.frames[0][1].provenance == visor.INTERPOLATED and loaded.frames[0][1].label == "cup"
    assert b1.mask_digest(loaded) == b1.mask_digest(clip)


def test_sparse_masks_are_human() -> None:
    sparse = {"video_annotations": [
        {"image": {"name": "P99_01_frame_0000000010.jpg", "video": "P99_01"},
         "annotations": [{"name": "left hand", "segments": square(225, 225, 450)}]},
    ]}
    clip = visor.clip_masks(sparse, 10, 1, fps=50.0, shape=SHAPE)
    assert clip.meta["source"] == "sparse"
    assert clip.frames[0][0].provenance == visor.HUMAN


def test_interpolation_zip_loads_like_its_json() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("P99_01_interpolations.json", json.dumps(dense_doc()))
    assert visor.load_annotations(buffer.getvalue()) == dense_doc()
    with pytest.raises(ValueError):
        visor.load_annotations(b"{}")


def test_compare_scores_only_labelled_reference_frames() -> None:
    reference = visor.clip_masks(dense_doc(), 11, 11, fps=50.0, shape=SHAPE)
    perfect = ClipMasks(reference.classes, *SHAPE, 50.0, frames=[list(f) for f in reference.frames])
    perfect.frames[5] = list(reference.frames[0])  # wrong only on an unlabelled frame
    scores = compare(perfect, reference)
    assert scores["frames"] == 5 and scores["scopes"]["foreground"]["J"] == 1.0


def test_hold_first_is_perfect_on_a_static_clip_and_imperfect_on_motion() -> None:
    static = visor.clip_masks(dense_doc(), 20, 2, fps=50.0, shape=SHAPE)
    assert compare(b1.hold_first(static), static)["scopes"]["foreground"]["J"] == 1.0
    moving = visor.clip_masks(dense_doc(), 10, 4, fps=50.0, shape=SHAPE)
    assert compare(b1.hold_first(moving), moving)["scopes"]["foreground"]["J"] < 1.0


def test_rule_summary_counts_holds_and_offsets() -> None:
    rows: list[dict[str, Any]] = [
        {"rule_holds": {"visor_minus_one": True, "epic_time": True, "epic_minus_one": False},
         "rule_offset": {"visor_minus_one": 0, "epic_time": 0, "epic_minus_one": 2},
         "rule_mae": {"visor_minus_one": 1.0, "epic_time": 1.1, "epic_minus_one": 7.0}, "best_mae": 1.0},
        {"rule_holds": {"visor_minus_one": True, "epic_time": False, "epic_minus_one": False},
         "rule_offset": {"visor_minus_one": 1, "epic_time": -1, "epic_minus_one": 3},
         "rule_mae": {"visor_minus_one": 1.2, "epic_time": 6.0, "epic_minus_one": None}, "best_mae": 1.0},
    ]
    for row in rows:  # visor_time agrees with visor_minus_one at 50 fps
        for key in ("rule_holds", "rule_offset", "rule_mae"):
            row[key]["visor_time"] = row[key]["visor_minus_one"]
    summary = b1.summarize_rules(rows)
    assert set(summary) == set(b1.RULES)
    assert summary["visor_minus_one"]["all_hold"] and summary["visor_minus_one"]["offset_to_best"] == {"0": 1, "1": 1}
    assert not summary["epic_time"]["all_hold"] and summary["epic_time"]["max_mae_excess"] == 5.0
    assert summary["epic_minus_one"]["holds"] == 0
    assert b1.spaced(list("abcdefghij"), 3) == ["a", "e", "j"]
    assert b1.spaced(list("abc"), 1) == ["c"]


def test_convert_item_and_validator(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    (archive / "dense").mkdir(parents=True)
    (archive / "annotations").mkdir()
    doc = dense_doc()
    (archive / "dense" / "P99_01_interpolations.json").write_text(json.dumps(doc))
    sparse = {"video_annotations": [{"image": {"name": "P99_01_frame_0000000010.jpg", "video": "P99_01"},
                                     "annotations": [{"name": "left hand", "segments": square(225, 225, 450)},
                                                     {"name": "cup", "segments": square(1124, 675, 225)}]}]}
    (archive / "annotations" / "P99_01.json").write_text(json.dumps(sparse))
    mapping = {f"P99_01_frame_{n:010d}.jpg": f"frame_{n:010d}.jpg" for n in (10, 12, 13, 20, 21)}
    (archive / "frame_mapping.json").write_text(json.dumps({"P99_01": mapping}))
    item: dict[str, Any] = {"id": "P99_01_0000000010", "video": "P99_01", "first_visor_frame": 10, "frames": 4, "fps": 50.0,
            "first_video_index": 9,
            "dense_member": "dense/P99_01_interpolations.json", "video_file": {"sha256": "0" * 64},
            "dense_sha256": b1.file_sha256(archive / "dense" / "P99_01_interpolations.json")}
    row = b1.convert_item(item, str(archive), 4, str(tmp_path / "publish"))
    assert row["dense_sha256_matches"] and row["first_is_keyframe"] and row["labelled"] == 4
    assert row["aligned_exactly"] and row["video_indices"] == [9, 12]
    agreement = row["human_agreement_first_frame"]
    assert agreement["left hand"] > 0.95 and agreement["objects"]["cup"] > 0.95
    assert (tmp_path / "publish" / "masks" / item["id"] / "masks.rle").is_file()

    stage = tmp_path / "stage"
    b1.write_json(stage / "evalset.json", {
        "frames_per_item": 4, "items": [row],
        "dense_vs_human_first_frame_iou": {"hands": {"n": 1, "median": agreement["left hand"]}},
    })
    checks = b1.validate_evalset(stage)
    assert all(checks.values()), checks
