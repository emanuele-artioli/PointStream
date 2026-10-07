"""B1b's plan, held-out scoring, fill and checks, with a stand-in tracker (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.visor import b1b
from src.segmentation import hand_objects, visor
from src.segmentation.masks import ClipMasks

SHAPE = (108, 192)  # a tenth of 1080p keeps the tests fast


def square(x: float, y: float, size: float) -> list[list[list[float]]]:
    return [[[x, y], [x + size, y], [x + size, y + size], [x, y + size]]]


def ann(name: str, segments: list[Any], kind: int = 1) -> dict[str, Any]:
    return {"id": name, "name": name, "type": kind, "segments": segments}


def entry(number: int, annotations: list[dict[str, Any]], run: str | None = None,
          bounds: tuple[int, int] | None = None) -> dict[str, Any]:
    image: dict[str, Any] = {"name": f"P99_101_frame_{number:010d}.jpg", "video": "P99_101"}
    if run is not None:
        image["interpolation"] = run
    if bounds is not None:
        image["interpolation_start_frame"] = f"P99_101_frame_{bounds[0]:010d}.jpg"
        image["interpolation_end_frame"] = f"P99_101_frame_{bounds[1]:010d}.jpg"
    return {"image": image, "annotations": annotations}


HAND = square(200, 300, 400)  # 1080p coordinates
CUP = square(1200, 600, 200)
KNIFE = square(900, 100, 150)


def sparse_doc() -> dict[str, Any]:
    """Keyframes 10, 20, 30: the hand everywhere, the cup at 10 and 30, a knife at 20."""
    return {"video_annotations": [
        entry(10, [ann("left hand", HAND), ann("cup", CUP)]),
        entry(20, [ann("left hand", HAND), ann("knife", KNIFE)]),
        entry(30, [ann("left hand", HAND), ann("cup", CUP)]),
    ]}


def dense_doc() -> dict[str, Any]:
    """Runs 10-20 and 20-30 at 854x480; the cup is dropped on 13-17."""
    scale = 854 / 1920
    hand = [[[x * scale, y * scale] for x, y in poly] for poly in HAND]
    cup = [[[x * scale, y * scale] for x, y in poly] for poly in CUP]
    entries = []
    for number in range(10, 31):
        run, bounds = ("r1", (10, 20)) if number <= 20 else ("r2", (20, 30))
        annotations = [ann("left hand", hand, int(number in (10, 20, 30)))]
        if not 13 <= number <= 17 and number not in range(20, 30):
            annotations.append(ann("cup", cup, int(number in (10, 30))))
        entries.append(entry(number, annotations, run, bounds))
    entries.append(entry(20, [ann("left hand", hand)], "r2", (20, 30)))
    return {"info": {}, "video_annotations": entries}


def mapping() -> dict[str, str]:
    return {f"P99_101_frame_{n:010d}.jpg": f"frame_{n:010d}.jpg" for n in (10, 20, 30)}


ITEM: dict[str, Any] = {"id": "P99_101_0000000013", "video": "P99_101", "fps": 50.0, "first_visor_frame": 13,
        "first_video_index": 12, "frames": 10, "run": {"first": 10, "last": 30}, "sparse_jpegs": []}


class HoldTracker:
    """Each frame gets the masks of the latest prompted frame at or before it."""

    def load_frames(self, frames_dir: Path) -> tuple[list[int], int, int]:
        return list(range(len(list(Path(frames_dir).glob("*.png"))))), *SHAPE

    def track(self, images: Any, height: int, width: int, prompts: dict[int, dict[int, np.ndarray]],
              start: int = 0, frames: int | None = None, reverse: bool = False) -> Any:
        order = range(start, -1, -1) if reverse else range(start, len(images))
        for index in order:
            source = min(p for p in prompts if p >= index) if reverse else max(p for p in prompts if p <= index)
            yield index, {o: (m.copy(), 0.9) for o, m in prompts[source].items()}

    def peak_gpu_mib(self) -> float:
        return 0.0


def detections() -> list[hand_objects.FrameDetections]:
    box = hand_objects.Box(250 / 1920, 350 / 1080, 550 / 1920, 650 / 1080)
    hand = hand_objects.Hand(box, 0.95, "left hand", "portable_object", (0.0, 0.0))
    return [hand_objects.FrameDetections("P99_101", k, (hand,), ()) for k in range(1, 41)]


def test_plan_bounds_the_window_with_keyframes() -> None:
    plan = b1b.plan_item(ITEM, sparse_doc(), mapping(), shape=SHAPE)
    assert (plan.span_start, plan.span_end) == (9, 29)
    assert plan.keyframes == {10: 0, 20: 10, 30: 20}
    assert plan.window == (3, 12)
    assert [(o.class_name, o.label) for o in plan.objects] == [("left hand", "left hand"), ("active object", "cup"),
                                                              ("active object", "knife")]
    assert sorted(plan.human[10]) == [1, 3]  # at keyframe 20: the hand and the knife, not the cup
    assert plan.pairs() == [(0, 10), (10, 20)]


def test_window_without_bounding_keyframe_is_an_error() -> None:
    item = {**ITEM, "first_video_index": 25, "frames": 10}
    with pytest.raises(ValueError, match="bounds the window"):
        b1b.plan_item(item, sparse_doc(), mapping(), shape=SHAPE)


@pytest.mark.parametrize("fps", [50.0, 60000 / 1001])
def test_video_to_epic_frame_inverts_the_reader_rule(fps: float) -> None:
    for index in range(0, 5000, 7):
        epic = b1b.video_to_epic_frame(index, fps)
        if epic is not None:
            assert visor.epic_frame_to_video_index(epic, fps) == index
    assert sum(b1b.video_to_epic_frame(i, fps) is None for i in range(5000)) <= 5


def test_hand_agreement() -> None:
    mask = np.zeros((1080, 1920), bool)
    mask[300:700, 200:600] = True
    hand = detections()[0].hands[0]
    assert b1b.hand_agreement(mask, "left hand", [hand]) == "agrees"
    right = hand_objects.Hand(hand.box, 0.9, "right hand", "no_contact", (0.0, 0.0))
    assert b1b.hand_agreement(mask, "left hand", [right]) == "other_side"
    assert b1b.hand_agreement(mask, "left hand", []) == "no_detection"
    assert b1b.hand_agreement(np.zeros_like(mask), "left hand", [hand]) == "empty"


def test_item_end_to_end(tmp_path: Path) -> None:
    import cv2

    plan = b1b.plan_item(ITEM, sparse_doc(), mapping(), shape=SHAPE)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(plan.length):
        cv2.imwrite(str(frames_dir / f"{i}.png"), np.full((*SHAPE, 3), 90, np.uint8))
    dense = dense_doc()
    anchors = {10: 9, 20: 19, 30: 29}
    clip = visor.clip_masks(dense, 13, 10, fps=50.0, alignment=visor.frame_alignment(dense, anchors), shape=SHAPE)
    row = b1b.process_item(HoldTracker(), plan, frames_dir, dense, sparse_doc(), clip, detections(),
                           tmp_path / "publish", review=2, profile=False)
    # Held out: the hand holds still, so SAM (holding) and the floor both score 1 at b.
    hands = [r for r in row["validation"] if r["group"] == "hands"]
    assert [r["J"] for r in hands] == [1.0, 1.0] and all(r["hold_J"] == 1.0 for r in hands)
    cup_a = next(r for r in row["validation"] if r["label"] == "cup" and r["a"] == 0)
    assert cup_a["labelled_b"] is False and cup_a["released"] is False  # holding never lets the cup go
    # Fill: each frame from its nearer keyframe. VISOR 13-15 are nearer keyframe 10 (hand, cup): the
    # cup the dense masks drop there is added. VISOR 16-22 are nearer keyframe 20 (hand, knife): the
    # knife the dense masks never have is added, and the cup is not (unlabelled at 20).
    filled = ClipMasks.load(tmp_path / "publish" / "masks" / ITEM["id"] / "masks.rle")
    added = {i: [inst.label for inst in f if inst.provenance == b1b.SAM_TIER] for i, f in enumerate(filled.frames)}
    assert added == {**{i: ["cup"] for i in range(3)}, **{i: ["knife"] for i in range(3, 10)}}
    assert row["fill"]["instances_added"] == {"active object": 10}
    assert all(inst.track_id > b1b.FILL_TRACK_BASE for f in filled.frames for inst in f if inst.provenance == b1b.SAM_TIER)
    assert row["frames_tracked"] == {"forward": 22, "backward": 10}  # gaps share their keyframe
    before, after = row["missing_objects"]["visor_dense"], row["missing_objects"]["visor_dense_sam_fill"]
    assert after["mean_share"] < before["mean_share"]
    assert row["prompt_reproduced_iou"]["median"] == 1.0
    assert row["hand_check"]["human_keyframe"] == {"agrees": 3}
    assert sum(row["hand_check"]["sam_heldout"].values()) == sum(row["hand_check"]["dense_same_frames"].values())
    assert len(row["review"]) == 2 and all((tmp_path / "publish" / r["file"]).is_file() for r in row["review"])
    assert all(r["window_frame"] != 7 for r in row["review"])  # span index 10 is a keyframe


def test_review_frames_are_stable_and_skip_keyframes() -> None:
    plan = b1b.plan_item(ITEM, sparse_doc(), mapping(), shape=SHAPE)
    picks = b1b.review_frames(plan, 3)
    assert picks == b1b.review_frames(plan, 3) and len(picks) == 3
    assert all(3 <= p <= 12 and p != 10 for p in picks)


def test_decision_rule() -> None:
    good = {"n": 40, "mean_J": 0.8, "median_J": 0.85, "hold_mean_J": 0.5, "hard_n": 12, "hard_mean_J": 0.7,
            "released_n": 6, "released_share": 0.8}
    weak = {**good, "mean_J": 0.55, "median_J": 0.6}
    decision = b1b.decide({"hands": good, "objects": weak}, {"sam_heldout": 0.6, "dense_same_frames": 0.62})
    assert decision["hands"]["adopt"] and not decision["objects"]["adopt"]
    decision = b1b.decide({"hands": good, "objects": good}, {"sam_heldout": 0.4, "dense_same_frames": 0.62})
    assert not decision["hands"]["adopt"] and decision["hands"]["checks"]["detector_agreement"] is False


def test_validator_on_a_written_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    plan = b1b.plan_item(ITEM, sparse_doc(), mapping(), shape=SHAPE)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(plan.length):
        cv2.imwrite(str(frames_dir / f"{i}.png"), np.zeros((*SHAPE, 3), np.uint8))
    dense = dense_doc()
    clip = visor.clip_masks(dense, 13, 10, fps=50.0, alignment=visor.frame_alignment(dense, {10: 9, 20: 19, 30: 29}),
                            shape=SHAPE)
    row = b1b.process_item(HoldTracker(), plan, frames_dir, dense, sparse_doc(), clip, detections(),
                           tmp_path / "publish", review=2, profile=False)
    row.update({"decode": {"jpeg_gate": [{"holds": True}]}, "dense_record_mask_sha256": row["dense_mask_sha256"],
                "hand_objects": {"sha256": "x" * 64, "manifest_sha256": "x" * 64},
                "kernels": {"kernel_launches": 10, "attention_family": ["flash"]}})
    result: dict[str, Any] = {"items": [row], "settings": {"review": 2},
              "validation_summary": b1b.summarize_validation(row["validation"]),
              "sam": {"runtime": {"gpu": "NVIDIA RTX A6000", "capability": [8, 6]}, "missing_keys": [],
                      "unexpected_keys": [], "loaded_keys": 5, "model_keys": 5,
                      "checkpoint_sha256": "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"}}
    (tmp_path / "b1b.json").write_text(json.dumps(result, default=str))
    checks = b1b.validate_result(tmp_path)
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    result["sam"]["runtime"]["gpu"] = "Quadro RTX 8000"
    (tmp_path / "b1b.json").write_text(json.dumps(result, default=str))
    assert not b1b.validate_result(tmp_path)["sam_on_ada_or_a6000"]


def test_checkpoint_remap_keeps_tracker_and_vision_backbone() -> None:
    from src.segmentation.sam31_tracker import remap_checkpoint

    renamed, skipped = remap_checkpoint({
        "tracker.model.sam_mask_decoder.w": 1, "detector.backbone.vision_backbone.trunk.w": 2,
        "detector.backbone.language_backbone.w": 3, "detector.transformer.w": 4,
    })
    assert renamed == {"sam_mask_decoder.w": 1, "backbone.vision_backbone.trunk.w": 2}
    assert sorted(skipped) == ["detector.backbone.language_backbone.w", "detector.transformer.w"]


def test_merge_keeps_only_adopted_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    plan = b1b.plan_item(ITEM, sparse_doc(), mapping(), shape=SHAPE)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(plan.length):
        cv2.imwrite(str(frames_dir / f"{i}.png"), np.zeros((*SHAPE, 3), np.uint8))
    dense = dense_doc()
    clip = visor.clip_masks(dense, 13, 10, fps=50.0, alignment=visor.frame_alignment(dense, {10: 9, 20: 19, 30: 29}),
                            shape=SHAPE)
    job = tmp_path / "job"
    row = b1b.process_item(HoldTracker(), plan, frames_dir, dense, sparse_doc(), clip, detections(),
                           job / "publish", review=0, profile=False)
    (job / "b1b.json").write_text(json.dumps({"items": [row]}, default=str))
    monkeypatch.setenv("PS_STAGE_DIR", str(tmp_path / "stage"))
    monkeypatch.setenv("PS_SCRATCH_DIR", str(tmp_path / "scratch"))
    for groups, sam_left in (("hands,objects", 10), ("hands", 0)):
        assert b1b.main(["merge", "--result", str(job / "b1b.json"), "--masks", str(job), "--groups", groups]) == 0
        merged = ClipMasks.load(tmp_path / "scratch" / "publish" / "masks" / ITEM["id"] / "masks.rle")
        assert sum(inst.provenance == b1b.SAM_TIER for f in merged.frames for inst in f) == sam_left
        assert sum(len(f) for f in merged.frames) == sum(len(f) for f in clip.frames) + sam_left
