from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.segmentation.sam31 import (
    Sam31Segmenter,
    Sam31SequenceSegmenter,
    _install_compatible_session_start,
)


class _FakePredictor:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.sessions = 0

    def handle_request(self, request: dict):
        self.requests.append(request)
        if request["type"] == "start_session":
            self.sessions += 1
            return {"session_id": f"session-{self.sessions}"}
        if request["type"] == "add_prompt":
            return {
                "outputs": {
                    "out_obj_ids": np.asarray([8]),
                    "out_binary_masks": np.ones((1, 1, 4, 6), dtype=np.uint8),
                }
            }
        return {}

    def handle_stream_request(self, request: dict):
        self.requests.append(request)
        yield {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": np.asarray([8]),
                "out_binary_masks": np.ones((1, 1, 4, 6), dtype=np.uint8),
            },
        }


def test_sam31_scales_box_prompt_to_relative_coordinates() -> None:
    predictor = _FakePredictor()
    segmenter = Sam31SequenceSegmenter(predictor=predictor)
    segmenter.start_session(
        "racket", "/frames", frame_width=60, frame_height=40, policy="offline_causal"
    )
    masks = segmenter.add_prompt(
        "racket",
        frame_index=0,
        object_id="racket-1",
        text="tennis racket",
        bbox=(10, 8, 50, 32),
    )
    request = predictor.requests[-1]
    assert request["bounding_boxes"] == [[0.5, 0.5, 2 / 3, 0.6]]
    assert "points" not in request
    assert request["rel_coordinates"] is True
    assert masks[0].object_id == "racket-1"
    assert masks[0].mask is not None
    assert masks[0].mask.shape == (40, 60)
    segmenter.close_session("racket")


def test_sam31_scales_point_only_prompts_and_rejects_mixed_prompt_forms() -> None:
    predictor = _FakePredictor()
    segmenter = Sam31SequenceSegmenter(predictor=predictor)
    segmenter.start_session(
        "racket", "/frames", frame_width=60, frame_height=40, policy="offline_causal"
    )
    segmenter.add_prompt(
        "racket",
        frame_index=0,
        object_id="racket-1",
        points=[(20, 12)],
        point_labels=[1],
        tracker_id=8,
    )
    request = predictor.requests[-1]
    assert request["points"] == [[1 / 3, 0.3]]
    assert request["point_labels"] == [1]
    assert request["obj_id"] == 8
    assert "text" not in request
    assert "bounding_boxes" not in request
    with pytest.raises(ValueError, match="cannot be combined"):
        segmenter.add_prompt(
            "racket",
            frame_index=0,
            object_id="racket-2",
            text="tennis racket",
            points=[(20, 12)],
            tracker_id=9,
        )
    segmenter.close_session("racket")


def test_sam31_runtime_policy_rejects_bidirectional_propagation() -> None:
    predictor = _FakePredictor()
    segmenter = Sam31SequenceSegmenter(predictor=predictor)
    segmenter.start_session(
        "player", "/frames", frame_width=6, frame_height=4, policy="runtime_causal"
    )
    segmenter.add_prompt("player", frame_index=0, object_id="player-1", text="player")
    with pytest.raises(ValueError, match="forward-only"):
        segmenter.propagate("player", policy="runtime_causal", direction="both", frame_count=1)


def test_sam31_sessions_are_separate_by_object_class() -> None:
    predictor = _FakePredictor()
    segmenter = Sam31SequenceSegmenter(predictor=predictor)
    segmenter.start_session(
        "player", "/frames", frame_width=6, frame_height=4, policy="offline_causal"
    )
    segmenter.start_session(
        "racket", "/frames", frame_width=6, frame_height=4, policy="offline_causal"
    )
    assert (
        len(
            {
                segmenter.sessions[("player", "default")].session_id,
                segmenter.sessions[("racket", "default")].session_id,
            }
        )
        == 2
    )


def test_sam31_prompt_and_propagation_preserve_missing_objects_and_frames() -> None:
    class MissingPredictor(_FakePredictor):
        def handle_request(self, request: dict):
            self.requests.append(request)
            if request["type"] == "start_session":
                self.sessions += 1
                return {"session_id": f"session-{self.sessions}"}
            if request["type"] == "add_prompt":
                return {"outputs": {"out_obj_ids": [], "out_binary_masks": []}}
            return {}

        def handle_stream_request(self, request: dict):
            self.requests.append(request)
            yield {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.asarray([], dtype=np.int64),
                    "out_binary_masks": np.asarray([]),
                },
            }

    segmenter = Sam31SequenceSegmenter(predictor=MissingPredictor())
    segmenter.start_session(
        "racket", "/frames", frame_width=60, frame_height=40, policy="offline_causal"
    )
    first = segmenter.add_prompt(
        "racket", frame_index=0, object_id="racket-1", text="tennis racket"
    )
    assert len(first) == 1
    assert first[0].status.value == "missing"
    assert first[0].reason == "prompt_returned_no_mask"
    propagated = segmenter.propagate(
        "racket", policy="offline_causal", direction="forward", frame_count=2
    )
    assert [(item.frame_index, item.object_id) for item in propagated] == [
        (0, "racket-1"),
        (1, "racket-1"),
    ]
    assert all(item.mask is None and item.status.value == "missing" for item in propagated)


def test_sam31_propagation_preserves_the_tracker_id_for_each_object() -> None:
    predictor = _FakePredictor()
    segmenter = Sam31SequenceSegmenter(predictor=predictor)
    segmenter.start_session(
        "racket", "/frames", frame_width=60, frame_height=40, policy="offline_causal"
    )
    segmenter.add_prompt("racket", frame_index=0, object_id="racket-1", text="tennis racket")

    propagated = segmenter.propagate(
        "racket", policy="offline_causal", direction="forward", frame_count=1
    )

    assert len(propagated) == 1
    assert propagated[0].object_id == "racket-1"
    assert propagated[0].tracker_id == 8


def _jpeg_frames(folder: Path, count: int, size: tuple[int, int] = (6, 4)) -> Path:
    folder.mkdir()
    for index in range(count):
        Image.new("RGB", size).save(folder / f"{index:05d}.jpg")
    return folder


def test_segment_frames_runs_one_text_session_per_class(tmp_path: Path) -> None:
    predictor = _FakePredictor()
    segmenter = Sam31SequenceSegmenter(predictor=predictor)
    frames = _jpeg_frames(tmp_path / "frames", 2)

    masks = segmenter.segment_frames(frames, {"arm": "arm", "hand": "hand"}, fps=25.0)

    assert masks.classes == ("arm", "hand") and len(masks) == 2
    assert (masks.height, masks.width, masks.fps) == (4, 6, 25.0)
    assert [inst.class_name for inst in masks.frames[0]] == ["arm", "hand"]
    assert masks.frames[1] == []  # nothing propagated there: empty, not a whole-frame fill
    assert masks.frames[0][0].track_id == 8
    prompts = [item["text"] for item in predictor.requests if item["type"] == "add_prompt"]
    assert prompts == ["arm", "hand"]
    starts = [item for item in predictor.requests if item["type"] == "start_session"]
    closes = [item for item in predictor.requests if item["type"] == "close_session"]
    assert len(starts) == len(closes) == 2
    directions = [
        item["propagation_direction"]
        for item in predictor.requests
        if item["type"] == "propagate_in_video"
    ]
    assert directions == ["both", "both"]
    assert segmenter.sessions == {}


def test_segment_frames_runtime_policy_propagates_forward_and_shares_identical_prompts(
    tmp_path: Path,
) -> None:
    predictor = _FakePredictor()
    segmenter = Sam31SequenceSegmenter(predictor=predictor)
    frames = _jpeg_frames(tmp_path / "frames", 1)

    masks = segmenter.segment_frames(
        frames, {"player": "person", "body": "person"}, policy="runtime_causal"
    )

    assert [inst.class_name for inst in masks.frames[0]] == ["player", "body"]
    assert len([item for item in predictor.requests if item["type"] == "add_prompt"]) == 1
    assert [
        item["propagation_direction"]
        for item in predictor.requests
        if item["type"] == "propagate_in_video"
    ] == ["forward"]


def test_front_end_runs_the_worker_under_the_sam_interpreter(tmp_path: Path) -> None:
    front = Sam31Segmenter(python=tmp_path / "py", checkpoint=tmp_path / "ckpt.pt")
    command = front.worker_command(tmp_path / "r.json", tmp_path / "o")
    assert command[:3] == [str(tmp_path / "py"), "-m", "src.segmentation.sam31"]
    assert front.source_revision == "2345a4ad109ac29c569da749c91d84f10dc08c40"
    with pytest.raises(FileNotFoundError, match="SAM31_CHECKPOINT"):
        front.segment(tmp_path / "clip.mp4", None)


def test_verified_predictor_session_initializer_signature_is_compatible() -> None:
    class _Model:
        def __init__(self):
            self.kwargs = None

        def init_state(self, resource_path, *, offload_state_to_cpu=False):
            self.kwargs = {
                "resource_path": resource_path,
                "offload_state_to_cpu": offload_state_to_cpu,
            }
            return {"resource_path": resource_path}

    class _LegacyPredictor:
        def __init__(self):
            self.model = _Model()
            self._all_inference_states = {}
            self.async_loading_frames = False

        def start_session(self, resource_path, session_id=None):
            raise AssertionError("compatibility initializer did not replace legacy method")

    predictor = _LegacyPredictor()
    _install_compatible_session_start(predictor)
    result = predictor.start_session("/external/frames", session_id="fixed")
    assert result == {"session_id": "fixed"}
    assert predictor.model.kwargs == {
        "resource_path": "/external/frames",
        "offload_state_to_cpu": False,
    }
    assert predictor._all_inference_states["fixed"]["state"] == {
        "resource_path": "/external/frames"
    }


def test_older_cuda_uses_efficient_then_math_sdpa_fallback() -> None:
    from types import SimpleNamespace

    from src.segmentation.sam31 import _configure_sdpa_backend

    calls: list[tuple[object, bool]] = []

    class Backends:
        FLASH_ATTENTION = object()
        EFFICIENT_ATTENTION = object()
        MATH = object()

    def sdpa_kernel(selected: object, set_priority: bool = False) -> tuple[object, bool]:
        calls.append((selected, set_priority))
        return selected, set_priority

    attention = SimpleNamespace(SDPBackend=Backends, sdpa_kernel=sdpa_kernel)
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (7, 5)),
        nn=SimpleNamespace(attention=attention),
    )

    assert _configure_sdpa_backend(torch_module) == "efficient_then_math_fallback"
    assert attention.sdpa_kernel.__wrapped__ is sdpa_kernel
    result = attention.sdpa_kernel(Backends.FLASH_ATTENTION)

    expected = [Backends.EFFICIENT_ATTENTION, Backends.MATH]
    assert calls == [(expected, True)]
    assert result == (expected, True)


def test_ampere_or_newer_keeps_native_flash_sdpa() -> None:
    from types import SimpleNamespace

    from src.segmentation.sam31 import _configure_sdpa_backend

    def original(selected: object, set_priority: bool = False) -> tuple[object, bool]:
        return selected, set_priority

    attention = SimpleNamespace(SDPBackend=object(), sdpa_kernel=original)
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 6)),
        nn=SimpleNamespace(attention=attention),
    )

    assert _configure_sdpa_backend(torch_module) == "native_flash"
    assert attention.sdpa_kernel is original


@pytest.mark.parametrize(("max_frames", "frames", "windows"), [(None, 5, 3), (3, 3, 2), (4, 4, 2)])
def test_long_clips_run_in_windows_with_unique_track_ids(
    tmp_path: Path, max_frames: int | None, frames: int, windows: int
) -> None:
    from src.segmentation.sam31 import TRACK_ID_STRIDE, segment_chunks

    source = tmp_path / "clip"
    source.mkdir()
    for index in range(5):
        Image.new("RGB", (6, 4)).save(source / f"{index:03d}.png")
    request = {
        "source": str(source),
        "max_frames": max_frames,
        "chunk_frames": 2,
        "fps": 25.0,
        "concepts": {"hand": "hand"},
        "policy": "offline_bidirectional",
    }
    segmenter = Sam31SequenceSegmenter(predictor=_FakePredictor())
    masks, segment_s, extract_s = segment_chunks(segmenter, request)
    assert len(masks) == frames and masks.meta["windows"] == windows
    # The fake predictor reports tracker 8 on the first frame of every window.
    starts = [index for index, frame in enumerate(masks.frames) if frame]
    assert starts == list(range(0, frames, 2))
    expected = [w * TRACK_ID_STRIDE + 8 for w in range(windows)]
    assert [masks.frames[i][0].track_id for i in starts] == expected
    assert segment_s >= 0 and extract_s >= 0
