"""Flagship generators: Animate-Anyone is evaluable; StableAnimator is wrapped.

Behaviour cases, plausible misuse, and an explicit skip list. GPU/weight
forwards are integration concerns and are not asserted here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.components.generation.animate_anyone import (
    FINETUNE_MATCHES,
    FINETUNE_META,
    REQUIRED_PROFILE_ENTRIES,
    TENNIS_MATCH_FINETUNE_CAVEAT,
    AnimateAnyoneGenerator,
    resolve_checkpoint,
)
from src.components.generation.stable_animator import (
    HF_LICENSE_CHECKED,
    HF_MODEL_CARD,
    LICENCE_NOTES,
    SVD_LICENSE_BLOCK,
    StableAnimatorGenerator,
)
from src.contracts.conditioning import ConditioningBundle, GenerationParams
from src.contracts.errors import MissingConditioningError


def _chw(height: int = 16, width: int = 16, fill: int = 0) -> np.ndarray:
    return np.full((3, height, width), fill, dtype=np.uint8)


def _pose_appearance(size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    return _chw(size, size, fill=255), _chw(size, size, fill=40)


def _bundles(n: int = 2, size: int = 16) -> list[ConditioningBundle]:
    pose, appearance = _pose_appearance(size)
    return [
        ConditioningBundle(appearance=appearance, pose=pose, frame_index=i) for i in range(n)
    ]


def _fake_runtime(fill: int = 3) -> Any:
    def runtime(bundles: list[ConditioningBundle], **kwargs: Any) -> list[np.ndarray]:
        runtime.calls.append({"n": len(bundles), **kwargs})  # type: ignore[attr-defined]
        h = bundles[0].appearance.shape[1]  # type: ignore[union-attr]
        w = bundles[0].appearance.shape[2]  # type: ignore[union-attr]
        return [_chw(h, w, fill=fill) for _ in bundles]

    runtime.calls = []  # type: ignore[attr-defined]
    return runtime


# -- Animate-Anyone: checkpoint wiring ---------------------------------------


def test_resolve_checkpoint_names_missing_profile_entries(tmp_path: Path) -> None:
    empty = tmp_path / "profile"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="denoising_unet.pth") as excinfo:
        resolve_checkpoint(empty)
    assert "tennis" in str(excinfo.value).lower() or "match" in str(excinfo.value).lower()


def test_resolve_checkpoint_accepts_a_complete_profile(tmp_path: Path) -> None:
    root = tmp_path / "finetuned_tennis"
    root.mkdir()
    for name in REQUIRED_PROFILE_ENTRIES:
        target = root / name
        if name.endswith(".pth"):
            target.write_bytes(b"stub")
        else:
            target.mkdir()
    assert resolve_checkpoint(root) == root.resolve()


def test_resolve_checkpoint_rejects_a_file_path(tmp_path: Path) -> None:
    blob = tmp_path / "weights.pth"
    blob.write_bytes(b"nope")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        resolve_checkpoint(blob)


def test_finetune_caveat_names_the_training_meta_not_a_general_model() -> None:
    assert FINETUNE_META in TENNIS_MATCH_FINETUNE_CAVEAT
    assert "not a general human model" in TENNIS_MATCH_FINETUNE_CAVEAT
    assert "alcaraz_highlights" in FINETUNE_MATCHES
    assert len(FINETUNE_MATCHES) == 7


def test_animate_anyone_empty_sequence_is_rejected() -> None:
    gen = AnimateAnyoneGenerator(runtime=_fake_runtime(), width=8, height=8)
    with pytest.raises(ValueError, match="at least one"):
        gen.generate_sequence([], seed=0, device="cpu", params=GenerationParams())


def test_animate_anyone_missing_pose_on_later_bundle_fails_by_name() -> None:
    gen = AnimateAnyoneGenerator(runtime=_fake_runtime(), width=8, height=8)
    pose, appearance = _pose_appearance(8)
    bundles = [
        ConditioningBundle(appearance=appearance, pose=pose, frame_index=0),
        ConditioningBundle(appearance=appearance, frame_index=1),
    ]
    with pytest.raises(MissingConditioningError, match="pose"):
        gen.generate_sequence(
            bundles, seed=0, device="cpu", params=GenerationParams(width=8, height=8)
        )


def test_animate_anyone_without_runtime_names_the_caveat_when_profile_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "no-such-profile"
    monkeypatch.setattr(
        "src.components.generation.animate_anyone._DEFAULT_PROFILE_CANDIDATES",
        (missing,),
    )
    gen = AnimateAnyoneGenerator(checkpoint=None, width=8, height=8)
    with pytest.raises(FileNotFoundError, match="not found"):
        gen.generate_sequence(
            _bundles(1, size=8),
            seed=0,
            device="cpu",
            params=GenerationParams(width=8, height=8),
        )


def test_cuda_peak_helpers_are_safe_on_cpu() -> None:
    from src.components.generation.animate_anyone import _cuda_peak_bytes, _reset_cuda_peak

    _reset_cuda_peak("cpu")
    assert _cuda_peak_bytes("cpu") == 0
    runtime = _fake_runtime(fill=9)
    gen = AnimateAnyoneGenerator(runtime=runtime, width=8, height=8)
    frames = gen.generate_sequence(
        _bundles(3, size=8), seed=1, device="cpu", params=GenerationParams(width=8, height=8)
    )
    assert len(frames) == 3
    assert frames[0].shape == (3, 8, 8)
    assert frames[0].dtype == np.uint8


def test_prepare_letterboxes_appearance_and_pose_independently_when_sizes_differ() -> None:
    """Coding-task geometry: keyframe appearance and a later pose, different crops.

    Canvas 100x50. Appearance is 50x100 (tall); pose is 100x50 (wide). The old
    path resized the pose onto the appearance canvas, so the skeleton landed in
    a 25-wide strip and scoring against the later frame compared two layouts.
    Independent letterbox: appearance 25x50 at x=37; pose fills 100x50.
    """
    from src.components.generation.pose import fit_to_canvas

    appearance = np.full((3, 100, 50), 200, dtype=np.uint8)
    pose = np.full((3, 50, 100), 255, dtype=np.uint8)
    gen = AnimateAnyoneGenerator(runtime=_fake_runtime(), width=100, height=50)
    prepared = gen._prepare(
        ConditioningBundle(appearance=appearance, pose=pose),
        GenerationParams(width=100, height=50),
    )
    expected_app = fit_to_canvas(50, 100, 100, 50)
    expected_pose = fit_to_canvas(100, 50, 100, 50)
    assert prepared["letterbox"] == expected_app
    assert prepared["pose_letterbox"] == expected_pose
    app_hwc = prepared["appearance"]
    pose_hwc = prepared["pose"]
    assert app_hwc.shape[:2] == (50, 100)
    assert pose_hwc.shape[:2] == (50, 100)
    assert app_hwc[:, :37].sum() == 0
    assert int(app_hwc[:, 37:62].max()) == 200
    assert int(pose_hwc.min()) == 255
    assert not np.array_equal(app_hwc, pose_hwc)


def test_prepare_does_not_replace_the_reference_with_the_pose_canvas() -> None:
    """A dropped reference looks like a pose-coloured canvas. Appearance stays red."""
    appearance = np.zeros((3, 16, 8), dtype=np.uint8)
    appearance[0] = 180
    pose = np.zeros((3, 8, 16), dtype=np.uint8)
    pose[1] = 220
    gen = AnimateAnyoneGenerator(runtime=_fake_runtime(), width=16, height=16)
    prepared = gen._prepare(
        ConditioningBundle(appearance=appearance, pose=pose),
        GenerationParams(width=16, height=16),
    )
    ref = prepared["appearance"]
    pose_canvas = prepared["pose"]
    assert int(ref[:, :, 0].max()) == 180
    assert int(ref[:, :, 1].max()) == 0
    assert int(pose_canvas[:, :, 1].max()) == 220
    assert int(pose_canvas[:, :, 0].max()) == 0


def test_prepare_same_size_inputs_share_one_letterbox() -> None:
    appearance = np.full((3, 16, 8), 40, dtype=np.uint8)
    pose = np.full((3, 16, 8), 255, dtype=np.uint8)
    gen = AnimateAnyoneGenerator(runtime=_fake_runtime(), width=16, height=16)
    prepared = gen._prepare(
        ConditioningBundle(appearance=appearance, pose=pose),
        GenerationParams(width=16, height=16),
    )
    assert prepared["letterbox"] == prepared["pose_letterbox"]


def test_reference_feed_stats_flag_a_blank_or_pose_copied_reference() -> None:
    from src.components.generation.animate_anyone import _reference_feed_stats

    pose = np.full((8, 8, 3), 90, dtype=np.uint8)
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    copied = pose.copy()
    real = np.full((8, 8, 3), 40, dtype=np.uint8)
    blank_stats = _reference_feed_stats(blank, pose)
    copied_stats = _reference_feed_stats(copied, pose)
    real_stats = _reference_feed_stats(real, pose)
    assert blank_stats["reference_is_blank"] is True
    assert copied_stats["reference_equals_pose"] is True
    assert real_stats["reference_is_blank"] is False
    assert real_stats["reference_equals_pose"] is False
    assert real_stats["reference_mean"] == 40.0


# -- StableAnimator: licence + contract --------------------------------------


def test_stable_animator_licence_notes_record_the_hf_card_and_svd_block() -> None:
    assert HF_LICENSE_CHECKED == "apache-2.0"
    assert "FrancisRing/StableAnimator" in HF_MODEL_CARD
    assert "apache-2.0" in LICENCE_NOTES
    assert "Stability" in SVD_LICENSE_BLOCK
    assert "not bundled" in SVD_LICENSE_BLOCK
    gen = StableAnimatorGenerator()
    assert gen.licence_notes == LICENCE_NOTES


def test_stable_animator_constructs_unlike_mofa() -> None:
    from src.contracts.capabilities import CONDITION_APPEARANCE, CONDITION_POSE

    gen = StableAnimatorGenerator()
    assert gen.required == (CONDITION_POSE, CONDITION_APPEARANCE)


def test_stable_animator_without_runtime_names_svd_and_refuses_to_invent_frames() -> None:
    gen = StableAnimatorGenerator(width=8, height=8)
    with pytest.raises(RuntimeError, match="Stability") as excinfo:
        gen.generate_sequence(
            _bundles(2, size=8),
            seed=0,
            device="cpu",
            params=GenerationParams(width=8, height=8),
        )
    message = str(excinfo.value)
    assert "not bundled" in message
    assert "pointstream" in message.lower()


def test_stable_animator_injected_runtime_receives_letterboxed_pose_and_faces() -> None:
    seen: dict[str, Any] = {}

    def runtime(
        bundles: list[ConditioningBundle],
        **kwargs: Any,
    ) -> list[np.ndarray]:
        seen["n"] = len(bundles)
        seen["faces"] = kwargs.get("faces")
        seen["prepared"] = kwargs.get("prepared")
        return [_chw(8, 8, fill=1) for _ in bundles]

    gen = StableAnimatorGenerator(runtime=runtime, width=8, height=8)
    params = GenerationParams(width=8, height=8, extra={"faces": "from-extra"})
    frames = gen.generate_sequence(_bundles(2, size=8), seed=4, device="cpu", params=params)
    assert seen["n"] == 2
    assert seen["faces"] == "from-extra"
    assert seen["prepared"][0]["pose"].shape[0] == 8
    assert len(frames) == 2


def test_stable_animator_does_not_invent_faces_when_extra_omits_them() -> None:
    seen: list[Any] = []

    def runtime(bundles: list[ConditioningBundle], **kwargs: Any) -> list[np.ndarray]:
        seen.append(kwargs.get("faces"))
        return [_chw(8, 8) for _ in bundles]

    gen = StableAnimatorGenerator(runtime=runtime, width=8, height=8)
    gen.generate_sequence(
        _bundles(1, size=8), seed=0, device="cpu", params=GenerationParams(width=8, height=8)
    )
    assert seen == [None]


def test_stable_animator_missing_pose_fails_by_name_not_with_a_shape_error() -> None:
    gen = StableAnimatorGenerator(runtime=_fake_runtime(), width=8, height=8)
    with pytest.raises(MissingConditioningError, match="pose"):
        gen.generate(
            ConditioningBundle(appearance=_chw(8, 8, fill=10)),
            seed=0,
            device="cpu",
            params=GenerationParams(width=8, height=8),
        )


def test_stable_animator_empty_sequence_is_rejected() -> None:
    gen = StableAnimatorGenerator(runtime=_fake_runtime())
    with pytest.raises(ValueError, match="at least one"):
        gen.generate_sequence([], seed=0, device="cpu", params=GenerationParams())


def test_stable_animator_does_not_read_a_pose_out_of_the_appearance_slot() -> None:
    """A caller who only has appearance must not silently animate from it."""
    gen = StableAnimatorGenerator(runtime=_fake_runtime(), width=8, height=8)
    appearance = _chw(8, 8, fill=77)
    with pytest.raises(MissingConditioningError, match="pose"):
        gen.generate_sequence(
            [ConditioningBundle(appearance=appearance)],
            seed=0,
            device="cpu",
            params=GenerationParams(width=8, height=8),
        )
