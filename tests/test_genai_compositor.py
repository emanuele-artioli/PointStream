from __future__ import annotations

import sys
from types import SimpleNamespace
import types

import cv2
import numpy as np
import pytest
import torch

import src.decoder.genai_compositor as gc


def _install_fake_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fromarray(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.uint8)

    image_module = types.ModuleType("PIL.Image")
    setattr(image_module, "fromarray", _fromarray)

    pil_module = types.ModuleType("PIL")
    setattr(pil_module, "Image", image_module)

    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)


def _make_pose_sequence(valid_confidence: float = 0.95) -> torch.Tensor:
    pose = torch.zeros((1, 18, 3), dtype=torch.float32)
    pose[0, :, 0] = torch.linspace(70.0, 120.0, 18)
    pose[0, :, 1] = torch.linspace(30.0, 150.0, 18)
    pose[0, :, 2] = valid_confidence
    return pose


def test_mock_compositor_process_changes_frame() -> None:
    compositor = gc.MockCompositor()

    reference = torch.full((3, 96, 64), 200, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    background = torch.zeros((3, 180, 320), dtype=torch.uint8)

    out = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
    )

    assert tuple(out.shape) == (3, 180, 320)
    assert out.dtype == torch.uint8
    assert int(torch.count_nonzero(out)) > 0


def test_mock_compositor_handles_no_visible_pose() -> None:
    compositor = gc.MockCompositor()

    reference = torch.full((3, 80, 48), 120, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.0)
    background = torch.zeros((3, 120, 200), dtype=torch.uint8)

    out = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
    )
    assert int(torch.count_nonzero(out)) > 0


class _DummyStrategy(gc.BaseGenAIStrategy):
    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        _ = dense_dwpose_tensor
        _ = seed
        _ = device
        h = int(reference_crop_tensor.shape[1])
        w = int(reference_crop_tensor.shape[2])
        actor = np.zeros((h, w, 3), dtype=np.uint8)
        actor[:, :, 1] = 220
        actor[:, :, 2] = 80
        return torch.from_numpy(actor).permute(2, 0, 1).contiguous().to(torch.uint8)


def test_diffusers_compositor_process_with_stub_strategy(monkeypatch) -> None:
    monkeypatch.setattr(gc.DiffusersCompositor, "_build_strategy", lambda self, backend: _DummyStrategy())

    compositor = gc.DiffusersCompositor(backend="controlnet", seed=1234)
    reference = torch.full((3, 64, 48), 180, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    background = torch.zeros((3, 180, 320), dtype=torch.uint8)

    out = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
    )

    assert tuple(out.shape) == (3, 180, 320)
    assert int(torch.count_nonzero(out)) > 0


def test_diffusers_compositor_metadata_mask_mode_limits_blending(monkeypatch) -> None:
    monkeypatch.setenv("POINTSTREAM_COMPOSITING_MASK_MODE", "metadata-source-mask")
    monkeypatch.setattr(gc.DiffusersCompositor, "_build_strategy", lambda self, backend: _DummyStrategy())

    compositor = gc.DiffusersCompositor(backend="controlnet", seed=1234, device="cpu")
    compositor._alpha_temporal_smoothing = 0.0

    reference = torch.full((3, 64, 48), 180, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    background = torch.zeros((3, 180, 320), dtype=torch.uint8)

    metadata_mask = np.zeros((24, 16), dtype=np.uint8)
    metadata_mask[6:18, 5:11] = 255

    out_with_metadata = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
        actor_identity="actor_with_metadata",
        metadata_mask=metadata_mask,
    )
    out_without_metadata = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
        actor_identity="actor_without_metadata",
        metadata_mask=None,
    )

    assert int(torch.count_nonzero(out_with_metadata)) < int(torch.count_nonzero(out_without_metadata))


def test_diffusers_compositor_metadata_mode_uses_metadata_bbox_for_placement(monkeypatch) -> None:
    monkeypatch.setenv("POINTSTREAM_COMPOSITING_MASK_MODE", "metadata-source-mask")
    monkeypatch.setattr(gc.DiffusersCompositor, "_build_strategy", lambda self, backend: _DummyStrategy())

    compositor = gc.DiffusersCompositor(backend="controlnet", seed=1234, device="cpu")
    compositor._alpha_temporal_smoothing = 0.0

    reference = torch.full((3, 64, 48), 180, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    background = torch.zeros((3, 180, 320), dtype=torch.uint8)

    metadata_mask = np.full((24, 16), 255, dtype=np.uint8)
    metadata_bbox = (10, 12, 34, 44)

    def _should_not_use_pose_bbox(*_args, **_kwargs):
        raise AssertionError("_estimate_bbox_from_pose should not be called in metadata-source-mask mode")

    monkeypatch.setattr(compositor, "_estimate_bbox_from_pose", _should_not_use_pose_bbox)

    out = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
        actor_identity="actor_metadata_bbox",
        metadata_mask=metadata_mask,
        metadata_bbox=metadata_bbox,
    )

    out_np = out.permute(1, 2, 0).cpu().numpy()
    nonzero = np.argwhere(np.any(out_np > 0, axis=2))
    assert nonzero.size > 0

    y_min = int(np.min(nonzero[:, 0]))
    x_min = int(np.min(nonzero[:, 1]))
    y_max = int(np.max(nonzero[:, 0])) + 1
    x_max = int(np.max(nonzero[:, 1])) + 1

    assert x_min >= metadata_bbox[0]
    assert y_min >= metadata_bbox[1]
    assert x_max <= metadata_bbox[2]
    assert y_max <= metadata_bbox[3]


def test_diffusers_compositor_metadata_mode_intersects_generated_alpha(monkeypatch) -> None:
    monkeypatch.setenv("POINTSTREAM_COMPOSITING_MASK_MODE", "metadata-source-mask")
    monkeypatch.setattr(gc.DiffusersCompositor, "_build_strategy", lambda self, backend: _DummyStrategy())

    compositor = gc.DiffusersCompositor(backend="animate-anyone", seed=7, device="cpu")
    compositor._alpha_temporal_smoothing = 0.0

    reference = torch.full((3, 64, 48), 180, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    background = torch.zeros((3, 180, 320), dtype=torch.uint8)

    metadata_mask = np.full((24, 16), 255, dtype=np.uint8)
    metadata_bbox = (40, 50, 64, 82)

    monkeypatch.setattr(
        compositor,
        "_segment_generated_actor",
        lambda actor_resized, is_animate_anyone: np.zeros(actor_resized.shape[:2], dtype=np.float32),
    )

    out = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
        actor_identity="actor_metadata_alpha_intersection",
        metadata_mask=metadata_mask,
        metadata_bbox=metadata_bbox,
    )

    # Generated alpha is forced to zero; metadata mode must respect intersection and avoid blending.
    assert int(torch.count_nonzero(out)) == 0


def test_diffusers_compositor_postgen_seg_client_falls_back_to_heuristic(monkeypatch) -> None:
    monkeypatch.setenv("POINTSTREAM_COMPOSITING_MASK_MODE", "postgen-seg-client")
    monkeypatch.setenv("POINTSTREAM_POSTGEN_SEGMENTER_BACKEND", "yolo")
    monkeypatch.setattr(gc.DiffusersCompositor, "_build_strategy", lambda self, backend: _DummyStrategy())

    compositor = gc.DiffusersCompositor(backend="animate-anyone", seed=4, device="cpu")
    monkeypatch.setattr(compositor, "_ensure_postgen_segmenter", lambda: None)

    actor = np.zeros((96, 64, 3), dtype=np.uint8)
    cv2.rectangle(actor, (18, 20), (45, 82), color=(40, 220, 160), thickness=-1)

    alpha = compositor._segment_generated_actor(actor_resized=actor, is_animate_anyone=True)
    assert alpha is not None
    assert int(np.count_nonzero(alpha > 0.01)) > 0


def test_diffusers_compositor_invalid_backend_raises() -> None:
    with pytest.raises(ValueError):
        gc.DiffusersCompositor(backend="unknown-backend")


def test_diffusers_compositor_black_background_mask_skips_empty_foreground() -> None:
    compositor = gc.DiffusersCompositor(backend="animate-anyone", seed=3, device="cpu")
    actor = np.zeros((64, 48, 3), dtype=np.uint8)

    alpha = compositor._segment_black_background(actor)
    assert alpha is None


def test_diffusers_compositor_black_background_mask_returns_soft_alpha() -> None:
    compositor = gc.DiffusersCompositor(backend="animate-anyone", seed=3, device="cpu")
    actor = np.zeros((80, 56, 3), dtype=np.uint8)
    cv2.rectangle(actor, (15, 10), (42, 70), color=(35, 210, 160), thickness=-1)

    alpha = compositor._segment_black_background(actor)
    assert alpha is not None
    assert alpha.dtype == np.float32
    assert float(np.max(alpha)) > 0.9
    assert int(np.count_nonzero(alpha > 0.01)) > 0


def test_diffusers_compositor_alpha_smoothing_is_actor_scoped() -> None:
    compositor = gc.DiffusersCompositor(backend="animate-anyone", seed=3, device="cpu")
    compositor._alpha_temporal_smoothing = 0.5

    base_mask = np.ones((8, 8), dtype=np.float32)
    out_a0 = compositor._apply_temporal_alpha_smoothing(base_mask, actor_identity="actor_a")
    out_a1 = compositor._apply_temporal_alpha_smoothing(np.zeros((8, 8), dtype=np.float32), actor_identity="actor_a")
    out_b0 = compositor._apply_temporal_alpha_smoothing(base_mask, actor_identity="actor_b")

    assert out_a0 is not None
    assert out_a1 is not None
    assert out_b0 is not None
    assert float(np.mean(out_a1)) < 1.0
    assert float(np.mean(out_a1)) > 0.0
    assert float(np.mean(out_b0)) == pytest.approx(1.0)


def test_diffusers_compositor_adaptive_black_threshold_never_lowers_base() -> None:
    compositor = gc.DiffusersCompositor(backend="animate-anyone", seed=3, device="cpu")
    actor = np.zeros((24, 20, 3), dtype=np.uint8)
    actor[6:18, 7:13] = np.array([10, 180, 200], dtype=np.uint8)

    threshold, span = compositor._estimate_adaptive_black_thresholds(
        actor_bgr=actor,
        base_threshold=8,
        base_span_threshold=6,
    )

    assert threshold >= 8
    assert span >= 6


def test_baseline_controlnet_strategy_generate_with_fake_pipe(monkeypatch) -> None:
    _install_fake_pillow(monkeypatch)
    pil_image = sys.modules["PIL.Image"]

    strategy = gc.BaselineControlNetStrategy()

    class _FakePipe:
        def __call__(self, **kwargs):
            _ = kwargs
            fake_rgb = np.full((64, 48, 3), 140, dtype=np.uint8)
            return SimpleNamespace(images=[pil_image.fromarray(fake_rgb)])

    monkeypatch.setattr(strategy, "_ensure_pipeline", lambda device: _FakePipe())

    reference = torch.full((3, 64, 48), 90, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    out = strategy.generate(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        seed=2026,
        device=torch.device("cpu"),
    )

    assert tuple(out.shape) == (3, 64, 48)
    assert out.dtype == torch.uint8


def test_animate_anyone_strategy_missing_repo_raises(monkeypatch) -> None:
    monkeypatch.delenv("POINTSTREAM_ANIMATE_ANYONE_REPO_DIR", raising=False)
    strategy = gc.AnimateAnyoneStrategy(repo_dir=None)

    with pytest.raises(FileNotFoundError):
        strategy.generate(
            reference_crop_tensor=torch.zeros((3, 32, 32), dtype=torch.uint8),
            dense_dwpose_tensor=_make_pose_sequence(valid_confidence=0.9),
            seed=1,
            device=torch.device("cpu"),
        )


def test_animate_anyone_strategy_generate_with_stub_runtime(monkeypatch) -> None:
    strategy = gc.AnimateAnyoneStrategy(repo_dir="/tmp/does-not-matter")

    def _runtime_fn(reference_image_bgr, dense_pose_sequence, seed, device):
        _ = dense_pose_sequence
        _ = seed
        _ = device
        return np.asarray(reference_image_bgr, dtype=np.uint8)

    monkeypatch.setattr(strategy, "_ensure_runtime", lambda: _runtime_fn)

    reference = torch.full((3, 40, 30), 77, dtype=torch.uint8)
    out = strategy.generate(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=_make_pose_sequence(valid_confidence=0.9),
        seed=44,
        device=torch.device("cpu"),
    )

    assert tuple(out.shape) == (3, 40, 30)


def test_numpy_and_pose_helpers() -> None:
    float_img = torch.rand((3, 40, 30), dtype=torch.float32) * 255.0
    np_img = gc._to_numpy_bgr(float_img)
    assert np_img.dtype == np.uint8
    assert np_img.shape == (40, 30, 3)

    pose = _make_pose_sequence(valid_confidence=0.8)[0]
    condition = gc._render_pose_condition(pose, output_height=64, output_width=64)
    assert condition.shape == (64, 64, 3)
    assert int(np.max(condition)) > 0

    with pytest.raises(ValueError):
        gc._render_pose_condition(torch.zeros((17, 3), dtype=torch.float32), output_height=64, output_width=64)

    with pytest.raises(ValueError):
        gc._to_numpy_bgr(torch.zeros((2, 40, 30), dtype=torch.uint8))
