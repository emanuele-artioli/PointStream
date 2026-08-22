"""Generator behaviour: typed conditioning, declared temporal, no overloaded slot."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from src.components.generation import REGISTRY as GENERATORS
from src.components.generation import validate
from src.components.generation.animate_anyone import (
    TENNIS_MATCH_FINETUNE_CAVEAT,
    AnimateAnyoneGenerator,
)
from src.components.generation.controlnet import (
    FINAL_EPOCH,
    ControlNetGenerator,
    render_trajectory_control,
    resolve_controlnet_checkpoint,
)
from src.components.generation.mofa import MofaVideoGenerator
from src.components.generation.pix2pix import Pix2PixGenerator
from src.components.generation.spade import Spade4TennisGenerator
from src.components.generation.upscale import UpscaleRefineGenerator
from src.contracts.conditioning import (
    ConditioningBundle,
    GenerationParams,
    require_sequence,
    supports_sequence,
)
from src.contracts.config import GeneratorConfig, default
from src.contracts.errors import (
    ConfigError,
    MissingConditioningError,
    UnsupportedCapabilityError,
)


def _chw(height: int = 16, width: int = 16, fill: int = 0) -> np.ndarray:
    return np.full((3, height, width), fill, dtype=np.uint8)


def _checker(size: int = 32) -> np.ndarray:
    board = np.zeros((3, size, size), dtype=np.uint8)
    board[:, 0::2, 0::2] = 200
    board[:, 1::2, 1::2] = 200
    board[:, 0::2, 1::2] = 40
    board[:, 1::2, 0::2] = 40
    return board


class _FakePipe:
    def __call__(self, **kwargs: Any) -> np.ndarray:
        return kwargs["image"]


def test_require_names_the_missing_field_rather_than_failing_on_a_shape():
    bundle = ConditioningBundle(appearance=_chw())
    with pytest.raises(MissingConditioningError, match="pose") as excinfo:
        bundle.require("pose")
    assert "appearance" in str(excinfo.value)
    with pytest.raises(ValueError, match="Unknown conditioning kind"):
        bundle.require("dense_dwpose_tensor")


def test_pose_controlnet_require_fails_by_name_when_the_pose_slot_is_empty():
    gen = ControlNetGenerator(variant="pose", pipeline=_FakePipe())
    with pytest.raises(MissingConditioningError, match="pose"):
        gen.generate(
            ConditioningBundle(appearance=_chw()),
            seed=0,
            device="cpu",
            params=GenerationParams(),
        )


def test_canny_controlnet_does_not_read_a_mask_out_of_the_pose_slot():
    """The defect: one parameter meant pose, or canny, or (pose, mask)."""
    gen = ControlNetGenerator(variant="canny", pipeline=_FakePipe())
    pose_only = ConditioningBundle(appearance=_chw(fill=10), pose=_chw(fill=255))
    with pytest.raises(MissingConditioningError, match="canny"):
        gen.generate(pose_only, seed=1, device="cpu", params=GenerationParams())
    canny = np.zeros((16, 16), dtype=np.uint8)
    canny[4:12, 4:12] = 255
    out = gen.generate(
        ConditioningBundle(appearance=_chw(fill=10), canny=canny),
        seed=1,
        device="cpu",
        params=GenerationParams(width=16, height=16),
    )
    assert out.shape[0] == 3


def test_no_registered_generator_accepts_conditioning_as_a_later_positional():
    for spec in GENERATORS:
        cls = spec.resolve()
        generate = getattr(cls, "generate")
        params = list(inspect.signature(generate).parameters.values())
        assert params[0].name == "self"
        assert params[1].name == "conditioning"
        for param in params[2:]:
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{spec.name}.generate parameter {param.name!r} is {param.kind!s}, "
                "not keyword-only"
            )


def test_calling_generate_with_seed_as_a_positional_argument_is_a_typeerror():
    gen = UpscaleRefineGenerator()
    bundle = ConditioningBundle(appearance=_checker())
    params = GenerationParams(width=32, height=32)
    with pytest.raises(TypeError):
        gen.generate(bundle, 0, "cpu", params)  # type: ignore[misc]


def test_temporal_capability_is_read_from_the_declaration_not_from_class_identity():
    assert supports_sequence(GENERATORS.spec("animate-anyone"))
    assert supports_sequence(GENERATORS.spec("mofa-video"))
    assert not supports_sequence(GENERATORS.spec("canny-controlnet"))
    assert not supports_sequence(GENERATORS.spec("upscale-refine"))
    aa = GENERATORS.spec("animate-anyone").resolve()
    mofa = GENERATORS.spec("mofa-video").resolve()
    assert aa is AnimateAnyoneGenerator
    assert mofa is MofaVideoGenerator
    assert not issubclass(mofa, aa)
    require_sequence(GENERATORS.spec("animate-anyone"))
    with pytest.raises(UnsupportedCapabilityError, match="temporal-sequence"):
        require_sequence(GENERATORS.spec("pix2pix"))


def test_animate_anyone_caveat_travels_with_the_registry_entry():
    spec = GENERATORS.spec("animate-anyone")
    assert "single tennis match" in spec.summary
    assert "single tennis match" in TENNIS_MATCH_FINETUNE_CAVEAT


def test_animate_anyone_sequence_uses_the_injected_runtime_not_a_per_frame_hack():
    seen: list[int] = []

    def runtime(bundles: list[ConditioningBundle], **_kwargs: Any) -> list[np.ndarray]:
        seen.append(len(bundles))
        return [_chw(8, 8, fill=3) for _ in bundles]

    gen = AnimateAnyoneGenerator(runtime=runtime, width=8, height=8)
    pose = _chw(8, 8, fill=255)
    appearance = _chw(8, 8, fill=40)
    bundles = [
        ConditioningBundle(appearance=appearance, pose=pose, frame_index=0),
        ConditioningBundle(appearance=appearance, pose=pose, frame_index=1),
    ]
    frames = gen.generate_sequence(
        bundles, seed=7, device="cpu", params=GenerationParams(width=8, height=8)
    )
    assert seen == [2]
    assert len(frames) == 2


def test_upscale_refine_differs_from_identity():
    gen = UpscaleRefineGenerator(sharpen=0.8)
    appearance = _checker(32)
    out = gen.generate(
        ConditioningBundle(appearance=appearance),
        seed=0,
        device="cpu",
        params=GenerationParams(width=32, height=32),
    )
    assert out.shape == appearance.shape
    assert not np.array_equal(out, appearance)
    # A second call with the same seed is bit-identical: this is not diffusion.
    again = gen.generate(
        ConditioningBundle(appearance=appearance),
        seed=99,
        device="cpu",
        params=GenerationParams(width=32, height=32),
    )
    np.testing.assert_array_equal(out, again)


def test_upscale_refine_without_appearance_fails_by_name():
    with pytest.raises(MissingConditioningError, match="appearance"):
        UpscaleRefineGenerator().generate(
            ConditioningBundle(), seed=0, device="cpu", params=GenerationParams()
        )


def test_pix2pix_with_an_injected_model_concatenates_pose_and_appearance():
    captured: list[np.ndarray] = []

    def model(stacked: np.ndarray) -> np.ndarray:
        captured.append(stacked)
        return stacked[:3]

    gen = Pix2PixGenerator(model=model, width=16, height=16)
    appearance = _chw(16, 16, fill=10)
    pose = _chw(16, 16, fill=200)
    out = gen.generate(
        ConditioningBundle(appearance=appearance, pose=pose),
        seed=0,
        device="cpu",
        params=GenerationParams(width=16, height=16),
    )
    assert captured[0].shape[0] == 6
    assert out.shape[0] == 3


def test_mofa_video_refuses_construction_and_does_not_copy_weights():
    with pytest.raises(RuntimeError, match="candidate"):
        GENERATORS.build("mofa-video")
    with pytest.raises(RuntimeError, match="Stability"):
        MofaVideoGenerator()


def test_default_config_does_not_require_a_generator_named_none():
    assert "none" not in GENERATORS
    validate(default())


def test_an_unknown_generator_name_is_rejected_at_backend_validation():
    with pytest.raises(ConfigError):
        validate(default().with_(generator=GeneratorConfig(backend="definitely-not-registered")))


def test_resolved_name_canny_controlnet_is_the_registry_key():
    cfg = default().with_(generator=GeneratorConfig(backend="controlnet", variant="canny"))
    assert cfg.generator.resolved_name == "canny-controlnet"
    GENERATORS.spec(cfg.generator.resolved_name)


def test_controlnet_without_weights_names_the_missing_pipeline():
    gen = ControlNetGenerator(variant="pose", checkpoint="/no/such/pose-controlnet")
    with pytest.raises(FileNotFoundError, match="epoch 10"):
        gen.generate(
            ConditioningBundle(appearance=_chw(), pose=_chw(fill=255)),
            seed=0,
            device="cpu",
            params=GenerationParams(width=16, height=16),
        )


def test_controlnet_epoch_resolution_uses_the_named_epoch_not_the_last_sorted(tmp_path):
    """Silently taking checkpoint-epoch-99 because it sorts last is the defect."""
    base = tmp_path / "pose-controlnet"
    (base / "checkpoint-epoch-1").mkdir(parents=True)
    (base / "checkpoint-epoch-99").mkdir()
    with pytest.raises(FileNotFoundError, match="Refusing to pick another by sort order"):
        resolve_controlnet_checkpoint("pose", str(base), epoch=10)
    path, epoch = resolve_controlnet_checkpoint("pose", str(base), epoch=1)
    assert epoch == 1
    assert path.name == "checkpoint-epoch-1"
    assert FINAL_EPOCH["pose"] == 10
    assert FINAL_EPOCH["seg"] == 7


def test_trajectory_variant_requires_a_motion_field_not_a_pose():
    gen = ControlNetGenerator(variant="trajectory", pipeline=_FakePipe())
    with pytest.raises(MissingConditioningError, match="motion-field"):
        gen.generate(
            ConditioningBundle(appearance=_chw(), pose=_chw(fill=255)),
            seed=0,
            device="cpu",
            params=GenerationParams(width=16, height=16),
        )


def test_trajectory_render_draws_sticks_that_are_not_a_blank_canvas():
    field = np.zeros((2, 32, 32), dtype=np.float32)
    field[0, 8, 8] = 12.0
    field[1, 8, 8] = -6.0
    canvas = render_trajectory_control(field, width=32, height=32, stride=8)
    assert canvas.shape == (32, 32, 3)
    assert canvas.dtype == np.uint8
    assert canvas.sum() > 0
    blank = render_trajectory_control(
        np.zeros((2, 32, 32), dtype=np.float32), width=32, height=32
    )
    assert blank.sum() == 0


def test_trajectory_render_rejects_a_pose_shaped_array_as_a_motion_field():
    with pytest.raises(ValueError, match=r"\(2, H, W\)"):
        render_trajectory_control(np.zeros((3, 16, 16), dtype=np.float32), width=16, height=16)


def test_trajectory_prepare_letterboxes_the_rendered_control():
    gen = ControlNetGenerator(variant="trajectory", width=100, height=50)
    appearance = np.full((3, 100, 50), 128, dtype=np.uint8)
    field = np.zeros((2, 100, 50), dtype=np.float32)
    field[0, 50, 25] = 8.0
    prepared = gen.prepare(
        ConditioningBundle(appearance=appearance, motion_field=field),
        GenerationParams(width=100, height=50),
    )
    assert "trajectory" in prepared
    assert prepared["trajectory"].shape[:2] == (50, 100)
    gen._pipeline = _FakePipe()
    out = gen.generate(
        ConditioningBundle(appearance=appearance, motion_field=field),
        seed=7,
        device="cpu",
        params=GenerationParams(width=100, height=50),
    )
    assert gen.last_seed == 7
    assert out.shape[0] == 3


def test_seed_is_recorded_on_a_fake_pipeline_call():
    gen = ControlNetGenerator(variant="pose", pipeline=_FakePipe())
    gen.generate(
        ConditioningBundle(appearance=_chw(), pose=_chw(fill=255)),
        seed=42,
        device="cpu",
        params=GenerationParams(width=16, height=16),
    )
    assert gen.last_seed == 42


def test_pix2pix_without_weights_names_the_missing_file(tmp_path):
    missing = tmp_path / "no_such_pix2pix.pt"
    gen = Pix2PixGenerator(checkpoint=str(missing), width=16, height=16)
    with pytest.raises(FileNotFoundError, match="pix2pix"):
        gen.generate(
            ConditioningBundle(appearance=_chw(), pose=_chw(fill=255)),
            seed=0,
            device="cpu",
            params=GenerationParams(width=16, height=16),
        )


def test_spade_without_weights_names_the_missing_file(tmp_path):
    missing = tmp_path / "no_such_spade.pt"
    gen = Spade4TennisGenerator(checkpoint=str(missing), width=16, height=16)
    with pytest.raises(FileNotFoundError, match="spade4tennis"):
        gen.generate(
            ConditioningBundle(appearance=_chw(), pose=_chw(fill=255)),
            seed=0,
            device="cpu",
            params=GenerationParams(width=16, height=16),
        )


@pytest.mark.integration
def test_pose_controlnet_resolves_epoch_10_on_disk() -> None:
    from pathlib import Path

    base = Path("assets/weights/pose-controlnet")
    if not base.exists():
        pytest.skip(f"no pose-controlnet weights at {base}")
    path, epoch = resolve_controlnet_checkpoint("pose", str(base), epoch=10)
    assert epoch == 10
    assert path.name == "checkpoint-epoch-10"
    assert (path / "diffusion_pytorch_model.safetensors").is_file()
    gen = ControlNetGenerator(variant="pose", checkpoint=str(base), epoch=10)
    assert gen.epoch == 10
    assert gen.checkpoint == str(base)
