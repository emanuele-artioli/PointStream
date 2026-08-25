"""ControlNetDataset reference sampling. Behaviour, not coverage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.train_controlnet import ControlNetDataset, compose_pose_on_appearance_tensor


class _FakeTokenizer:
    model_max_length = 8

    def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
        _ = text, kwargs
        return SimpleNamespace(input_ids=torch.zeros(8, dtype=torch.long))


def _write_png(path: Path, fill: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 12, 3), fill, dtype=np.uint8)).save(path)


def test_include_reference_samples_a_different_frame_from_the_same_track(tmp_path: Path) -> None:
    video = tmp_path / "alcaraz_ruud" / "segmentations" / "scene_001"
    track = video / "track_0001"
    pose = video / "track_0001_pose_body"
    _write_png(track / "frame_000010.png", 10)
    _write_png(track / "frame_000011.png", 200)
    _write_png(pose / "frame_000000.png", 255)
    _write_png(pose / "frame_000001.png", 255)
    dataset = ControlNetDataset(
        str(tmp_path),
        "pose",
        target_size=16,
        tokenizer=_FakeTokenizer(),
        include_reference=True,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert "reference_pixel_values" in sample
    # Item 0 is the dark frame (fill 10, then Normalize on the target only).
    # The other colour frame is fill 200; cond_transform keeps [0, 1].
    assert float(sample["pixel_values"].mean()) < 0.0
    assert float(sample["reference_pixel_values"].mean()) > 0.5


def test_include_reference_skips_a_one_frame_track(tmp_path: Path) -> None:
    video = tmp_path / "alcaraz_ruud" / "segmentations" / "scene_001"
    _write_png(video / "track_0001" / "frame_000000.png", 10)
    _write_png(video / "track_0001_pose_body" / "frame_000000.png", 255)
    dataset = ControlNetDataset(
        str(tmp_path),
        "pose",
        target_size=16,
        tokenizer=_FakeTokenizer(),
        include_reference=True,
    )
    assert len(dataset) == 0


def test_compose_tensor_puts_appearance_in_the_pose_black() -> None:
    pose = torch.zeros(1, 3, 4, 4)
    pose[:, :, 1:3, 1:3] = 1.0
    appearance = torch.full((1, 3, 4, 4), 0.2)
    composed = compose_pose_on_appearance_tensor(pose, appearance)
    assert float(composed[0, 0, 0, 0]) == pytest.approx(0.2)
    assert float(composed[0, 0, 1, 1]) == pytest.approx(1.0)


def test_ip_adapter_refuses_to_train_without_a_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image-embedding pathway"):
        ControlNetDataset(str(tmp_path), "ip-adapter", tokenizer=_FakeTokenizer())


def test_ip_adapter_pairs_pose_not_the_seg_branch(tmp_path: Path) -> None:
    video = tmp_path / "alcaraz_ruud" / "segmentations" / "scene_001"
    track = video / "track_0001"
    pose = video / "track_0001_pose_body"
    _write_png(track / "frame_000010.png", 10)
    _write_png(track / "frame_000011.png", 200)
    _write_png(pose / "frame_000000.png", 40)
    _write_png(pose / "frame_000001.png", 40)
    dataset = ControlNetDataset(
        str(tmp_path),
        "ip-adapter",
        target_size=16,
        tokenizer=_FakeTokenizer(),
        include_reference=True,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert "reference_pixel_values" in sample
    # Pose canvas is fill 40 with black letterbox, not the colour crop (fill 10).
    cond_mean = float(sample["conditioning_pixel_values"].mean())
    colour_mean = float(sample["pixel_values"].mean())
    assert cond_mean > 0.08
    assert cond_mean > colour_mean


def test_ip_adapter_control_is_pose_not_painted_reference() -> None:
    from scripts.train_controlnet import controlnet_cond_for_batch

    pose = torch.zeros(1, 3, 4, 4)
    pose[0, :, 1, 1] = 1.0
    reference = torch.full((1, 3, 4, 4), 0.4)
    batch = {
        "conditioning_pixel_values": pose,
        "reference_pixel_values": reference,
    }
    out = controlnet_cond_for_batch(
        batch, condition_type="ip-adapter", include_reference=True, weight_dtype=torch.float32
    )
    assert torch.equal(out, pose)


def test_pose_reference_still_paints_under_the_skeleton() -> None:
    from scripts.train_controlnet import controlnet_cond_for_batch

    pose = torch.zeros(1, 3, 4, 4)
    pose[0, :, 1, 1] = 1.0
    reference = torch.full((1, 3, 4, 4), 0.4)
    batch = {
        "conditioning_pixel_values": pose,
        "reference_pixel_values": reference,
    }
    out = controlnet_cond_for_batch(
        batch, condition_type="pose", include_reference=True, weight_dtype=torch.float32
    )
    assert float(out[0, 0, 0, 0]) == pytest.approx(0.4)
    assert float(out[0, 0, 1, 1]) == pytest.approx(1.0)


def test_collect_ip_adapter_parameters_skips_the_backbone() -> None:
    from scripts.train_controlnet import collect_ip_adapter_parameters

    class _IPProc(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_k_ip = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])

    class _UNet:
        def __init__(self) -> None:
            self.encoder_hid_proj = torch.nn.Linear(3, 3, bias=False)
            self.attn_processors = {
                "ip": _IPProc(),
                "self": torch.nn.Linear(2, 2, bias=False),
            }

    unet = _UNet()
    params = collect_ip_adapter_parameters(unet)
    ids = {id(p) for p in params}
    assert id(next(unet.encoder_hid_proj.parameters())) in ids
    assert id(next(unet.attn_processors["ip"].parameters())) in ids
    assert id(next(unet.attn_processors["self"].parameters())) not in ids
