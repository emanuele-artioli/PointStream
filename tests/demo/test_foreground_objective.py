from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from demo.models.foreground_objective import bgr_uint8_to_rgb_tensor, compose, smoke_hand_objective  # noqa: E402 - optional torch dependency
from demo.models.foreground_smoke_net import checkpoint_kind  # noqa: E402 - optional torch dependency


def _tensor(rgb, alpha):
    return torch.cat((rgb, alpha), dim=1)


def test_exact_rgb_and_alpha_has_zero_loss():
    rgb = torch.rand(1, 3, 4, 5) * 2 - 1
    alpha = torch.zeros(1, 1, 4, 5)
    alpha[:, :, 1:3, 1:4] = 1
    total, parts = smoke_hand_objective(_tensor(rgb, alpha), rgb, alpha)
    assert total.item() == pytest.approx(0.0, abs=1e-6)
    assert all(value.item() == pytest.approx(0.0, abs=1e-6) for value in parts.values())


def test_all_zero_prediction_alpha_is_penalized_inside_and_gradient_increases_it():
    rgb = torch.zeros(1, 3, 2, 2)
    target_alpha = torch.ones(1, 1, 2, 2)
    output = _tensor(rgb.clone(), torch.zeros_like(target_alpha)).requires_grad_()
    total, parts = smoke_hand_objective(output, rgb, target_alpha)
    total.backward()
    assert parts["balanced_alpha_mae"].item() == pytest.approx(1.0)
    assert output.grad[:, 3:4].mean().item() < 0


def test_all_one_prediction_alpha_has_outside_penalty_and_gradient_decreases_it():
    rgb = torch.zeros(1, 3, 2, 2)
    target_alpha = torch.zeros(1, 1, 2, 2)
    target_alpha[:, :, 0, 0] = 1
    output = _tensor(rgb.clone(), torch.ones_like(target_alpha)).requires_grad_()
    total, parts = smoke_hand_objective(output, rgb, target_alpha)
    total.backward()
    assert parts["balanced_alpha_mae"].item() > 0
    assert output.grad[:, 3:4, 0, 1:].mean().item() > 0


def test_empty_and_full_masks_use_one_region_and_stay_finite():
    source = torch.ones(1, 3, 2, 2) * 0.5
    empty = torch.zeros(1, 1, 2, 2)
    wrong_rgb = torch.ones_like(source) * -0.5
    total, parts = smoke_hand_objective(_tensor(wrong_rgb, empty), source, empty)
    assert torch.isfinite(total)
    assert parts["masked_rgb_mae"].item() == 0
    assert parts["composite_mae"].item() == 0
    assert parts["balanced_alpha_mae"].item() == 0

    full = torch.ones_like(empty)
    wrong_alpha = torch.zeros_like(full)
    total, parts = smoke_hand_objective(_tensor(source, wrong_alpha), source, full)
    assert torch.isfinite(total)
    assert parts["balanced_alpha_mae"].item() == pytest.approx(1.0)


def test_rgb_outside_zero_alpha_cannot_change_composite_but_leak_is_scored():
    source = torch.zeros(1, 3, 2, 2)
    target_alpha = torch.zeros(1, 1, 2, 2)
    target_alpha[:, :, 0, 0] = 1
    pred_a = target_alpha.clone()
    rgb_a = source.clone()
    rgb_b = source.clone()
    rgb_b[:, :, 1, 1] = 1
    _a, parts_a = smoke_hand_objective(_tensor(rgb_a, pred_a), source, target_alpha)
    _b, parts_b = smoke_hand_objective(_tensor(rgb_b, pred_a), source, target_alpha)
    assert parts_a["composite_mae"].item() == pytest.approx(parts_b["composite_mae"].item())
    assert compose(rgb_a, pred_a)[:, :, 1, 1].tolist() == compose(rgb_b, pred_a)[:, :, 1, 1].tolist()
    leaked = pred_a.clone()
    leaked[:, :, 1, 1] = 0.2
    _, leak_parts = smoke_hand_objective(_tensor(rgb_b, leaked), source, target_alpha)
    assert leak_parts["composite_mae"].item() > parts_b["composite_mae"].item()


def test_masked_rgb_mae_uses_alpha_area_and_channel_normalization():
    source = torch.zeros(1, 3, 1, 2)
    source[:, :, 0, 0] = 1
    source[:, :, 0, 1] = 0.5
    alpha = torch.tensor([[[[1.0, 0.5]]]])
    prediction = torch.zeros_like(source)
    total, parts = smoke_hand_objective(_tensor(prediction, alpha), source, alpha)
    expected_masked = (3 * 1.0 + 3 * 0.5 * 0.5) / (3 * 1.5)
    assert parts["masked_rgb_mae"].item() == pytest.approx(expected_masked)
    assert total.item() == pytest.approx(sum(value.item() for value in parts.values()))


def test_batch_reduction_is_not_the_mean_of_unequal_masks():
    # Equal spatial dimensions permit batching; alpha areas still differ 1:4.
    small_rgb = torch.zeros(1, 3, 2, 2)
    small_rgb[:, :, 0, 0] = 1
    small_alpha = torch.zeros(1, 1, 2, 2)
    small_alpha[:, :, 0, 0] = 1
    small_pred = torch.zeros_like(small_rgb)
    large_rgb = torch.ones(1, 3, 2, 2)
    large_alpha = torch.ones(1, 1, 2, 2)
    large_pred = large_rgb.clone()
    _, small = smoke_hand_objective(_tensor(small_pred, small_alpha), small_rgb, small_alpha)
    _, large = smoke_hand_objective(_tensor(large_pred, large_alpha), large_rgb, large_alpha)
    batch_rgb = torch.cat((small_rgb, large_rgb), dim=0)
    batch_alpha = torch.cat((small_alpha, large_alpha), dim=0)
    batch_pred = torch.cat((small_pred, large_pred), dim=0)
    _, batch = smoke_hand_objective(_tensor(batch_pred, batch_alpha), batch_rgb, batch_alpha)
    per_image_mean = 0.5 * (small["masked_rgb_mae"] + large["masked_rgb_mae"])
    assert batch["masked_rgb_mae"].item() == pytest.approx(1.0 / 5.0)
    assert per_image_mean.item() == pytest.approx(0.5)
    assert batch["masked_rgb_mae"].item() != pytest.approx(per_image_mean.item())


def test_bgr_conversion_uses_the_black_minus_one_convention():
    import numpy as np

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 255
    tensor = bgr_uint8_to_rgb_tensor(image)
    assert tensor.shape == (3, 2, 2)
    assert tensor[2].mean().item() == pytest.approx(1.0)
    assert tensor[0].mean().item() == pytest.approx(-1.0)
    shown = compose(tensor.unsqueeze(0), torch.zeros(1, 1, 2, 2))
    assert shown.mean().item() == pytest.approx(-1.0)


def test_decoder_side_objective_needs_no_source_mask_and_rejects_bad_alpha_range():
    rgb = torch.zeros(1, 3, 2, 2)
    alpha = torch.full((1, 1, 2, 2), 1.1)
    with pytest.raises(ValueError, match="predicted alpha"):
        smoke_hand_objective(_tensor(rgb, alpha), rgb, torch.zeros_like(alpha))


def test_legacy_rgb_checkpoint_is_not_assigned_oracle_alpha():
    assert checkpoint_kind({"out_channels": 3, "state_dict": {}}) == "legacy"
    legacy = {"state_dict": {"final.0.weight": type("Weight", (), {"shape": (3, 128, 4, 4)})()}}
    assert checkpoint_kind(legacy) == "legacy"
