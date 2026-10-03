from __future__ import annotations

import pytest
import torch

from demo.models.hand_objective import smoke_hand_objective


def _tensor(rgb, alpha):
    return torch.cat((rgb, alpha), dim=1)


def test_exact_rgb_and_alpha_has_zero_loss():
    rgb = torch.rand(1, 3, 4, 5) * 2 - 1
    alpha = torch.zeros(1, 1, 4, 5)
    alpha[:, :, 1:3, 1:4] = 1
    total, parts = smoke_hand_objective(_tensor(rgb, alpha), rgb, alpha)
    assert total.item() == pytest.approx(0.0, abs=1e-7)
    assert all(value.item() == pytest.approx(0.0, abs=1e-7) for value in parts.values())


def test_all_zero_prediction_alpha_is_penalized_inside_and_gradient_increases_it():
    rgb = torch.zeros(1, 3, 2, 2)
    target_alpha = torch.ones(1, 1, 2, 2)
    output = _tensor(rgb.clone(), torch.zeros_like(target_alpha)).requires_grad_()
    total, parts = smoke_hand_objective(output, rgb, target_alpha)
    total.backward()
    assert parts["balanced_alpha_mae"].item() > 0
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


def test_unmatted_source_is_masked_once_and_empty_or_full_masks_stay_finite():
    source = torch.ones(1, 3, 2, 2) * 0.5
    empty = torch.zeros(1, 1, 2, 2)
    wrong_rgb = torch.ones_like(source) * -0.5
    out = _tensor(wrong_rgb, empty)
    total, parts = smoke_hand_objective(out, source, empty)
    assert torch.isfinite(total)
    assert parts["masked_rgb_mae"].item() == 0
    assert parts["composite_mae"].item() == 0
    assert parts["balanced_alpha_mae"].item() == 0

    full = torch.ones_like(empty)
    total, _ = smoke_hand_objective(_tensor(source, full), source, full)
    assert torch.isfinite(total)


def test_rgb_outside_zero_alpha_cannot_change_composite_but_leak_is_scored():
    source = torch.zeros(1, 3, 2, 2)
    target_alpha = torch.zeros(1, 1, 2, 2)
    target_alpha[:, :, 0, 0] = 1
    pred_a = target_alpha.clone()
    rgb_a = source.clone()
    rgb_b = source.clone()
    rgb_b[:, :, 1, 1] = 1
    a, parts_a = smoke_hand_objective(_tensor(rgb_a, pred_a), source, target_alpha)
    b, parts_b = smoke_hand_objective(_tensor(rgb_b, pred_a), source, target_alpha)
    assert parts_a["composite_mae"].item() == pytest.approx(parts_b["composite_mae"].item())
    assert parts_a["masked_rgb_mae"].item() == pytest.approx(parts_b["masked_rgb_mae"].item())
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

    tiny_alpha = torch.tensor([[[[0.1]]]])
    tiny_source = torch.ones(1, 3, 1, 1)
    tiny_output = _tensor(torch.zeros_like(tiny_source), torch.zeros_like(tiny_alpha))
    _, tiny_parts = smoke_hand_objective(tiny_output, tiny_source, tiny_alpha)
    assert tiny_parts["masked_rgb_mae"].item() == pytest.approx(1.0)


def test_decoder_side_objective_needs_no_source_mask_and_rejects_bad_alpha_range():
    rgb = torch.zeros(1, 3, 2, 2)
    alpha = torch.full((1, 1, 2, 2), 1.1)
    with pytest.raises(ValueError, match="predicted alpha"):
        smoke_hand_objective(_tensor(rgb, alpha), rgb, torch.zeros_like(alpha))

    out_of_range = torch.full_like(rgb, 1.1)
    with pytest.raises(ValueError, match="RGB"):
        smoke_hand_objective(_tensor(out_of_range, torch.zeros_like(alpha)), rgb, torch.zeros_like(alpha))
