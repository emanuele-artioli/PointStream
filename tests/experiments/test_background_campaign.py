"""Decision rules for background campaign part 2. No encodes."""

from __future__ import annotations

from experiments.modular.background_campaign import (
    background_rate_win,
    choose_setup,
    long_window_gate,
    required_foreground,
)


def test_required_foreground_matches_the_part1_panorama() -> None:
    required = required_foreground(24.58878360131408, 23.153573093720933)
    assert abs(required - 25.204) < 0.01


def test_setup_choice_prefers_the_arm_that_needs_less_foreground() -> None:
    rows = [
        {"representation": "cleaned_video", "qp": 46, "total_bytes": 108_192, "psnr_bg": 31.35},
        {"representation": "registered_panorama", "qp": 46, "total_bytes": 35_763, "psnr_bg": 23.15},
        {"representation": "source", "qp": 46, "total_bytes": 112_295, "psnr_bg": 31.36},
    ]
    choice = choose_setup(rows, 112_295, 24.58878360131408, 21.69)
    assert choice["representation"] == "registered_panorama"
    assert choice["pipeline_setup"] is True
    assert choice["foreground_budget_bytes"] == 112_295 - 35_763


def test_setup_choice_rejects_a_budget_under_8kb() -> None:
    rows = [{"representation": "cleaned_video", "qp": 46, "total_bytes": 108_192, "psnr_bg": 31.35}]
    choice = choose_setup(rows, 112_295, 24.59, 21.69)
    assert choice["fits"] is False
    assert choice["pipeline_setup"] is False


def test_background_rate_win_needs_both_the_court_and_the_budget() -> None:
    assert background_rate_win(31.0, 31.36, 8_000)
    assert not background_rate_win(30.0, 31.36, 80_000)
    assert not background_rate_win(31.3, 31.36, 4_103)


def test_long_window_opens_only_when_the_48_frame_setup_is_close() -> None:
    assert long_window_gate(
        {"fits": True, "foreground_budget_bytes": 80_000, "required_fg_minus_anchor_fg": 3.5}
    )
    assert not long_window_gate(
        {"fits": True, "foreground_budget_bytes": 80_000, "required_fg_minus_anchor_fg": 6.1}
    )
    assert not long_window_gate({"fits": False})
