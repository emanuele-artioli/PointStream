"""Cost admission must use measured worker and native campaign clocks."""
import pytest

from experiments.tennis.player_requalification import reference_smoke_cost_gate


def test_reference_cost_gate_uses_both_caps() -> None:
    receipt = {"load_seconds": 24.0}
    timing = {"worker_wall_seconds": 100.0}
    gate = reference_smoke_cost_gate(receipt, timing, {"campaign_clock": {"budget_end": 600}}, 100)
    assert gate["forecast_seconds"] == 310
    with pytest.raises(ValueError, match="forecast"):
        reference_smoke_cost_gate(receipt, timing, {"campaign_clock": {"budget_end": 469}}, 100)
    with pytest.raises(ValueError, match="forecast"):
        reference_smoke_cost_gate(receipt, {"worker_wall_seconds": 200}, {"campaign_clock": {"budget_end": 1000}}, 100)
    with pytest.raises(ValueError, match="shorter"):
        reference_smoke_cost_gate(receipt, {"worker_wall_seconds": 20}, {"campaign_clock": {"budget_end": 1000}}, 100)
