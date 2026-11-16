"""Check that the frozen paired replay can use the actual fleet contract."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from experiments.tennis.player_requalification_specs import build_specs


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs/experiments/player-requalification-20261116"


def specs(inventory: Path = FROZEN / "inputs.json"):
    return build_specs(FROZEN / "config.json", inventory,
                       "/home/itec/emanuele/Datasets/pointstream-data/tennis/proposed/config.json",
                       (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat())


def test_frozen_profiles_have_matching_inputs_and_bounded_stages():
    jobs = specs()
    assert set(jobs) == {"original", "tennis_only", "match_adapted"}
    assert sum(j["budget_seconds"] for j in jobs.values()) == 5400
    first = jobs["original"]
    for name, job in jobs.items():
        assert job["inputs"] == first["inputs"]
        assert job["environment"] == first["environment"]
        assert job["arguments"][2] == name
        assert job["smoke"]["seconds"] <= 600
        assert job["full"]["seconds"] <= 2700
        assert job["budget_seconds"] <= 2700
        assert job["scale"] == {"units": {"smoke": 1, "full": 2}}
        # Validator arguments are not expanded by the fleet. It must recover
        # the actual staged paths and scale from the stage dispatch receipt.
        assert job["validator"][-1] == "validate-stage"
        assert not any("{staged:" in arg for arg in job["validator"])


def test_frozen_conditioning_matches_independently_authenticated_members():
    config = json.loads((FROZEN / "config.json").read_text())
    audit = json.loads((ROOT / "docs/experiments/w8-inputs/receiver-20261101/player-conditioning-byte-audit-20261116.json").read_text())
    assert config["source"]["native_fps"] == [50, 1]
    for planned, checked in zip(config["units"], audit["unit_data"], strict=True):
        assert planned["unit"] == checked["unit"]
        assert planned["sha256"] == checked["sha256"]
        assert planned["path"] == "staged:conditioning/" + checked["member"]
        assert planned["start_t"] == 0 and planned["run"] == 0
        assert checked["ticks"]["first_24"] == list(range(24))


def test_missing_staging_input_cannot_produce_requests(tmp_path):
    inventory = json.loads((FROZEN / "inputs.json").read_text())
    inventory["inputs"].pop()
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="eleven-input"):
        specs(path)
