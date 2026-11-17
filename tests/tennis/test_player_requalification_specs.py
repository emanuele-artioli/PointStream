"""Check that the frozen paired replay can use the actual fleet contract."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from experiments.tennis.player_requalification_specs import build_motion_spec, build_specs


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


def test_motion_requests_are_single_player_and_bound_allocation():
    folder = ROOT / "docs/experiments/player-motion-20261116"
    proposal = json.loads((folder / 'proposal.json').read_text())
    assert proposal['status'] == 'proposed_not_released'
    jobs = [build_motion_spec(folder / f'player-{i}.json', folder / 'inputs.json',
                             f'/home/itec/emanuele/Datasets/pointstream-data/tennis/proposed/player-{i}.json',
                             (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()) for i in (2, 1)]
    assert sum(j['budget_seconds'] for j in jobs) == proposal['authority']['max_gpu_seconds'] == 4200
    assert sum(r['cpu_lifecycle_cap_seconds'] + r['external_cpu_cap_seconds']
               for r in proposal['requests']) == proposal['authority']['max_cpu_seconds'] == 7200
    assert [r['player'] for r in proposal['requests']] == [2, 1]
    for job in jobs:
        assert job['single_cpu_campaign'] and job['cpu_threads'] == 1
        assert job['scale'] == {'native_frames': {'smoke': 44, 'full': 240}}
        assert job['smoke']['seconds'] == 450 and job['full']['seconds'] == 1500
        assert job['arguments'][1:3] == ['--profile', 'tennis_only']
        assert job['arguments'][job['arguments'].index('--max-units') + 1] == '1'
        assert 'match' not in {item['name'] for item in job['staged_inputs']}
        assert job['validator'][-1] == 'validate-stage'
    assert jobs[0]['inputs'][0]['sha256'] != jobs[1]['inputs'][0]['sha256']
    assert jobs[0]['inputs'][1:] == jobs[1]['inputs'][1:]


def test_motion_requests_reject_extra_players_or_match_resources(tmp_path):
    folder = ROOT / "docs/experiments/player-motion-20261116"
    config = json.loads((folder / 'player-2.json').read_text())
    config['units'] *= 2
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    deadline = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
    with pytest.raises(ValueError, match='one frozen'):
        build_motion_spec(path, folder / 'inputs.json', '/home/itec/emanuele/Datasets/test.json', deadline)
    with pytest.raises(ValueError, match='ten-input'):
        build_motion_spec(folder / 'player-2.json', FROZEN / 'inputs.json',
                          '/home/itec/emanuele/Datasets/test.json', deadline)


def test_motion_extended_policy_drives_matching_caps_and_rejects_unknown(tmp_path):
    folder = ROOT / 'docs/experiments/player-motion-20261116'
    config = json.loads((folder / 'player-2.json').read_text())
    config['motion_admission_policy'] = 'extended-1800-v1'
    path = tmp_path / 'extended.json'
    path.write_text(json.dumps(config))
    deadline = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
    spec = build_motion_spec(path, folder / 'inputs.json', '/home/itec/emanuele/Datasets/test.json', deadline)
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['budget_seconds'], spec['validator_seconds']) == (450, 1800, 2400, 60)
    config['motion_admission_policy'] = 'unknown'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='unknown motion'):
        build_motion_spec(path, folder / 'inputs.json', '/home/itec/emanuele/Datasets/test.json', deadline)
