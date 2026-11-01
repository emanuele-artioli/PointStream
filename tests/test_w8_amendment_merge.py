import hashlib
import json

import pytest

from experiments.tennis import w8_amendment_merge as amendment


def test_screen_retains_prior_anchors_and_uses_actual_target(tmp_path, monkeypatch):
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'manifest_sha256': amendment.SELECTION_SHA, 'spec_sha256': 'spec'}))
    monkeypatch.setattr(amendment, 'MANIFEST', tmp_path / 'hashes.json')
    amendment.MANIFEST.write_text('{"results": {}}')
    anchors = {f: [{'kbps': 10., 'lpips': .3, 'dists': .3},
                   {'kbps': 45.82333333333333, 'lpips': .3, 'dists': .3}]
               for f in amendment.w8_oracle.FAMILIES}
    prior = {'archive': 'prior'}
    monkeypatch.setattr(amendment.w8_oracle, 'accepted_nad000', lambda *_: ({'lpips': .1, 'dists': .1}, anchors))
    trial_point = {'stream_sha256': amendment.STREAM_SHA, 'crf': 67,
                   'rate_kbps': 29804 * 8 / 4.8 / 1000, 'foreground_lpips': .2, 'foreground_dists': .2}
    monkeypatch.setattr(amendment.w9_repair_intervals, 'verified_merge', lambda *args: {'by_policy': {'s2_o20': [trial_point]}})
    trial = {'audit': str(audit), 'audit_sha256': hashlib.sha256(audit.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='normalized spec'):
        amendment.screen(tmp_path, prior, trial, 'different')
    with pytest.raises(ValueError):
        amendment.screen(tmp_path, prior, {**trial, 'audit_sha256': '0' * 64}, 'spec')
    trial_point['stream_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='selected trial'):
        amendment.screen(tmp_path, prior, trial, 'spec')
    trial_point['stream_sha256'] = amendment.STREAM_SHA
    result = amendment.screen(tmp_path, prior, trial, 'spec')
    assert result['gate']['decision'] == 'unresolved'  # Other families still lack the bracket.
    assert result['target_kbps'] == 46.315062222222224
    assert len(result['anchors']['s2_o20']) == 3
    assert len(result['anchors']['s4_o0']) == 2


def test_candidate_normalizes_and_expired_deadline_blocks_submission():
    from pathlib import Path

    from experiments.jobs import inbox

    path = Path(__file__).resolve().parents[1] / 'docs/experiments/w8-inputs/amendment-20261101/score-spec-candidate.json'
    spec = json.loads(path.read_text())
    assert inbox.validate_spec(spec) == spec
    with pytest.raises(Exception, match='deadline has already expired'):
        inbox.validate_spec(spec, now=1_700_000_000)
    assert spec['entrypoint'] == ['-m', 'experiments.tennis.w8_score_job']
    assert spec['single_cpu_campaign'] is True
    assert spec['cpu_threads'] == 1


def test_second_candidate_keeps_first_identities_and_allocations():
    from pathlib import Path

    from experiments.jobs import inbox

    root = Path(__file__).resolve().parents[1] / 'docs/experiments/w8-inputs/amendment-20261101'
    first = json.loads((root / 'score-spec-candidate.json').read_text())
    second = json.loads((root / 'score-spec-second-candidate.json').read_text())
    assert inbox.validate_spec(second) == second
    for key in ('inputs', 'staged_inputs', 'environment', 'arguments', 'hosts', 'gpu_models',
                'cpu_threads', 'single_cpu_campaign', 'defer_failed_publication'):
        assert second[key] == first[key]
    assert (second['smoke']['seconds'], second['full']['seconds'], second['validator_seconds'],
            second['budget_seconds']) == (20, 175, 5, 212)
    assert second['deadline'] == '2026-11-01T07:56:32.398155Z'
