import numpy as np
import pytest

from experiments.tennis.k6b_ideal_player import check_images, control_error, decision


def test_decision_requires_one_global_ideal_policy_and_sensitivity():
    anchor = {'lpips': .2, 'dists': .1}
    scores = {'lpips': np.array([[.3], [.3], [.19], [.21]]),
              'dists': np.array([[.2], [.2], [.11], [.09]])}
    assert decision(scores, anchor)['verdict'] == 'MIXED/UNRESOLVED'
    scores['dists'][2] = .09
    assert decision(scores, anchor)['verdict'] == 'HEADROOM'
    scores['lpips'][2] = .1995
    assert decision(scores, anchor)['verdict'] == 'MIXED/UNRESOLVED'
    scores['lpips'][2:] = .21
    scores['dists'][2:] = .11
    assert decision(scores, anchor)['verdict'] == 'NO-GO'


def test_actual_candidate_does_not_promote_oracle_and_nonfinite_rejected():
    s = {m: np.array([[.3], [.01], [.3], [.3]]) for m in ('lpips', 'dists')}
    assert decision(s, {'lpips': .2, 'dists': .2})['verdict'] == 'NO-GO'
    s['dists'][2] = np.nan
    with pytest.raises(ValueError, match='finite'):
        decision(s, {'lpips': .2, 'dists': .2})


def test_controls_are_per_frame_not_just_means():
    expected = {m: [[.1, .3], [.2, .4]] for m in ('lpips', 'dists')}
    s = {m: np.array([[.1, .3], [.2, .4], [0, 0], [0, 0]]) for m in expected}
    assert max(control_error(s, expected, [0, 1]).values()) == 0
    s['lpips'][0] = [.3, .1]
    with pytest.raises(ValueError, match='historical'):
        control_error(s, expected, [0, 1])


def test_original_shape_required():
    with pytest.raises(ValueError, match='resolution'):
        check_images(np.zeros((5, 12, 12, 3), dtype=np.uint8), [''] * 5)


@pytest.mark.parametrize('mutation', ['none', 'input', 'cost', 'memory', 'coverage'])
def test_native_validator_rejects_changed_identity_and_resource_contract(tmp_path, monkeypatch, mutation):
    import io
    import json
    import tarfile
    from experiments.tennis.k6b_ideal_player import POLICIES, WEIGHTS, validate

    stage = tmp_path / 'smoke'
    stage.mkdir()
    s = {'ticks': [0, 3, 6], 'policies': list(POLICIES), 'cuda': True, 'gpu': 'RTX A6000',
         'timings': {p: [1., 1., 1.] for p in POLICIES}, 'peak_allocated_mib': 3500,
         'peak_reserved_mib': 4000, 'control_errors': {'lpips': 0., 'dists': 0.},
         'weight_sha256': WEIGHTS, 'input_sha256': 'input',
         'hashes': {str(t): ['h'] * 5 for t in (0, 3, 6)}}
    spec = {'full': {'seconds': 550}, 'budget_seconds': 1100,
            'staged_inputs': [{'name': 'images', 'sha256': 'input'}],
            'scale': {'judged': {'smoke': 3, 'full': 0}}}
    if mutation == 'input':
        s['input_sha256'] = 'changed'
    elif mutation == 'cost':
        spec['full']['seconds'] = 400
    elif mutation == 'memory':
        s['peak_reserved_mib'] = 8100
    elif mutation == 'coverage':
        s['hashes'].pop('6')
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    (stage / 'oracle-summary.json').write_text(json.dumps(s))
    (stage / 'execution.json').write_text(json.dumps({'seconds': 15}))
    (stage / 'staging.json').write_text(json.dumps({'publication_seconds': .1}))
    with tarfile.open(stage / 'published.tar', 'w') as tar:
        data = io.BytesIO()
        np.savez(data, ticks=[0, 3, 6], lpips=np.zeros((4, 3)), dists=np.zeros((4, 3)))
        for name, payload in [('publish/scores.npz', data.getvalue()), ('publish/oracle.json', json.dumps(s).encode())]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    result = tmp_path / 'validation.json'
    monkeypatch.setenv('PS_VALIDATION_PATH', str(result))
    if mutation == 'none':
        validate()
        assert json.loads(result.read_text())['passed']
    else:
        with pytest.raises(SystemExit):
            validate()
        assert not json.loads(result.read_text())['passed']
