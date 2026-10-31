import pytest

from experiments.tennis import w7_specs, w7_tier2_fit_specs as fit, window_manifest as wm


def windows():
    return [wm.WindowSpec(f'nad{100 + i}', 'nadal_kyrgios', str(100 + i), 0, 240, 50, 'a' * 64, 'b' * 64)
            for i in range(3)]


def test_prepare_replaces_only_fit_clip_inputs_and_windows():
    template = {'entrypoint': ['-m', 'experiments.tennis.k6b'],
                'arguments': ['prepare-match', '--windows', 'old/1@120', '--sources', '{staged:c0}'],
                'staged_inputs': [{'name': n, 'path': '/old', 'sha256': 'f' * 64, 'extract': n == 'inputs'}
                                  for n in ('inputs', 'pose', 'c0', 'c1', 'c2')],
                'smoke': {'seconds': 600}, 'deadline_epoch': 1}
    root = w7_specs.DATA + '/tennis/w7-tier2/fit'
    spec = fit.modify(template, 'prepare', windows(), (root + '/inputs.tar', 'c' * 64), root, '2026-11-02T12:00:00+00:00')
    assert spec['arguments'][2:5] == [w.id for w in windows()]
    assert spec['staged_inputs'][1]['path'] == '/old'
    assert spec['staged_inputs'][2]['path'] == root + '/sources/nadal_kyrgios__100.mp4'
    assert spec['staged_inputs'][0]['extract'] is True and 'deadline_epoch' not in spec
    assert template['arguments'][2] == 'old/1@120'


def test_fit_keeps_recipe_and_replaces_match_and_fit_data():
    template = {'entrypoint': ['-m', 'experiments.tennis.k6b_fit'],
                'arguments': ['fit', '--minutes', '10', '--rank', '4', '--quantize', '8', '--also-unquantized',
                              '--base', 'tennis-base@fit-100.pt', '--match', 'alcaraz_hurkacz', '--name', 'ah-C'],
                'staged_inputs': [{'name': 'fitdata', 'path': '/old', 'sha256': 'f' * 64}], 'smoke': {'seconds': 600}}
    spec = fit.modify(template, 'fit', windows(), ('/data/fit.tar', 'c' * 64), '/data', '2026-11-02T12:00:00+00:00')
    assert spec['arguments'][spec['arguments'].index('--match') + 1] == 'nadal_kyrgios'
    assert spec['arguments'][spec['arguments'].index('--name') + 1] == 'nk-C'
    assert spec['staged_inputs'][0]['sha256'] == 'c' * 64
    template['arguments'][template['arguments'].index('--rank') + 1] = '8'
    with pytest.raises(ValueError, match='changed frozen recipe'):
        fit.modify(template, 'fit', windows(), ('/data/fit.tar', 'c' * 64), '/data', '2026-11-02T12:00:00+00:00')
