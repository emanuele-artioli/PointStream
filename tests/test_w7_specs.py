"""W7 spec builder: shape of the emitted specs and the recorded per-window GPU estimate."""
import json
from pathlib import Path

from experiments.tennis import w7_specs as sp
from experiments.tennis import window_manifest as wm

SELECTION = Path(__file__).resolve().parents[1] / 'docs/experiments/w7-specs/selection.json'


def test_native_base_specs():
    windows = wm.from_selection(json.loads(SELECTION.read_text()))
    manifest = (sp.W7_DIR + '/windows.json', 'a' * 64)
    smoke, full = sp.native_base(windows, manifest, 24), sp.native_base(windows, manifest, 240)
    assert [i['name'] for i in smoke['staged_inputs']] == ['windows', 'al028']
    assert [i['name'] for i in full['staged_inputs']] == ['windows'] + [w.name for w in windows]
    for spec in (smoke, full):
        assert spec['verify_staged_inputs'] is True and spec['hosts'] == ['gpu3', 'gpu5', 'gpu6']
        assert spec['smoke']['seconds'] <= 600 and spec['full']['seconds'] <= 2600
        assert all(i['path'].startswith(sp.DATA + '/') for i in spec['staged_inputs'])
        assert spec['arguments'].count('--window') == spec['arguments'].count('--source')


def test_w1_stage_is_single_window_and_gpu_estimate_recorded():
    w = wm.from_selection(json.loads(SELECTION.read_text()))[0]
    spec = sp.w1_window_stage('experiments.tennis.w1_score', w, ('/x/windows.json', 'b' * 64), ('/x/state.json', 'c' * 64), ['--judged', '{judged}'],
                              {}, gpu=True, full_seconds=2400)
    assert spec['arguments'][:6] == ['--windows-file', '{staged:windows}', '--window-state', '{staged:state}', '--window', 'al028']
    assert spec['gpu_models'] == ['RTX A6000', 'RTX 6000 Ada'] and spec['full']['seconds'] == 2400
    assert sp.per_window_gpu_seconds() == 4654
