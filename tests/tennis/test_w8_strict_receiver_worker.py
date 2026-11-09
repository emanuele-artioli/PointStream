import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tennis import w8_strict_receiver_worker as worker


def test_preload_masks_legacy_positional_command(monkeypatch):
    original_import = builtins.__import__
    seen = []
    def tracked(name, *args, **kwargs):
        if name.startswith('experiments.tennis'):
            seen.append(tuple(sys.argv))
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', tracked)
    monkeypatch.setattr(sys, 'argv', ['worker.py', 'worker', '--config', 'x'])
    result = worker.preload()
    assert len(result) == 4 and seen
    assert all(value == ('worker.py',) for value in seen)
    assert sys.argv == ['worker.py', 'worker', '--config', 'x']


@pytest.mark.parametrize('kind,steps', [('generate', None), ('cost_probe', 30), ('cost_probe', 6)])
def test_worker_rejects_unapproved_generation_modes_before_any_model(monkeypatch, tmp_path, kind, steps):
    bootstrap = SimpleNamespace(load_config=lambda *_: {'worker_kind': kind, 'probe_steps': steps, 'allow_paths': [str(tmp_path)]})
    monkeypatch.setattr(worker, 'preload', lambda: (None, None, bootstrap, None))
    monkeypatch.setattr(worker, 'validate_source_origins', lambda _: {'receiver': 'receiver.py'})
    monkeypatch.setattr(worker.sys, 'platform', 'linux')
    monkeypatch.setattr(worker.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    with pytest.raises(ValueError, match='only'):
        worker.launch(Path('/config'), 'sha', tmp_path / 'loader', 'sha', tmp_path, tmp_path)


def test_extra_threads_and_descriptors_fail_closed(monkeypatch):
    monkeypatch.setattr(worker.os, 'listdir', lambda p: ['1', '2'] if p.endswith('/task') else [])
    with pytest.raises(RuntimeError, match='single-threaded'):
        worker.clean_process()
    monkeypatch.setattr(worker.os, 'listdir', lambda p: ['1'] if p.endswith('/task') else ['0', '1', '2', '3'])
    original_exists = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda p: True if str(p) == '/proc/self/fd/3' else original_exists(p))
    with pytest.raises(RuntimeError, match='descriptor'):
        worker.clean_process()


def test_source_origins_rejects_preloaded_project_module_outside_root(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    foreign = tmp_path / 'shadow.py'
    foreign.write_text('value = 1\n')
    monkeypatch.chdir(root)
    worker.preload()
    monkeypatch.setitem(sys.modules, 'experiments.tennis.shadow', SimpleNamespace(__file__=str(foreign)))
    with pytest.raises(ValueError, match='outside authenticated source root'):
        worker.validate_source_origins(root)


def test_qualification_context_keeps_recipe_and_cannot_pass_candidate(tmp_path):
    import json
    import numpy as np
    from experiments.tennis import w8_receiver_generate as generation

    class Renderer:
        def render(self, reference, poses, steps, seed):
            assert reference == 'reference' and poses.tolist() == list(range(24))
            assert steps == 30 and seed == 123 * 1009
            return list(range(24))

    unit = {'run': np.zeros(240, dtype=int), 't': np.arange(240), 'pose': np.arange(240),
            'reference': 'reference', 'judged': np.arange(240) % 3 == 0}
    ticks, crops = generation.render_qualification_context(Renderer(), unit, 123, 0)
    assert ticks == crops == list(range(0, 24, 3))
    (tmp_path / 'sealed.json').write_text(json.dumps({'scope': 'context_qualification_only',
                                                       'recipe': {'steps': 30}}))
    with pytest.raises(ValueError, match='scope'):
        generation.validate_sealed(tmp_path, {})
    unit['run'][0] = 1
    with pytest.raises(ValueError, match='first 24'):
        generation.render_qualification_context(Renderer(), unit, 123, 0)


def test_qualification_rejects_same_directory_wrong_crop_or_seal(tmp_path):
    import json
    from experiments.tennis import w8_input_job as fixed, w8_receiver_finish as finish

    unit = fixed.UNITS[0]
    canonical_crop = tmp_path / f'{unit}@match5.npz'
    wrong_crop = tmp_path / 'other.npz'
    canonical_seal = tmp_path / 'sealed.json'
    wrong_seal = tmp_path / 'other.json'
    canonical_crop.write_bytes(b'crop')
    wrong_crop.write_bytes(b'other')
    canonical_seal.write_text(json.dumps({}))
    wrong_seal.write_text(json.dumps({}))
    paths = {f'crop:{unit}': wrong_crop, f'generation_seal:{unit}': canonical_seal}
    with pytest.raises(ValueError, match='canonical paths'):
        finish._generation_seals(paths, qualification=True)
    paths = {f'crop:{unit}': canonical_crop, f'generation_seal:{unit}': wrong_seal}
    with pytest.raises(ValueError, match='canonical paths'):
        finish._generation_seals(paths, qualification=True)


@pytest.mark.parametrize('kind', ['generate', 'context_qualification'])
def test_witness_modes_require_opt_in_materialization_and_clock(monkeypatch, tmp_path, kind):
    bootstrap = SimpleNamespace(load_config=lambda *_: {'worker_kind': kind, 'allow_paths': [str(tmp_path)]})
    monkeypatch.setattr(worker, 'preload', lambda: (None, None, bootstrap, None))
    monkeypatch.setattr(worker, 'validate_source_origins', lambda _: {'receiver': 'receiver.py'})
    monkeypatch.setattr(worker.sys, 'platform', 'linux')
    monkeypatch.setattr(worker.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    with pytest.raises(ValueError, match='materialized source and clock'):
        worker.launch(Path('/config'), 'sha', tmp_path / 'loader', 'sha', tmp_path, tmp_path,
                      witness_mode=kind)
