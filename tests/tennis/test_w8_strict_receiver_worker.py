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
