import hashlib
import json
from pathlib import Path

import pytest

from experiments.generative import gvcrt_receiver_worker as worker


def configuration(tmp_path):
    runtime = tmp_path / '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661' / 'tree'
    runtime.mkdir(parents=True)
    author = tmp_path / 'author'
    author.mkdir()
    (author / '.gvcrt-source.json').write_text('marker')
    wire, output = tmp_path / 'wire', tmp_path / 'output'
    wire.mkdir()
    output.mkdir()
    (wire / 'stream.bin').write_bytes(b'stream')
    (wire / 'manifest.json').write_text('{}')
    files = {}
    for name in ('bridge', 'entropy', 'i_model', 'p_model'):
        p = tmp_path / name
        p.write_bytes(name.encode())
        files[name] = str(p)
    denied = {}
    aliases = []
    for name in sorted(worker.DENIALS - {'aliases'}):
        p = tmp_path / f'protected_{name}'
        p.write_bytes(name.encode())
        a = tmp_path / f'alias_{name}'
        a.symlink_to(p)
        denied[name] = [str(p)]
        aliases.append(str(a))
    denied['aliases'] = aliases
    canary = tmp_path / 'canary'
    canary.mkdir()
    for name in ('write', 'truncate', 'remove', 'rename', 'link'):
        (canary / name).write_bytes(b'canary')
    c = {'canary_dir': str(canary), 'runtime': str(runtime), 'author': str(author), 'wire': str(wire), 'output': str(output),
         **files, 'device_paths': [], 'denied_paths': denied, 'parent_user': 'u', 'parent_net': 'n',
         'visible_gpu': 'GPU-123', 'cpu_seconds': 60}
    return c


def write_config(tmp_path, c):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(c))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def fake_hash(monkeypatch):
    real = worker.sha
    def call(p):
        if p.name == 'bridge':
            return worker.BRIDGE_SHA
        if p.name == 'entropy':
            return worker.ENTROPY_SHA
        if p.name == '.gvcrt-source.json':
            return 'b211cbcc4280acad546a41283ec95c63c7ca95c8d0fd6fdb07af7120932cbe97'
        if p.name in ('i_model', 'p_model'):
            raise AssertionError('probe read a model body')
        return real(p)
    monkeypatch.setattr(worker, 'sha', call)


def test_probe_validates_policy_without_reading_weights(tmp_path, monkeypatch):
    c = configuration(tmp_path)
    fake_hash(monkeypatch)
    path, sha = write_config(tmp_path, c)
    assert worker.load_config(path, sha, probe=True)['visible_gpu'] == 'GPU-123'


def test_probe_rejects_source_alias_in_wire(tmp_path, monkeypatch):
    c = configuration(tmp_path)
    fake_hash(monkeypatch)
    wire = Path(c['wire'])
    (wire / 'stream.bin').unlink()
    (wire / 'stream.bin').symlink_to(c['denied_paths']['source_archive'][0])
    path, sha = write_config(tmp_path, c)
    with pytest.raises(ValueError, match='regular local files'):
        worker.load_config(path, sha, probe=True)


def test_probe_rejects_source_without_a_distinct_alias(tmp_path, monkeypatch):
    c = configuration(tmp_path)
    fake_hash(monkeypatch)
    c['denied_paths']['aliases'] = c['denied_paths']['aliases'][1:]
    path, sha = write_config(tmp_path, c)
    with pytest.raises(ValueError, match='alias'):
        worker.load_config(path, sha, probe=True)


def test_probe_rejects_hidden_hardlink_in_allowed_runtime(tmp_path, monkeypatch):
    import os

    c = configuration(tmp_path)
    fake_hash(monkeypatch)
    os.link(c['denied_paths']['source_archive'][0], Path(c['runtime']) / 'hidden.dat')
    path, sha = write_config(tmp_path, c)
    with pytest.raises(ValueError, match='protected hardlink'):
        worker.load_config(path, sha, probe=True)


def test_landlock_requires_truncate_control_and_only_output_has_mutation_rights(tmp_path):
    with pytest.raises(RuntimeError, match='ABI >=3'):
        worker.handled_rights(2)
    assert worker.handled_rights(3) & worker.TRUNCATE
    assert not worker.handled_rights(3) & worker.IOCTL_DEV
    assert worker.handled_rights(5) & worker.IOCTL_DEV
    source = tmp_path / 'source'
    source.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    assert worker.allowed_rights(source, output, 3) == worker.READ_FILE | worker.READ_DIR
    rights = worker.allowed_rights(output, output, 3)
    for bit in (worker.WRITE_FILE, worker.MAKE_REG, worker.MAKE_DIR, worker.REMOVE_FILE,
                worker.REMOVE_DIR, worker.TRUNCATE, worker.REFER):
        assert rights & bit
    for bit in (worker.EXECUTE, worker.MAKE_CHAR, worker.MAKE_BLOCK, worker.MAKE_SOCK, worker.MAKE_FIFO):
        assert not rights & bit


def test_system_library_directory_cannot_share_protected_inode_device(tmp_path):
    source = tmp_path / 'protected'
    source.write_bytes(b'source')
    st = source.stat()
    with pytest.raises(ValueError, match='shares a filesystem'):
        worker.reject_shared_system_filesystem([tmp_path], {(st.st_dev, st.st_ino)})


def test_direct_receiver_import_does_not_load_pointstream_or_torch():
    import subprocess
    import sys

    source = Path(worker.__file__).resolve()
    code = "import runpy,sys;runpy.run_path(sys.argv[1],run_name='receiver_import_test');assert 'src' not in sys.modules;assert 'torch' not in sys.modules;assert 'experiments' not in sys.modules"
    subprocess.run([sys.executable, '-I', '-c', code, str(source)], check=True, timeout=10)


@pytest.mark.parametrize('mode', ['probe', 'decode'])
def test_entropy_loaded_by_file_only_after_isolation_for_decode(tmp_path, monkeypatch, mode):
    import sys
    from types import SimpleNamespace

    events = []
    c = {'output': str(tmp_path), 'bridge': '/bridge.py', 'entropy': '/private/entropy.so',
         'author': '/author', 'i_model': '/i', 'p_model': '/p', 'wire': '/wire',
         'denied_paths': {'source_archive': ['/secret']}}
    monkeypatch.setattr(sys, 'argv', ['worker', mode, '--config', '/config', '--config-sha256', 'a'])
    monkeypatch.setattr(worker, 'load_config', lambda *args, **kwargs: c)
    monkeypatch.setattr(worker, 'bootstrap', lambda _: events.append('isolation') or {})
    monkeypatch.setattr(worker, 'cuda_probe', lambda: events.append('probe') or {})
    monkeypatch.delitem(sys.modules, 'MLCodec_extensions_cpp', raising=False)

    class Loader:
        def exec_module(self, module):
            if module.name == 'MLCodec_extensions_cpp':
                assert sys.modules['MLCodec_extensions_cpp'] is module
                events.append('entropy')
            else:
                events.append('bridge_import')

    def spec(name, path):
        if name == 'MLCodec_extensions_cpp':
            assert path == c['entropy']
        return SimpleNamespace(name=name, loader=Loader())

    def module_from_spec(s):
        return SimpleNamespace(name=s.name, decode=lambda _: events.append('decode'))

    monkeypatch.setattr(worker.importlib.util, 'spec_from_file_location', spec)
    monkeypatch.setattr(worker.importlib.util, 'module_from_spec', module_from_spec)
    assert worker.main() == 0
    assert events == (['bridge_import', 'isolation', 'probe'] if mode == 'probe'
                      else ['bridge_import', 'isolation', 'entropy', 'decode'])
    if mode == 'decode':
        monkeypatch.delitem(sys.modules, 'MLCodec_extensions_cpp')
