import hashlib
import io
import json
import struct
import tarfile

import numpy as np
import pytest

from experiments.tennis import w8_receiver_native_bootstrap as native
from experiments.tennis import w8_receiver_score as score


def test_native_ivf_header_and_one_exact_command():
    header = b'DKIF' + struct.pack('<HH4sHHIIII', 0, 32, b'AV01', 480, 272, 50, 1, 240, 0)
    native.validate_ivf_header(header)
    with pytest.raises(ValueError, match='format'):
        native.validate_ivf_header(header[:12] + struct.pack('<H', 270) + header[14:])
    policy = native._audit_policy(['/opt/ffmpeg', '-nostdin'])
    with pytest.raises(PermissionError):
        policy('subprocess.Popen', ('/opt/other', ['/opt/other'], None, None))
    policy('subprocess.Popen', ('/opt/ffmpeg', ['/opt/ffmpeg', '-nostdin'], None, None))
    with pytest.raises(PermissionError):
        policy('subprocess.Popen', ('/opt/ffmpeg', ['/opt/ffmpeg', '-nostdin'], None, None))
    with pytest.raises(PermissionError):
        policy('os.exec', ())


def test_historical_regions_verify_every_tick_and_archive_member(tmp_path):
    zeros = np.zeros((80, 259200), dtype=np.uint8)
    raw = io.BytesIO()
    np.savez_compressed(raw, ticks=score.finish.TICKS, r=zeros, s=zeros)
    blob = raw.getvalue()
    frame_sha = hashlib.sha256(zeros[0].tobytes()).hexdigest()
    meta = {'ticks': score.finish.TICKS.tolist(), 'matches_k6_record': True,
            'masks_sha256': hashlib.sha256(blob).hexdigest(),
            'hashes': {str(t): {'r': frame_sha, 's': frame_sha} for t in score.finish.TICKS}}
    path = tmp_path / 'masks.tar'
    def make(meta_value):
        with tarfile.open(path, 'w') as archive:
            for name, data in [('publish/masks.json', json.dumps(meta_value).encode()), ('publish/masks.npz', blob)]:
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    make(meta)
    result, _ = score._qualify_regions(path)
    assert result.shape == zeros.shape
    meta['hashes']['237']['s'] = 'bad'
    make(meta)
    with pytest.raises(ValueError, match='tick 237'):
        score._qualify_regions(path)


def test_score_does_not_open_original_inputs_before_candidate_seal(monkeypatch, tmp_path):
    monkeypatch.setattr(score.sys, 'platform', 'linux')
    monkeypatch.setattr(score.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    monkeypatch.setattr(score.resource, 'getrlimit', lambda *_: (2, 2))
    receipt = tmp_path / 'receipt.json'
    receipt.write_text('{}')
    called = []
    def fake_inputs(_config, names):
        called.append(names)
        assert names == set(score.finish.CANDIDATE_KEYS)
        return {name: tmp_path / 'candidate.npy' for name in names}
    monkeypatch.setattr(score.finish, '_inputs', fake_inputs)
    def reject(_paths):
        raise ValueError('candidate composite not sealed')
    monkeypatch.setattr(score.finish, 'validate_candidate_seals', reject)
    names = set(score.finish.CANDIDATE_KEYS) | {'source_rgb', 'receiver', 'region_archive', 'judge_archive', 'lpips', 'dists'}
    with pytest.raises(ValueError, match='not sealed'):
        score.run({'cpu_seconds': 2, 'files': {name: {} for name in names}})
    assert len(called) == 1


@pytest.mark.parametrize(('kind', 'function'), [('sam', 'run_sam_isolated'),
                                               ('rgb', 'run_conversion_isolated'),
                                               ('compose', 'run_compose_isolated')])
def test_stdlib_bootstrap_dispatches_finish_workers(monkeypatch, kind, function):
    from experiments.tennis import w8_receiver_bootstrap as bootstrap
    from experiments.tennis import w8_receiver_finish as finish

    config = {'worker_kind': kind, 'cpu_seconds': 5}
    calls = []
    monkeypatch.setattr(bootstrap, 'load_config', lambda *_: config)
    monkeypatch.setattr(bootstrap, 'bootstrap', lambda _: {'test': 'isolated'})
    monkeypatch.setattr(bootstrap.resource, 'setrlimit', lambda *_: None)
    monkeypatch.setattr(finish, function, lambda supplied, isolation: calls.append((supplied, isolation)))
    monkeypatch.setattr('sys.argv', ['worker', '--config', '/reviewed.json', '--config-sha256', 'a'*64,
                                   '--cpu-seconds', '5'])
    assert bootstrap.main() == 0
    assert calls == [(config, {'test': 'isolated'})]


def test_sam_source_mapping_rejects_unknown_and_tampered_modules(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys

    from experiments.tennis import w8_receiver_finish as finish

    path = tmp_path / 'sam3_model.py'
    path.write_text('value = 1\n')
    name = 'sam3.codex_fake_module'
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(path)))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert finish._sam_module_closure({str(path): sha})[name] == str(path)
    path.write_text('value = 2\n')
    with pytest.raises(ValueError, match='unverified SAM module'):
        finish._sam_module_closure({str(path): sha})
    with pytest.raises(PermissionError, match='bytecode'):
        finish._deny_sam_bytecode('open', ('/some/sam3/__pycache__/module.pyc', 'r', 0))


def test_score_bootstrap_sets_bound_before_exec(tmp_path, monkeypatch):
    import sys

    from experiments.tennis import w8_receiver_score_bootstrap as bootstrap

    path = tmp_path / 'config.json'
    data = json.dumps({'cpu_seconds': 7, 'child_cpu_seconds': 11}).encode()
    path.write_bytes(data)
    captured = []
    monkeypatch.setattr(bootstrap.resource, 'setrlimit', lambda _which, limits: captured.append(('limit', limits)))
    monkeypatch.setattr(bootstrap.os, 'execv', lambda binary, argv: captured.append(('exec', binary, argv)))
    monkeypatch.setattr(sys, 'argv', ['score-bootstrap', '--phase', 'source', '--config', str(path),
                                      '--config-sha256', hashlib.sha256(data).hexdigest(),
                                      '--cpu-seconds', '7', '--child-cpu-seconds', '11'])
    bootstrap.main()
    assert captured[0] == ('limit', (7, 11))
    assert captured[1][0] == 'exec'
