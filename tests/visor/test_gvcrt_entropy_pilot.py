import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from experiments.visor import gvcrt_entropy_pilot as pilot


def test_owned_subprocess_stays_in_callers_process_group(tmp_path):
    out = tmp_path / 'stdout'
    err = tmp_path / 'stderr'
    with out.open('w') as stdout, err.open('w') as stderr:
        code = pilot.run_owned([sys.executable, '-c', 'import os; print(os.getpgrp())'],
                               cwd=tmp_path, env=dict(os.environ), stdout=stdout, stderr=stderr, timeout=5)
    assert code == 0
    assert int(out.read_text()) == os.getpgrp()


def test_validator_rejects_tampered_stream_and_requires_fresh_replay(tmp_path, monkeypatch):
    source = tmp_path / 'source.tar.gz'
    source_hashes = {}
    with tarfile.open(source, 'w:gz') as archive:
        for name in pilot.SOURCES:
            raw = name.encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            archive.addfile(entry, io.BytesIO(raw))
            source_hashes[name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(pilot, 'SOURCE_SHA', pilot.digest(source))
    real_digest = pilot.digest
    monkeypatch.setattr(pilot, 'digest', lambda p: 'f' * 64 if p == Path('/usr/bin/g++').resolve() else real_digest(p))
    artifact = tmp_path / 'artifact'
    artifact.mkdir()
    (artifact / 'MLCodec_extensions_cpp.so').write_bytes(b'binary')
    stream = b'test'
    cases = [{'two_entropy_coders': dual, 'exact_roundtrip': True, 'stream_bytes': len(stream),
              'stream_hex': stream.hex(), 'sha256': hashlib.sha256(stream).hexdigest()} for dual in (False, True)]
    probe = {'cases': cases, 'numpy': 'test'}
    (artifact / 'probe.stdout').write_text(json.dumps(probe))
    receipt = {'passed': True, 'source_sha256': pilot.SOURCE_SHA, 'runtime_archive_sha256': pilot.RUNTIME_SHA,
               'header_sha256': pilot.HEADER_SHA, 'compiler_status': 0, 'probe_status': 0,
               'compiler_version': 'g++ test', 'compiler_sha256': 'f' * 64,
               'source_members_sha256': source_hashes, 'extension_sha256': pilot.digest(artifact / 'MLCodec_extensions_cpp.so'),
               'probe_stdout_sha256': pilot.digest(artifact / 'probe.stdout'), 'probe': probe}
    (artifact / 'pilot.json').write_text(json.dumps(receipt))
    calls = []
    def replay(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=json.dumps(probe))
    monkeypatch.setattr(pilot.subprocess, 'run', replay)
    assert pilot.validate(artifact, source)['passed'] is True
    assert len(calls) == 1
    cases[0]['stream_bytes'] = 99
    (artifact / 'probe.stdout').write_text(json.dumps(probe))
    receipt['probe'] = probe
    receipt['probe_stdout_sha256'] = pilot.digest(artifact / 'probe.stdout')
    (artifact / 'pilot.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='stream receipt'):
        pilot.validate(artifact, source)
