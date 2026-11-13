import hashlib
import struct
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from experiments.tennis.k5d_worker import resource, validate_native_header
from src.pointstream import ball, ball_stream


def stream() -> ball_stream.Stream:
    # Synthetic container bytes test parsing only; no codec evidence is implied.
    header = struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32, b"AV01", 16, 12, 30, 1, 2, 0)
    frames = b"".join(struct.pack("<IQ", 1, t) + b"x" for t in range(2))
    return ball_stream.Stream(16, 12, Fraction(30), 2, "svt-av1-ivf", "svt-test",
                              header + frames, ball.encode_frames([None, None], [None, None], 16, 12))


def test_ivf_header_is_bound_to_wire_geometry_and_count() -> None:
    sample = stream()
    validate_native_header(sample)
    for bad in (replace(sample, width=18), replace(sample, fps=Fraction(25)),
                replace(sample, frames=1), replace(sample, native=sample.native[:-1])):
        with pytest.raises(ValueError):
            validate_native_header(bad)


def test_resource_identifier_and_file_hash(tmp_path: Path) -> None:
    decoder = tmp_path / "decoder"
    decoder.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    mapping = {"resources": {"svt-test": {"codec": "svt-av1-ivf", "decoder": str(decoder), "decoder_sha256": digest}}}
    assert resource(stream(), mapping)["decoder"] == str(decoder)
    with pytest.raises(ValueError):
        resource(replace(stream(), resource_id="different"), mapping)
    decoder.write_bytes(b"different")
    with pytest.raises(ValueError, match="invalid installed"):
        resource(stream(), mapping)


def test_failed_isolated_native_persists_stderr_before_missing_preexec(tmp_path, monkeypatch):
    import json
    from experiments.tennis import k5d_worker as worker

    decoder = tmp_path / 'decoder'
    decoder.write_text('fixture')
    work = tmp_path / 'result'
    work.mkdir()
    observation = {'nonce': 'n', 'namespace_pids': [7], 'namespaces': {'pid': 'private'}, 'birth': '9', 'host_pgid': 40}
    (tmp_path / 'native-observer.json').write_text(json.dumps(observation))
    stderr = b'x' * 5000 + b'actual startup failure'
    class FailedProcess:
        pid = 7
        returncode = 73
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def communicate(self, **kwargs): return b'', stderr
        def kill(self): pass
        def wait(self): pass
    monkeypatch.setenv('PYTHONPATH', '/reviewed')
    monkeypatch.setattr(worker.subprocess, 'Popen', lambda *a, **kw: FailedProcess())
    isolation = {'nonce': 'n', 'campaign_host': {'campaign_pgid': 40}, 'actual_policy': {'execute_files': [str(decoder)]},
                 'namespaces': {'namespaces': {'pid': 'private'}}}
    with pytest.raises(RuntimeError, match='actual startup failure'):
        worker._execute_native([str(decoder)], work, isolation)
    record = json.loads((tmp_path / 'native-process-result.json').read_text())
    assert record['returncode'] == 73
    assert record['stderr_bytes'] == len(stderr) and record['stderr_sha256'] == hashlib.sha256(stderr).hexdigest()
    assert len(record['stderr_tail']) == 4096 and record['stderr_tail'].endswith('actual startup failure')
    assert not (tmp_path / 'native-preexec.json').exists()


def test_native_failure_before_observer_is_not_reported_as_timeout(tmp_path, monkeypatch):
    import json
    from experiments.tennis import k5d_worker as worker

    decoder = tmp_path / 'decoder'
    decoder.write_text('fixture')
    work = tmp_path / 'result'
    work.mkdir()
    class FailedProcess:
        pid = 7
        returncode = 1
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def poll(self): return 1
        def communicate(self, **kwargs): return b'', b'wrapper cannot import'
        def kill(self): pass
        def wait(self): pass
    monkeypatch.setenv('PYTHONPATH', '/reviewed')
    monkeypatch.setattr(worker.subprocess, 'Popen', lambda *a, **kw: FailedProcess())
    isolation = {'nonce': 'n', 'campaign_host': {'campaign_pgid': 40}, 'actual_policy': {'execute_files': [str(decoder)]},
                 'namespaces': {'namespaces': {'pid': 'private'}}}
    with pytest.raises(RuntimeError, match='wrapper cannot import'):
        worker._execute_native([str(decoder)], work, isolation)
    record = json.loads((tmp_path / 'native-process-result.json').read_text())
    assert record['returncode'] == 1 and record['stderr_tail'] == 'wrapper cannot import'
    assert not (tmp_path / 'native-observer.json').exists()
