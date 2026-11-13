import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'docs/experiments/w8-inputs/receiver-20261101/preseal_metadata_check.py'
spec = importlib.util.spec_from_file_location('admission', MODULE)
assert spec and spec.loader
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


def test_metadata_only_and_mismatch_rejection(tmp_path, monkeypatch):
    protected = {}
    for key in ('source', 'prepared', 'fit', 'golden', 'masks'):
        path = tmp_path / key
        path.write_bytes(b'private')
        protected[key] = str(path)
    monkeypatch.setattr(admission, 'PROTECTED', protected)
    public = tmp_path / 'public'
    public.write_bytes(b'public')
    baseline = {'schema': 1, 'scope': 'root-reviewed W8 preseal metadata baseline',
                'protected': {k: admission.stamp(v) for k, v in protected.items()},
                'paths': [{'role': 'packet', 'path': str(public), 'expected_sha256': 'a' * 64,
                           'stamp': admission.stamp(public)}]}
    control = tmp_path / 'baseline.json'
    control.write_text(json.dumps(baseline))
    digest = hashlib.sha256(control.read_bytes()).hexdigest()
    original = Path.open
    original_os_open = os.open
    def deny_content(path, *args, **kwargs):
        if str(path) in set(protected.values()) | {str(public)}:
            raise AssertionError('input content opened')
        return original(path, *args, **kwargs)
    def deny_os_content(path, flags, *args, **kwargs):
        if str(path) in set(protected.values()) | {str(public)}:
            raise AssertionError('input content opened')
        return original_os_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', deny_content)
    monkeypatch.setattr(os, 'open', deny_os_content)
    result = admission.check(control, digest, tmp_path / 'receipt.json')
    assert result['paths'][0]['content_verified_now'] is False
    with pytest.raises(ValueError, match='baseline SHA differs'):
        admission.check(control, 'b' * 64, tmp_path / 'wrong-sha.json')
    baseline['paths'][0]['stamp']['size'] += 1
    control.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match='metadata changed'):
        admission.check(control, hashlib.sha256(control.read_bytes()).hexdigest(), tmp_path / 'changed.json')
    baseline['paths'][0]['stamp'] = admission.stamp(public)
    os.link(protected['source'], tmp_path / 'alias')
    baseline['protected'] = {k: admission.stamp(v) for k, v in protected.items()}
    baseline['paths'][0]['path'] = str(tmp_path / 'alias')
    baseline['paths'][0]['stamp'] = admission.stamp(tmp_path / 'alias')
    control.write_text(json.dumps(baseline))
    digest = hashlib.sha256(control.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='aliases protected'):
        admission.check(control, digest, tmp_path / 'failure.json')
