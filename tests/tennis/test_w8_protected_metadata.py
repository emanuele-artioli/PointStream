import hashlib
import json
from pathlib import Path

import pytest

from experiments.tennis import w8_protected_metadata as metadata
from experiments.tennis import w8_receiver_bootstrap as bootstrap


def test_stat_capture_never_opens_any_protected_content(tmp_path, monkeypatch):
    protected = {}
    historical = {}
    for index, key in enumerate(('source', 'prepared', 'fit', 'golden', 'masks')):
        path = tmp_path / key
        path.write_bytes(b'private content')
        st = path.stat()
        protected[key] = (str(path), str(index) * 64)
        historical[key] = {'path': str(path), 'sha256': str(index) * 64, 'device': st.st_dev, 'inode': st.st_ino}
    monkeypatch.setattr(metadata, 'PROTECTED', protected)
    old = tmp_path / 'historical.json'
    old.write_text(json.dumps({'protected_inputs': historical}))
    old_open = Path.open
    def deny_private(path, *args, **kwargs):
        if str(path) in {value[0] for value in protected.values()}:
            raise AssertionError('protected file contents were opened')
        return old_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', deny_private)
    output = tmp_path / 'metadata.json'
    reference = metadata.capture(old, output, expected_historical_sha256=hashlib.sha256(old.read_bytes()).hexdigest())
    result = json.loads(output.read_text())
    assert result['content_verified_now'] is False and result['scope'] == metadata.SCOPE
    assert reference['historical_sha256'] == result['historical_receipt_sha256']
    metadata.validate(result, {'protected_inputs': historical}, check_current=True)
    result['protected_inputs']['golden']['reference_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='metadata differs'):
        metadata.validate(result, {'protected_inputs': historical})


@pytest.mark.parametrize('category', ['source', 'prepared', 'fit', 'golden', 'masks'])
def test_witness_denial_probe_never_reads_on_unexpected_open(tmp_path, monkeypatch, category):
    private = tmp_path / category
    private.write_bytes(b'private content')
    def deny_read(*_args, **_kwargs):
        raise AssertionError('protected bytes were read')
    monkeypatch.setattr(Path, 'open', deny_read)
    with pytest.raises(RuntimeError, match='protected path readable'):
        bootstrap._check_denials([private], open_only=True)


@pytest.mark.parametrize('category', ['source', 'prepared', 'fit', 'golden', 'masks'])
def test_witness_native_probe_never_reads_on_unexpected_open(tmp_path, monkeypatch, category):
    from experiments.tennis import w8_receiver_native_bootstrap as native

    private = tmp_path / category
    private.write_bytes(b'private content')
    def deny_read(*_args, **_kwargs):
        raise AssertionError('protected bytes were read')
    monkeypatch.setattr(Path, 'open', deny_read)
    with pytest.raises(ValueError, match='protected input readable'):
        native._probe_denials([private], open_only=True)


def test_all_witness_metadata_and_denial_probes_never_read_protected(tmp_path, monkeypatch):
    import builtins
    import errno
    import os

    protected, historical, aliases = {}, {}, []
    for index, key in enumerate(('source', 'prepared', 'fit', 'golden', 'masks')):
        path = tmp_path / key
        path.write_bytes(b'private')
        alias = tmp_path / f'alias-{key}'
        alias.symlink_to(path)
        aliases.append(alias)
        st = path.stat()
        protected[key] = (str(path), str(index) * 64)
        historical[key] = {'path': str(path), 'sha256': str(index) * 64, 'device': st.st_dev, 'inode': st.st_ino}
    monkeypatch.setattr(metadata, 'PROTECTED', protected)
    reference = tmp_path / 'historical.json'
    reference.write_text(json.dumps({'protected_inputs': historical}))
    forbidden = {row[0] for row in protected.values()} | {str(path) for path in aliases}
    old_path_open, old_open, old_os_open = Path.open, builtins.open, os.open
    probe_enabled = False
    calls = []
    def guarded_path_open(path, *args, **kwargs):
        if str(path) in forbidden:
            raise AssertionError('protected Path.open attempted')
        return old_path_open(path, *args, **kwargs)
    def guarded_open(path, *args, **kwargs):
        if str(path) in forbidden:
            raise AssertionError('protected builtins.open attempted')
        return old_open(path, *args, **kwargs)
    def guarded_os_open(path, flags, *args, **kwargs):
        if str(path) in forbidden:
            if not probe_enabled or flags & os.O_ACCMODE != os.O_RDONLY:
                raise AssertionError('unexpected protected os.open')
            calls.append(str(path))
            raise PermissionError(errno.EACCES, 'denied', str(path))
        return old_os_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_path_open)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    monkeypatch.setattr(os, 'open', guarded_os_open)
    output = tmp_path / 'metadata.json'
    metadata.capture(reference, output, expected_historical_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    metadata.validate(json.loads(output.read_text()), {'protected_inputs': historical}, check_current=True)
    probe_enabled = True
    denied = [Path(row[0]) for row in protected.values()] + aliases
    assert set(bootstrap._check_denials(denied, open_only=True)) == forbidden
    assert set(calls) == forbidden


def test_capture_rejects_protected_or_wrong_historical_before_read(tmp_path, monkeypatch):
    private = tmp_path / 'private'
    private.write_bytes(b'private')
    monkeypatch.setattr(metadata, 'PROTECTED', {'source': (str(private), 'a' * 64)})
    original = Path.read_bytes
    def no_private(path):
        if path == private:
            raise AssertionError('protected content read')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', no_private)
    with pytest.raises(ValueError, match='aliases protected'):
        metadata.capture(private, tmp_path / 'out', expected_historical_sha256='a' * 64)
    old = tmp_path / 'historical.json'
    old.write_text('{}')
    with pytest.raises(ValueError, match='SHA differs'):
        metadata.capture(old, tmp_path / 'out', expected_historical_sha256='b' * 64)


def test_historical_guard_rejects_alias_symlink_and_oversize_before_os_read(tmp_path, monkeypatch):
    import os

    protected = tmp_path / 'private'
    protected.write_bytes(b'private')
    alias = tmp_path / 'hardlink'
    os.link(protected, alias)
    symlink = tmp_path / 'symlink'
    symlink.symlink_to(protected)
    oversized = tmp_path / 'oversized'
    with oversized.open('wb') as stream:
        stream.truncate(2_000_001)
    monkeypatch.setattr(metadata, 'PROTECTED', {'source': (str(protected), 'a' * 64)})
    monkeypatch.setattr(os, 'read', lambda *_args: (_ for _ in ()).throw(AssertionError('unexpected content read')))
    for candidate in (protected, alias, symlink, oversized):
        with pytest.raises(ValueError):
            metadata.read_historical(candidate, 'a' * 64)


def test_historical_guard_rejects_opened_inode_swap_before_read(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    protected = tmp_path / 'private'
    protected.write_bytes(b'private')
    candidate = tmp_path / 'historical.json'
    candidate.write_text('{}')
    monkeypatch.setattr(metadata, 'PROTECTED', {'source': (str(protected), 'a' * 64)})
    original = os.fstat
    def swapped(fd):
        info = original(fd)
        return SimpleNamespace(st_dev=info.st_dev, st_ino=info.st_ino + 1, st_size=info.st_size,
                               st_mtime_ns=info.st_mtime_ns, st_ctime_ns=info.st_ctime_ns)
    monkeypatch.setattr(os, 'fstat', swapped)
    monkeypatch.setattr(os, 'read', lambda *_args: (_ for _ in ()).throw(AssertionError('unexpected content read')))
    with pytest.raises(ValueError, match='changed before opening'):
        metadata.read_historical(candidate, 'a' * 64)
