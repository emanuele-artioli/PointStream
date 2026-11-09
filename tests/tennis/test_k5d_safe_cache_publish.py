from __future__ import annotations

import errno
import hashlib
import io
import os
from pathlib import Path
import tarfile

import pytest

from experiments.tennis import k5d_safe_cache_publish as safe


def roots(tmp_path: Path) -> tuple[Path, Path]:
    task, cache = tmp_path / 'task', tmp_path / 'cache'
    task.mkdir()
    cache.mkdir()
    return task.resolve(), cache.resolve()


def source(tmp_path: Path, data: bytes = b'source') -> tuple[Path, str]:
    path = (tmp_path / 'asset.bin').resolve()
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def test_private_copy_reread_and_atomic_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, cache = roots(tmp_path)
    asset, sha = source(tmp_path)
    calls = []
    def install(src: Path, dst: Path) -> None:
        calls.append((src, dst))
        assert src.parent == task and not dst.exists()
        os.rename(src, dst)
    monkeypatch.setattr(safe, '_publish_noreplace', install)
    result = safe.publish(asset, sha, task, cache)
    assert result['published_noreplace'] and (cache / sha / asset.name).read_bytes() == b'source'
    assert len(calls) == 1 and not (task / sha).exists()


def test_existing_or_concurrent_cache_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, cache = roots(tmp_path)
    asset, sha = source(tmp_path)
    entry = cache / sha
    entry.mkdir()
    (entry / 'old').write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        safe.publish(asset, sha, task, cache)
    assert (entry / 'old').read_bytes() == b'preserve' and not (task / sha).exists()
    entry.rename(cache / 'other')
    def competing(_src: Path, dst: Path) -> None:
        dst.mkdir()
        (dst / 'racer').write_bytes(b'other owner')
        raise FileExistsError(errno.EEXIST, 'concurrent publisher')
    monkeypatch.setattr(safe, '_publish_noreplace', competing)
    with pytest.raises(FileExistsError):
        safe.publish(asset, sha, task, cache)
    assert (entry / 'racer').read_bytes() == b'other owner' and (task / sha / asset.name).is_file()


def test_bad_hash_symlink_and_private_partial_fail_closed(tmp_path: Path) -> None:
    task, cache = roots(tmp_path)
    asset, sha = source(tmp_path)
    with pytest.raises(safe.staging.StagingError):
        safe.publish(asset, '0' * 64, task, cache)
    assert not (cache / ('0' * 64)).exists()
    alias = tmp_path / 'alias'
    alias.symlink_to(asset)
    with pytest.raises(ValueError):
        safe.publish(alias, sha, task, cache)
    (task / sha).mkdir()
    with pytest.raises(FileExistsError):
        safe.publish(asset, sha, task, cache)
    assert not (cache / sha).exists()


def test_verified_tree_rejects_symlinks_and_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, cache = roots(tmp_path)
    archive = (tmp_path / 'model.tar').resolve()
    with tarfile.open(archive, 'w') as bundle:
        member = tarfile.TarInfo('model/weights')
        member.size = 3
        bundle.addfile(member, io.BytesIO(b'abc'))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(safe, '_publish_noreplace', lambda src, dst: os.rename(src, dst))
    result = safe.publish(archive, sha, task, cache, extract=True, max_extract_bytes=3)
    assert result['tree_file_sha256'] == {'model/weights': hashlib.sha256(b'abc').hexdigest()}
    assert (cache / sha / 'tree/model/weights').read_bytes() == b'abc'
    bad = (tmp_path / 'bad.tar').resolve()
    with tarfile.open(bad, 'w') as bundle:
        member = tarfile.TarInfo('link')
        member.type = tarfile.SYMTYPE
        member.linkname = '../outside'
        bundle.addfile(member)
    badsha = hashlib.sha256(bad.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='unsafe'):
        safe.publish(bad, badsha, task, cache, extract=True, max_extract_bytes=3)
    assert not (cache / badsha).exists() and (task / badsha).exists()


def test_unsupported_atomic_install_preserves_private_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, cache = roots(tmp_path)
    asset, sha = source(tmp_path)
    def unsupported(_src: Path, _dst: Path) -> None:
        raise OSError(errno.ENOSYS, 'no no-replace')
    monkeypatch.setattr(safe, '_publish_noreplace', unsupported)
    with pytest.raises(OSError) as exc:
        safe.publish(asset, sha, task, cache)
    assert exc.value.errno == errno.ENOSYS and (task / sha / asset.name).read_bytes() == b'source'
    assert not (cache / sha).exists()


@pytest.mark.skipif(safe.sys.platform != 'linux', reason='actual Linux renameat2')
def test_linux_noreplace_rejects_existing_directory(tmp_path: Path) -> None:
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    (b / 'keep').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        safe._publish_noreplace(a, b)
    assert a.is_dir() and (b / 'keep').read_bytes() == b'keep'


def test_interrupt_before_publication_preserves_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, cache = roots(tmp_path)
    asset, sha = source(tmp_path)
    def interrupted(_source: Path, _entry: Path, _sha: str) -> Path:
        raise KeyboardInterrupt()
    monkeypatch.setattr(safe.staging, '_copy_verified', interrupted)
    with pytest.raises(KeyboardInterrupt):
        safe.publish(asset, sha, task, cache)
    assert (task / sha).is_dir() and not (cache / sha).exists()


def test_extraction_limit_fails_before_tree_or_publication(tmp_path: Path) -> None:
    task, cache = roots(tmp_path)
    archive = (tmp_path / 'large.tar').resolve()
    with tarfile.open(archive, 'w') as bundle:
        member = tarfile.TarInfo('content')
        member.size = 4
        bundle.addfile(member, io.BytesIO(b'data'))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='byte limit'):
        safe.publish(archive, sha, task, cache, extract=True, max_extract_bytes=3)
    assert not (task / sha / 'tree').exists() and not (cache / sha).exists()


def test_reread_corruption_is_never_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, cache = roots(tmp_path)
    asset, sha = source(tmp_path)
    monkeypatch.setattr(safe, '_sha', lambda _path: 'f' * 64)
    with pytest.raises(ValueError, match='copied-byte'):
        safe.publish(asset, sha, task, cache)
    assert not (cache / sha).exists() and (task / sha / asset.name).is_file()


def test_cross_device_publication_error_preserves_private_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, cache = roots(tmp_path)
    asset, sha = source(tmp_path)
    def crossed(_src: Path, _dst: Path) -> None:
        raise OSError(errno.EXDEV, 'cross-device')
    monkeypatch.setattr(safe, '_publish_noreplace', crossed)
    with pytest.raises(OSError) as exc:
        safe.publish(asset, sha, task, cache)
    assert exc.value.errno == errno.EXDEV and not (cache / sha).exists()
    assert (task / sha / asset.name).read_bytes() == b'source'


def test_archive_root_and_executable_mode_are_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task, cache = roots(tmp_path)
    archive = (tmp_path / 'exec.tar').resolve()
    with tarfile.open(archive, 'w') as bundle:
        root = tarfile.TarInfo('.')
        root.type = tarfile.DIRTYPE
        bundle.addfile(root)
        executable = tarfile.TarInfo('./bin/tool')
        executable.size = 3
        executable.mode = 0o755
        bundle.addfile(executable, io.BytesIO(b'run'))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(safe, '_publish_noreplace', lambda src, dst: os.rename(src, dst))
    safe.publish(archive, sha, task, cache, extract=True, max_extract_bytes=3)
    assert (cache / sha / 'tree/bin/tool').stat().st_mode & 0o777 == 0o555


def test_existing_cache_symlink_is_never_followed(tmp_path: Path) -> None:
    task, cache = roots(tmp_path)
    asset, sha = source(tmp_path)
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'keep').write_bytes(b'preserve')
    (cache / sha).symlink_to(external, target_is_directory=True)
    with pytest.raises(FileExistsError):
        safe.publish(asset, sha, task, cache)
    assert (cache / sha).is_symlink() and (external / 'keep').read_bytes() == b'preserve'
    assert not (task / sha).exists()
