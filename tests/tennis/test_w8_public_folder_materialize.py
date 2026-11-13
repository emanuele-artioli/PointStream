import hashlib
from pathlib import Path

import pytest

from experiments.tennis import k6b_generate
from experiments.tennis import w8_public_folder_materialize as materialize


def synthetic(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    declared = {}
    for i, name in enumerate(materialize.SIZES):
        p = inputs / f'{i}.bin'
        p.write_bytes(('synthetic-' + name).encode())
        declared[name] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'size': p.stat().st_size}
    return declared


def test_materialized_regular_files_work_with_unchanged_folder_without_symlinks(tmp_path, monkeypatch):
    declared = synthetic(tmp_path)
    folders = tmp_path / 'output' / 'scratch' / 'folders'
    receipt = materialize.copy_declared(declared, folders)
    monkeypatch.setattr(Path, 'symlink_to', lambda *a, **kw: pytest.fail('model helper attempted symlink'))
    weights = {name: Path(item['path']) for name, item in declared.items()}
    for family in ('vae', 'image_encoder'):
        result = k6b_generate.folder(weights, family, folders.parent)
        for name, item in declared.items():
            if name.startswith(family + '/'):
                copy = result / name.split('/', 1)[1]
                assert copy.is_file() and not copy.is_symlink()
                assert hashlib.sha256(copy.read_bytes()).hexdigest() == item['sha256']
    materialize.cleanup_completed(folders, receipt)
    assert not any(p.is_file() for p in folders.rglob('*'))


def test_wrong_source_bytes_and_changed_copies_fail_closed(tmp_path):
    declared = synthetic(tmp_path)
    bad = dict(declared)
    bad['vae/config.json'] = {**bad['vae/config.json'], 'sha256': '0' * 64}
    with pytest.raises(ValueError, match='bytes differ'):
        materialize.copy_declared(bad, tmp_path / 'bad')
    assert (tmp_path / 'bad').exists()
    folders = tmp_path / 'good'
    receipt = materialize.copy_declared(declared, folders)
    target = folders / 'vae/config.json'
    target.write_text('changed')
    with pytest.raises(ValueError, match='refusing'):
        materialize.cleanup_completed(folders, receipt)
    assert target.read_text() == 'changed'
    with pytest.raises(ValueError, match='fresh'):
        materialize.copy_declared(declared, folders)


def test_model_profile_rejects_unpinned_or_missing_weight(tmp_path):
    declared = synthetic(tmp_path)
    files = {'weight:' + name: {'path': row['path'], 'sha256': row['sha256']} for name, row in declared.items()}
    pinned = {name: row['sha256'] for name, row in declared.items()}
    pinned['vae/config.json'] = '0' * 64
    with pytest.raises(ValueError, match='frozen asset'):
        materialize.materialize_public(files, tmp_path / 'folders', pinned)
