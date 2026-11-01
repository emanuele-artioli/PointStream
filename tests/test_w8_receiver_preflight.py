import hashlib
import io
import tarfile

import pytest

from experiments.tennis import w8_receiver_preflight as preflight


def _tar(path, name, data):
    with tarfile.open(path, 'w') as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def test_source_snapshot_binds_archive_and_extracted_bytes(tmp_path):
    tree = tmp_path / 'tree'
    tree.mkdir()
    (tree / 'code.py').write_bytes(b'code')
    archive = tmp_path / 'source.tar'
    _tar(archive, 'code.py', b'code')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert preflight.source_files(archive, tree, digest)[str(tree / 'code.py')] == hashlib.sha256(b'code').hexdigest()
    (tree / 'code.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source differs'):
        preflight.source_files(archive, tree, digest)
    with pytest.raises(ValueError, match='archive identity'):
        preflight.source_files(archive, tree, '0' * 64)


def test_source_snapshot_rejects_traversal(tmp_path):
    archive = tmp_path / 'source.tar'
    _tar(archive, '../outside', b'code')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='unsafe'):
        preflight.source_files(archive, tmp_path, digest)


def test_aa_source_archive_checks_installed_files(tmp_path):
    tree = tmp_path / 'installed'
    tree.mkdir()
    (tree / 'module.py').write_bytes(b'expected')
    archive = tmp_path / 'aa.tar'
    _tar(archive, 'project/module.py', b'expected')
    assert preflight.verify_aa_tree(archive, tree)['verified_files'] == 1
    (tree / 'module.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source mismatch'):
        preflight.verify_aa_tree(archive, tree)


def test_sam_record_ignores_generated_bytecode_but_not_unhashed_sources(tmp_path, monkeypatch):
    import base64
    import json
    from importlib import metadata
    from types import SimpleNamespace

    source = tmp_path / 'sam.py'
    source.write_bytes(b'source')
    digest = base64.urlsafe_b64encode(hashlib.sha256(b'source').digest()).rstrip(b'=').decode()
    class Entry:
        def __init__(self, name, value):
            self.name = name
            self.hash = SimpleNamespace(mode='sha256', value=value) if value else None
        def __str__(self):
            return self.name
    class Distribution:
        files = [Entry('sam.py', digest), Entry('__pycache__/sam.pyc', None), Entry('sam.dist-info/RECORD', None)]
        def read_text(self, name):
            assert name == 'direct_url.json'
            return json.dumps({'vcs_info': {'commit_id': '2345a4ad109ac29c569da749c91d84f10dc08c40'}})
        def locate_file(self, entry):
            return tmp_path / str(entry)
    distribution = Distribution()
    monkeypatch.setattr(metadata, 'distribution', lambda _: distribution)
    receipt = preflight.verify_sam_distribution(tmp_path)
    assert receipt['excluded_generated_bytecode'] == 1
    assert receipt['verified_files'][str(source)] == hashlib.sha256(b'source').hexdigest()
    distribution.files.append(Entry('unhashed.py', None))
    with pytest.raises(ValueError, match='lacks hash'):
        preflight.verify_sam_distribution(tmp_path)
    distribution.files[-1] = Entry('editable.pth', digest)
    with pytest.raises(ValueError, match='pth'):
        preflight.verify_sam_distribution(tmp_path)


def test_preflight_worker_config_passes_actual_bootstrap_parser(tmp_path, monkeypatch):
    import json
    from experiments.tennis import w8_receiver_bootstrap as bootstrap

    source = tmp_path / 'source_tree'
    source.mkdir()
    output = tmp_path / 'worker'
    output.mkdir()
    groups = {}
    aliases = []
    identities = {}
    for name in ('source', 'prepared', 'fit', 'golden', 'masks'):
        actual = tmp_path / name
        actual.write_bytes(name.encode())
        alias = tmp_path / f'{name}_alias'
        alias.symlink_to(actual)
        aliases.append(str(alias))
        groups[name] = [str(actual)]
        identities[name] = (str(actual), hashlib.sha256(actual.read_bytes()).hexdigest())
    groups['aliases'] = aliases
    monkeypatch.setattr(preflight, 'PROTECTED', identities)
    receipt = tmp_path / 'identity.json'
    receipt.write_text(json.dumps({'protected_inputs': {
        k: {'path': path, 'sha256': digest, 'device': __import__('os').stat(path).st_dev,
            'inode': __import__('os').stat(path).st_ino} for k, (path, digest) in identities.items()}}))
    config = preflight.worker_config(source, output, [], groups, 'user-parent', 'net-parent', receipt)
    config_path = tmp_path / 'worker.json'
    config_path.write_text(json.dumps(config, sort_keys=True))
    assert bootstrap.load_config(config_path, preflight.sha(config_path)) == config
    assert config['cpu_seconds'] == 12 and config['environment']['CUDA_VISIBLE_DEVICES'] == ''
