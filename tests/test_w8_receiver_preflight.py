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
    (tree / 'project').mkdir()
    (tree / 'project/module.py').write_bytes(b'expected')
    archive = tmp_path / 'aa.tar'
    _tar(archive, 'project/module.py', b'expected')
    assert preflight.verify_aa_tree(archive, tree, expected_links={}, expected_archive_sha=preflight.sha(archive))['verified_files'] == 1
    (tree / 'project/module.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source mismatch'):
        preflight.verify_aa_tree(archive, tree, expected_links={}, expected_archive_sha=preflight.sha(archive))


def test_sam_record_ignores_generated_bytecode_but_not_unhashed_sources(tmp_path, monkeypatch):
    import base64
    import json
    from importlib import metadata
    from types import SimpleNamespace

    site = tmp_path / 'lib/python3.12/site-packages'
    site.mkdir(parents=True)
    (tmp_path / 'lib/python3.1').symlink_to('python3.12')
    info = site / 'sam3-0.1.0.dist-info'
    info.mkdir()
    source = site / 'sam.py'
    source.write_bytes(b'source')
    digest = base64.urlsafe_b64encode(hashlib.sha256(b'source').digest()).rstrip(b'=').decode()
    class Entry:
        def __init__(self, name, value):
            self.name = name
            self.hash = SimpleNamespace(mode='sha256', value=value) if value else None
        def __str__(self):
            return self.name
    class Distribution:
        _path = info
        metadata = {'Name': 'sam3'}
        files = [Entry('sam.py', digest), Entry('__pycache__/sam.pyc', None), Entry('sam.dist-info/RECORD', None)]
        def read_text(self, name):
            assert name == 'direct_url.json'
            return json.dumps({'vcs_info': {'commit_id': '2345a4ad109ac29c569da749c91d84f10dc08c40'}})
        def locate_file(self, entry):
            return site / str(entry)
    distribution = Distribution()
    monkeypatch.setattr(metadata, 'distributions', lambda **_: [distribution])
    receipt = preflight.verify_sam_distribution(tmp_path)
    assert receipt['excluded_generated_bytecode'] == 1
    assert receipt['site_paths'] == [str(site.resolve())]
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


def test_identity_budget_amendment_preserves_pilot_rule():
    rate = 2.6378631591796985e-9
    size = 19856141745
    before = preflight.identity_projection([rate], size, 0.11, 20)
    after = preflight.identity_projection([rate], size, 0.11, 125)
    assert before['safety_factor'] == after['safety_factor'] == 2
    assert before['fixed_aa_sam_cpu_reserve'] == after['fixed_aa_sam_cpu_reserve'] == 5
    assert before['projected_cpu_seconds'] == after['projected_cpu_seconds']
    assert before['status'] == 'stop' and after['status'] == 'fit'
    with pytest.raises(ValueError, match='invalid identity pilot'):
        preflight.identity_projection([rate], size, 0.11, 126)


def test_amended_preflight_rejects_caps_before_any_host_action(tmp_path):
    with pytest.raises(ValueError, match='identity CPU limit'):
        preflight.run({}, '0' * 64, tmp_path, identity_cpu_limit=126)
    with pytest.raises(ValueError, match='identity wall limit'):
        preflight.run({}, '0' * 64, tmp_path, identity_cpu_limit=125, identity_wall_limit=1201)


def test_wall_allowance_reserves_unstarted_workers_and_closeout():
    assert preflight.wall_allowance(600, 100, 510, 2) == 255
    assert preflight.wall_allowance(600, 450, 120, 1) == 25
    with pytest.raises(RuntimeError, match='insufficient remaining wall'):
        preflight.wall_allowance(600, 355, 120, 2)
    with pytest.raises(ValueError, match='invalid wall deadline'):
        preflight.wall_allowance(float('nan'), 1, 120, 0)


def _aa_archive_with_links(path, links):
    with tarfile.open(path, 'w') as tar:
        data = b'archive-source'
        member = tarfile.TarInfo('animate_anyone/__init__.py')
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
        for name, target in links.items():
            entry = tarfile.TarInfo(name)
            entry.type = tarfile.SYMTYPE
            entry.linkname = target
            tar.addfile(entry)


def test_aa_links_are_exact_metadata_and_absence_is_permitted(tmp_path):
    tree = tmp_path / 'installed'
    (tree / 'animate_anyone/configs').mkdir(parents=True)
    (tree / 'animate_anyone/__init__.py').write_bytes(b'archive-source')
    archive = tmp_path / 'aa.tar'
    _aa_archive_with_links(archive, preflight.AA_LINKS)
    result = preflight.verify_aa_tree(archive, tree, expected_archive_sha=preflight.sha(archive))
    assert set(result['installed_link_status'].values()) == {'absent'}
    (tree / 'Models').mkdir()
    for name, target in preflight.AA_LINKS.items():
        (tree / name).symlink_to(target)
    result = preflight.verify_aa_tree(archive, tree, expected_archive_sha=preflight.sha(archive))
    assert set(result['installed_link_status'].values()) == {'matched'}
    (tree / 'src').unlink()
    (tree / 'src').symlink_to('/etc')
    with pytest.raises(ValueError, match='link metadata mismatch'):
        preflight.verify_aa_tree(archive, tree, expected_archive_sha=preflight.sha(archive))


def test_aa_rejects_unknown_or_changed_archive_links_and_repointed_target(tmp_path):
    tree = tmp_path / 'installed'
    (tree / 'animate_anyone/configs').mkdir(parents=True)
    (tree / 'animate_anyone/__init__.py').write_bytes(b'archive-source')
    archive = tmp_path / 'aa.tar'
    _aa_archive_with_links(archive, {**preflight.AA_LINKS, 'other': '/etc'})
    with pytest.raises(ValueError, match='unexpected AA link'):
        preflight.verify_aa_tree(archive, tree, expected_archive_sha=preflight.sha(archive))
    archive.unlink()
    _aa_archive_with_links(archive, {**preflight.AA_LINKS, 'src': '../escape'})
    with pytest.raises(ValueError, match='unexpected AA link'):
        preflight.verify_aa_tree(archive, tree, expected_archive_sha=preflight.sha(archive))
    archive.unlink()
    _aa_archive_with_links(archive, preflight.AA_LINKS)
    (tree / 'src').symlink_to('animate_anyone')
    (tree / 'animate_anyone/configs').rmdir()
    (tree / 'animate_anyone/configs').symlink_to('/etc')
    (tree / 'configs').symlink_to('animate_anyone/configs')
    with pytest.raises(ValueError, match='escapes or is repointed'):
        preflight.verify_aa_tree(archive, tree, expected_archive_sha=preflight.sha(archive))


def test_sam_deduplicates_aliases_but_rejects_distinct_installations(tmp_path, monkeypatch):
    from importlib import metadata

    site = tmp_path / 'lib/python3.12/site-packages'
    site.mkdir(parents=True)
    (tmp_path / 'lib/python3.1').symlink_to('python3.12')
    one, two = site / 'sam3-0.1.dist-info', site / 'sam3-0.2.dist-info'
    one.mkdir()
    two.mkdir()
    class Distribution:
        metadata = {'Name': 'sam3'}
        def __init__(self, path):
            self._path = path
    monkeypatch.setattr(metadata, 'distributions', lambda **_: [Distribution(one), Distribution(two)])
    with pytest.raises(ValueError, match='ambiguous SAM installations: 2'):
        preflight.verify_sam_distribution(tmp_path)


def test_checkpoint_contains_only_completed_verified_files(tmp_path):
    tree = tmp_path / 'tree'
    tree.mkdir()
    (tree / 'a').write_bytes(b'good')
    (tree / 'b').write_bytes(b'changed')
    archive = tmp_path / 'archive.tar'
    with tarfile.open(archive, 'w') as tar:
        for name, data in [('a', b'good'), ('b', b'expected')]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    import json
    checkpoint = tmp_path / 'checkpoint.jsonl'
    record = preflight.start_checkpoint(checkpoint)
    with pytest.raises(ValueError, match='source mismatch'):
        preflight.verify_aa_tree(archive, tree, expected_links={}, expected_archive_sha=preflight.sha(archive),
                                 checkpoint=record)
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['path'] == str(tree / 'a')
    assert rows[0]['sha256'] == hashlib.sha256(b'good').hexdigest()
    assert all(key in rows[0] for key in ('size', 'device', 'inode', 'mtime_ns', 'ctime_ns'))
    with pytest.raises(FileExistsError):
        preflight.start_checkpoint(checkpoint)


def test_asset_phase_runs_cheap_checks_before_large_inputs(tmp_path, monkeypatch):
    entries = {'main_archive': {'sha256': preflight.ENV_SHAS['main']},
               'jcv_archive': {'sha256': preflight.ENV_SHAS['jcv']},
               'aa_archive': {'sha256': preflight.AA_SHA}}
    def stop_cheap(*_args):
        raise RuntimeError('cheap failure')
    monkeypatch.setattr(preflight, 'cheap_assets', stop_cheap)
    with pytest.raises(RuntimeError, match='cheap failure'):
        preflight.asset_phase({'identity_files': entries}, tmp_path / 'assets.json', 125)
    assert (tmp_path / 'verified-checkpoint.jsonl').is_file()
    assert not (tmp_path / 'inventory.json').exists()


def test_stable_file_hash_detects_changes_during_read(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    path = tmp_path / 'asset'
    path.write_bytes(b'data')
    real_fstat = os.fstat
    calls = 0
    def fstat(fd):
        nonlocal calls
        calls += 1
        stat = real_fstat(fd)
        if calls == 1:
            return stat
        return SimpleNamespace(st_mode=stat.st_mode, st_dev=stat.st_dev, st_ino=stat.st_ino,
                               st_size=stat.st_size, st_mtime_ns=stat.st_mtime_ns + 1,
                               st_ctime_ns=stat.st_ctime_ns)
    monkeypatch.setattr(os, 'fstat', fstat)
    with pytest.raises(ValueError, match='changed during hash'):
        preflight.stable_file_hash(path)


def test_aa_models_alias_parent_cannot_redirect_to_external_tree(tmp_path):
    tree = tmp_path / 'installed'
    (tree / 'animate_anyone/configs').mkdir(parents=True)
    (tree / 'animate_anyone/__init__.py').write_bytes(b'archive-source')
    (tree / 'Models').symlink_to('/etc')
    archive = tmp_path / 'aa.tar'
    _aa_archive_with_links(archive, preflight.AA_LINKS)
    with pytest.raises(ValueError, match='link parent is a symlink'):
        preflight.verify_aa_tree(archive, tree, expected_archive_sha=preflight.sha(archive))


def test_cheap_command_verifies_config_and_refuses_reuse(tmp_path, monkeypatch):
    import json

    config = tmp_path / 'config.json'
    config.write_text('{"approved":true}')
    output = tmp_path / 'receipt.json'
    def cheap(obj, checkpoint):
        assert obj == {'approved': True}
        checkpoint({'kind': 'checked'})
        return {'passed': True, 'scope': 'AA/SAM only'}
    monkeypatch.setattr(preflight, 'cheap_assets', cheap)
    with pytest.raises(ValueError, match='config identity'):
        preflight.run_cheap(config, '0' * 64, output, tmp_path)
    assert not output.exists()
    preflight.run_cheap(config, preflight.sha(config), output, tmp_path)
    assert json.loads(output.read_text())['passed'] is True
    assert 'checked' in output.with_name(output.name + '.checkpoint.jsonl').read_text()
    with pytest.raises(ValueError, match='must be new'):
        preflight.run_cheap(config, preflight.sha(config), output, tmp_path)


def test_aa_canonical_member_does_not_use_legacy_alternative(tmp_path):
    tree = tmp_path / 'installed'
    (tree / 'package/scripts').mkdir(parents=True)
    (tree / 'scripts').mkdir()
    canonical = tree / 'package/scripts/render.py'
    legacy = tree / 'scripts/render.py'
    canonical.write_bytes(b'expected')
    legacy.write_bytes(b'legacy-different')
    archive = tmp_path / 'aa.tar'
    _tar(archive, 'package/scripts/render.py', b'expected')
    result = preflight.verify_aa_tree(archive, tree, expected_links={}, expected_archive_sha=preflight.sha(archive))
    assert result['files'] == {str(canonical): hashlib.sha256(b'expected').hexdigest()}
    canonical.unlink()
    legacy.write_bytes(b'expected')
    with pytest.raises(ValueError, match='canonical installed path missing'):
        preflight.verify_aa_tree(archive, tree, expected_links={}, expected_archive_sha=preflight.sha(archive))
