"""Bounded, CPU-only local supervisor for two W8 receiver bootstrap probes.

The caller supplies a reviewed configuration and a frozen source archive SHA.
No fleet submission, remote connection, model execution or download occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable

ENV_SHAS = {
    'main': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661',
    'jcv': 'bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e',
}
AA_SHA = 'bfca6cf1129a0154bf2c3431c0e01b34f5e815eda5ba504f54e001c8ceeb99ee'
EXPECTED_TORCH = {'main': '2.10.0+cu128', 'jcv': '2.7.0+cu128'}
PROTECTED = {
    'source': ('/home/itec/emanuele/Datasets/pointstream-data/tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__000.mp4',
               'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d'),
    'prepared': ('/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T162058Z-c9b2c47b/full/published.tar',
                 '94a5d6318fc113557fb24c8a32f2edffdf368f7a289484fddd4b19a749302bfa'),
    'fit': ('/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T150848Z-a582ecc0/full/published.tar',
            'cceb49b3d593c770e5b0d8dcd5d69b5b358bf88d76030cef8034674417907584'),
    'golden': ('/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T163512Z-1d0712ad/full/published.tar',
               'cce89567a57bd202db8dbd6a39599688bce8a52e048d53f2b38fd7fda7da4985'),
    'masks': ('/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T173108Z-0782499a/full/published.tar',
              '4eb91e79c287a49accc17fcd115819e7345bc078d49e1477ce43ccdf87cd549b'),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_files(archive: Path, tree: Path, expected_sha: str) -> dict[str, str]:
    """Verify actual extracted regular files against the exact supplied tar."""
    if len(expected_sha) != 64 or expected_sha == '0' * 64 or sha(archive) != expected_sha:
        raise ValueError('frozen source archive identity mismatch')
    tree = tree.resolve(strict=True)
    files = {}
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe source archive member')
            if member.isdir():
                continue
            if not member.isfile() or member.name in files:
                raise ValueError('unsupported or duplicate source archive member')
            actual = (tree / path).resolve(strict=True)
            if not actual.is_relative_to(tree) or not actual.is_file():
                raise ValueError('source path escapes frozen tree')
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError('unreadable source archive member')
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            if sha(actual) != h.hexdigest():
                raise ValueError(f'extracted source differs: {member.name}')
            files[str(actual)] = h.hexdigest()
    if not files:
        raise ValueError('empty source archive')
    return files


AA_LINKS = {
    'Models/finetuned_tennis': '/home/itec/emanuele/Models/AnimateAnyone/profiles/finetuned_tennis',
    'Models/original': '/home/itec/emanuele/Models/AnimateAnyone/profiles/original',
    'configs': 'animate_anyone/configs',
    'src': 'animate_anyone',
}


def stable_file_hash(path: Path) -> dict[str, Any]:
    """Hash one regular file and reject changes during or immediately after its read."""
    import stat

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f'not a regular file: {path}')
        digest = hashlib.sha256()
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        if identity(before) != identity(after) or identity(before) != identity(current):
            raise ValueError(f'file changed during hash: {path}')
        return {'path': str(path), 'sha256': digest.hexdigest(), 'size': before.st_size,
                'device': before.st_dev, 'inode': before.st_ino,
                'mtime_ns': before.st_mtime_ns, 'ctime_ns': before.st_ctime_ns}
    finally:
        os.close(descriptor)


def _plain_path(tree: Path, path: Path) -> bool:
    """Reject symlinks in every installed member's path without dereferencing them."""
    if not path.is_relative_to(tree):
        return False
    cursor = tree
    for part in path.relative_to(tree).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return False
    return True


def verify_aa_tree(archive: Path, tree: Path, *, expected_links: dict[str, str] | None = None,
                   expected_archive_sha: str = AA_SHA,
                   checkpoint: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Verify every regular source byte and four exact symlink metadata records."""
    tree = tree.resolve(strict=True)
    archive_record = stable_file_hash(archive)
    if archive_record['sha256'] != expected_archive_sha:
        raise ValueError('AA archive identity mismatch')
    links = AA_LINKS if expected_links is None else expected_links
    found_links: set[str] = set()
    installed_links: dict[str, str] = {}
    seen: set[str] = set()
    files: dict[str, str] = {}
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not parts and member.isdir():
                continue
            if not parts or Path(member.name).is_absolute() or '..' in parts:
                raise ValueError(f'unsafe AA archive entry: {member.name}')
            name = str(Path(*parts))
            if name in seen:
                raise ValueError(f'duplicate AA archive entry: {name}')
            seen.add(name)
            if member.isdir():
                continue
            if member.issym():
                if name not in links or links[name] != member.linkname:
                    raise ValueError(f'unexpected AA link metadata: {name}')
                installed = tree / name
                if not _plain_path(tree, installed.parent):
                    raise ValueError(f'AA link parent is a symlink: {name}')
                if not os.path.lexists(installed):
                    installed_links[name] = 'absent'
                    found_links.add(name)
                    continue
                if not installed.is_symlink() or os.readlink(installed) != member.linkname:
                    raise ValueError(f'installed AA link metadata mismatch: {name}')
                installed_links[name] = 'matched'
                if not Path(member.linkname).is_absolute():
                    target = tree / member.linkname
                    if not _plain_path(tree, target) or not target.exists() or not target.resolve().is_relative_to(tree):
                        raise ValueError(f'AA relative alias escapes or is repointed: {name}')
                found_links.add(name)
                continue  # Absolute model aliases are metadata only; never resolve their target.
            if not member.isfile():
                raise ValueError(f'unsupported AA archive entry: {name}')
            # Archive names are canonical. Prefix-stripped legacy scripts may
            # coexist with different bytes and must never substitute for them.
            installed = tree.joinpath(*parts)
            if not _plain_path(tree, installed) or not installed.is_file():
                raise ValueError(f'AA canonical installed path missing or unsafe: {name}')
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError('unreadable AA archive member')
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            record = stable_file_hash(installed)
            if record['sha256'] != digest.hexdigest():
                raise ValueError(f'AA installed source mismatch: {name}')
            files[str(installed)] = record['sha256']
            if checkpoint:
                checkpoint({'kind': 'aa_regular', **record})
    current = archive.stat(follow_symlinks=False)
    if ((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            != (archive_record['device'], archive_record['inode'], archive_record['size'],
                archive_record['mtime_ns'], archive_record['ctime_ns'])):
        raise ValueError('AA archive changed during member verification')
    if found_links != set(links) or not files:
        raise ValueError('AA expected link or regular-file coverage mismatch')
    return {'tree': str(tree), 'verified_files': len(files), 'files': files,
            'link_metadata': {name: links[name] for name in sorted(found_links)},
            'installed_link_status': installed_links,
            'scope': 'regular archive members and exact link metadata; no cached-environment equivalence'}


def verify_sam_distribution(prefix: Path, checkpoint: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Check installed SAM git identity and every hashed wheel RECORD entry."""
    import base64
    from importlib import metadata

    prefix = prefix.resolve(strict=True)
    site_paths = sorted({p.resolve(strict=True) for p in prefix.glob('lib/python*/site-packages') if p.is_dir()})
    if not site_paths or any(not path.is_relative_to(prefix) for path in site_paths):
        raise ValueError('SAM site-packages paths missing or outside environment')
    found: dict[Path, Any] = {}
    for site in site_paths:
        for candidate in metadata.distributions(path=[str(site)]):
            if str(candidate.metadata.get('Name', '')).lower().replace('-', '_') != 'sam3':
                continue
            location = Path(str(getattr(candidate, '_path', ''))).resolve(strict=True)
            if not location.is_relative_to(site):
                raise ValueError('SAM distribution metadata outside its site-packages')
            found[location] = candidate
    if len(found) != 1:
        raise ValueError(f'ambiguous SAM installations: {len(found)}')
    dist = next(iter(found.values()))
    direct = dist.read_text('direct_url.json')
    if direct is None:
        raise ValueError('SAM direct_url metadata absent')
    url = json.loads(direct)
    if url.get('dir_info', {}).get('editable') is True:
        raise ValueError('editable SAM installation requires separate pinned source verification')
    if url.get('vcs_info', {}).get('commit_id') != '2345a4ad109ac29c569da749c91d84f10dc08c40':
        raise ValueError('SAM source commit mismatch')
    if dist.files is None:
        raise ValueError('SAM RECORD absent')
    count = 0
    generated_bytecode = 0
    source_files_count = 0
    verified_files = {}
    for entry in dist.files:
        if str(entry).endswith('.pth'):
            raise ValueError('SAM .pth install requires separately pinned source verification')
        if str(entry).endswith('.pyc'):
            generated_bytecode += 1
            continue
        if entry.hash is None:
            if str(entry).endswith('/RECORD'):
                continue
            raise ValueError(f'SAM RECORD entry lacks hash: {entry}')
        if entry.hash.mode != 'sha256':
            raise ValueError('SAM RECORD uses unsupported digest')
        actual = Path(str(dist.locate_file(entry))).resolve(strict=True)
        if not actual.is_relative_to(prefix.resolve(strict=True)) or not actual.is_file():
            raise ValueError('SAM RECORD path outside installed environment')
        record = stable_file_hash(actual)
        hex_digest = record['sha256']
        digest = base64.urlsafe_b64encode(bytes.fromhex(hex_digest)).rstrip(b'=').decode()
        if digest != entry.hash.value:
            raise ValueError(f'SAM RECORD mismatch: {entry}')
        verified_files[str(actual)] = hex_digest
        if checkpoint:
            checkpoint({'kind': 'sam_record', **record})
        if str(entry).endswith('.py'):
            source_files_count += 1
        count += 1
    if not count or not source_files_count:
        raise ValueError('SAM RECORD has no hashed Python source')
    return {'commit': url['vcs_info']['commit_id'], 'direct_url_sha256': hashlib.sha256(direct.encode()).hexdigest(),
            'distribution_path': str(next(iter(found))), 'site_paths': [str(path) for path in site_paths],
            'record_hashed_files': count, 'record_hashed_sources': source_files_count,
            'verified_files': verified_files,
            'excluded_generated_bytecode': generated_bytecode,
            'bytecode_scope': 'cache bytes unverified; SAM worker must enforce source-only loading'}


def start_checkpoint(path: Path) -> Callable[[dict[str, Any]], None]:
    """Create a new provenance-only JSONL checkpoint; never read or resume an old one."""
    with path.open('x'):
        pass
    def append(record: dict[str, Any]) -> None:
        with path.open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
    return append


def cheap_assets(config: dict[str, Any], checkpoint: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Verify the small AA archive and SAM distribution before the large-file pilot."""
    entry = config['identity_files']['aa_archive']
    if set(entry) != {'path', 'sha256'} or entry['sha256'] != AA_SHA:
        raise ValueError('frozen AA archive identity required')
    archive = Path(entry['path']).resolve(strict=True)
    record = stable_file_hash(archive)
    if record['sha256'] != AA_SHA:
        raise ValueError('AA archive identity mismatch')
    checkpoint({'kind': 'aa_archive', **record})
    prefix = Path(config['runtimes']['main']['python']).resolve(strict=True).parents[1]
    if Path(sys.prefix).resolve() != prefix:
        raise ValueError('SAM identity check must run in MAIN environment')
    jcv_prefix = Path(config['runtimes']['jcv']['python']).resolve(strict=True).parents[1]
    aa = verify_aa_tree(archive, jcv_prefix / 'opt/Moore-AnimateAnyone', checkpoint=checkpoint)
    sam = verify_sam_distribution(prefix, checkpoint=checkpoint)
    return {'aa_installed': aa, 'sam_distribution': sam,
            'scope': 'small AA/SAM identity checks only; no Torch, models, GPU or full environment equivalence'}


def identity_projection(samples: list[float], total_bytes: int, used_cpu: float, limit: int) -> dict[str, Any]:
    if (not samples or any(rate < 0 for rate in samples) or total_bytes <= 0 or used_cpu < 0
            or type(limit) is not int or not 1 <= limit <= 125):
        raise ValueError('invalid identity pilot')
    projected = 2 * max(samples) * total_bytes + 5.0
    return {'sample_count': len(samples), 'worst_cpu_seconds_per_byte': max(samples),
            'fixed_aa_sam_cpu_reserve': 5, 'safety_factor': 2, 'used_cpu_seconds': used_cpu,
            'projected_cpu_seconds': projected, 'phase_cpu_limit': limit,
            'status': 'fit' if used_cpu + projected <= limit else 'stop',
            'scope': 'bounded CPU pilot; not a runtime guarantee'}


def asset_phase(config: dict[str, Any], output: Path, identity_cpu_limit: int = 20) -> None:
    """Hash declared assets and installed source under the bounded identity phase."""
    if type(identity_cpu_limit) is not int or not 1 <= identity_cpu_limit <= 125:
        raise ValueError('identity CPU limit must be between 1 and 125 seconds')
    entries = config['identity_files']
    required = {'main_archive': ENV_SHAS['main'], 'jcv_archive': ENV_SHAS['jcv'], 'aa_archive': AA_SHA}
    if not required.keys() <= entries.keys():
        raise ValueError('environment and AA archives required')
    checkpoint = start_checkpoint(output.parent / 'verified-checkpoint.jsonl')
    cheap = cheap_assets(config, checkpoint)
    aa, sam = cheap['aa_installed'], cheap['sam_distribution']
    inventory: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        path = Path(entry['path']).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f'identity input is not a file: {name}')
        inventory[name] = {'path': str(path), 'bytes': path.stat().st_size}
    for name, (protected_path, _) in PROTECTED.items():
        asset = Path(protected_path).resolve(strict=True)
        inventory[f'protected:{name}'] = {'path': str(asset), 'bytes': asset.stat().st_size}
    total_bytes = sum(v['bytes'] for v in inventory.values())
    with (output.parent / 'inventory.json').open('x') as stream:
        json.dump({'files': inventory, 'total_bytes': total_bytes,
                   'cpu_fit': 'unknown before measured bounded run'}, stream, sort_keys=True)
    samples = []
    for item in inventory.values():
        if item['bytes'] < 1024 * 1024:
            continue
        with Path(item['path']).open('rb') as stream:
            sample = stream.read(1024 * 1024)
        if len(sample) != 1024 * 1024:
            raise ValueError('identity input changed during pilot')
        before = resource.getrusage(resource.RUSAGE_SELF)
        hashlib.sha256(sample).digest()
        after = resource.getrusage(resource.RUSAGE_SELF)
        samples.append((after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime) / len(sample))
    if not samples:
        raise ValueError('identity hashing pilot has no representative large file')
    used = resource.getrusage(resource.RUSAGE_SELF)
    used_cpu = used.ru_utime + used.ru_stime
    pilot = identity_projection(samples, total_bytes, used_cpu, identity_cpu_limit)
    with (output.parent / 'pilot.json').open('x') as stream:
        json.dump(pilot, stream, sort_keys=True)
    if pilot['status'] != 'fit':
        raise RuntimeError('measured identity pilot does not fit the configured CPU-s bound')
    hashes = {}
    for name, entry in entries.items():
        if set(entry) != {'path', 'sha256'}:
            raise ValueError('invalid asset identity entry')
        if name in required and entry['sha256'] != required[name]:
            raise ValueError(f'changed frozen archive identity: {name}')
        path = Path(entry['path']).resolve(strict=True)
        record = stable_file_hash(path)
        if record['sha256'] != entry['sha256']:
            raise ValueError(f'asset hash mismatch: {name}')
        checkpoint({'kind': 'asset', 'name': name, **record})
        hashes[name] = {'path': str(path), 'sha256': record['sha256'], 'bytes': record['size']}
    protected = {}
    for group, (name, expected) in PROTECTED.items():
        if config['denied_paths'].get(group) != [name]:
            raise ValueError(f'protected probe input identity mismatch: {group}')
        record = stable_file_hash(Path(name).resolve(strict=True))
        if record['sha256'] != expected:
            raise ValueError(f'protected probe input identity mismatch: {group}')
        checkpoint({'kind': 'protected', 'name': group, **record})
        protected[group] = {'path': name, 'sha256': expected, 'device': record['device'], 'inode': record['inode']}
    output.write_text(json.dumps({'files': hashes, 'protected_inputs': protected, 'aa_installed': aa, 'sam_distribution': sam,
                                  'scope': 'archive hashes and AA/SAM file checks; full cached-env equivalence not proven'},
                                 sort_keys=True))


def _child_limit(seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, 165))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))


def _invoke(argv: list[str], env: dict[str, str], cwd: Path, output: Path, seconds: int, wall: int) -> dict[str, Any]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with (output / 'stdout').open('xb') as out, (output / 'stderr').open('xb') as err:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, cwd=cwd,
                                   env=env, close_fds=True, preexec_fn=lambda: _child_limit(seconds))
        (output / 'process.json').write_text(json.dumps({'pid': process.pid, 'argv': argv}, sort_keys=True))
        try:
            status = process.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            raise RuntimeError(f'bounded worker timed out: {process.pid}') from None
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {'pid': process.pid, 'status': status, 'wall_seconds': time.monotonic() - started,
              'cpu_seconds': (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)}
    (output / 'exit.json').write_text(json.dumps(result, sort_keys=True))
    if status:
        raise RuntimeError(f'worker failed: {process.pid}, status {status}; see {output}')
    return result


def worker_config(source: Path, worker_out: Path, allow_paths: list[str], denied_paths: dict[str, list[str]],
                  parent_user: str, parent_net: str, identity_receipt: Path) -> dict[str, Any]:
    return {'parent_user_namespace': parent_user, 'parent_network_namespace': parent_net,
            'allow_paths': [str(source), str(worker_out), *allow_paths], 'denied_paths': denied_paths,
            'cpu_seconds': 12, 'environment': {'CUDA_VISIBLE_DEVICES': '', 'TMPDIR': str(worker_out),
                                              'TMP': str(worker_out), 'TEMP': str(worker_out),
                                              'XDG_CACHE_HOME': str(worker_out / '.cache'),
                                              'TORCH_HOME': str(worker_out / '.cache' / 'torch')},
            'files': {}, 'tensor_fingerprints': {}, 'unit': '', 'output': str(worker_out),
            'worker_kind': 'preflight', 'protected_receipt': {'path': str(identity_receipt),
                                                             'sha256': sha(identity_receipt)}}


def wall_allowance(deadline: float, now: float, requested: int, remaining_workers: int) -> int:
    """Reserve each 120-second worker ceiling and five seconds of closeout."""
    if (not math.isfinite(deadline) or not math.isfinite(now) or type(requested) is not int
            or requested <= 0 or type(remaining_workers) is not int or remaining_workers < 0):
        raise ValueError('invalid wall deadline')
    available = math.floor(deadline - now - 120 * remaining_workers - 5)
    if available < 1:
        raise RuntimeError('insufficient remaining wall time for reserved stages')
    return min(requested, available)


def run(config: dict[str, Any], archive_sha: str, out: Path, identity_cpu_limit: int = 20,
        identity_wall_limit: int = 120, deadline_monotonic: float | None = None) -> dict[str, Any]:
    if type(identity_cpu_limit) is not int or not 1 <= identity_cpu_limit <= 125:
        raise ValueError('identity CPU limit must be between 1 and 125 seconds')
    if type(identity_wall_limit) is not int or not 1 <= identity_wall_limit <= 1200:
        raise ValueError('identity wall limit must be between 1 and 1200 seconds')
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux one-core affinity required')
    if set(config) != {'source_archive', 'source_tree', 'runtimes', 'identity_files', 'denied_paths'}:
        raise ValueError('unexpected preflight config fields')
    source = Path(config['source_tree']).resolve(strict=True)
    local = Path('/local/users/emanuele').resolve(strict=True)
    if not source.is_relative_to(local) or not out.resolve().is_relative_to(local):
        raise ValueError('host-local source and output required')
    if set(config['runtimes']) != {'main', 'jcv'}:
        raise ValueError('both frozen runtimes required')
    if any(out.iterdir()):
        raise ValueError('new empty output required')
    resource.setrlimit(resource.RLIMIT_CPU, (6, 165))
    archive = Path(config['source_archive']).resolve(strict=True)
    if not archive.is_relative_to(local):
        raise ValueError('source archive must be host-local')
    source_index = source_files(archive, source, archive_sha)
    if str(Path(__file__).resolve()) not in source_index:
        raise ValueError('supervisor is not part of frozen source archive')
    groups = config['denied_paths']
    if set(groups) != set(PROTECTED) or any(groups.get(k) != [v[0]] for k, v in PROTECTED.items()):
        raise ValueError('exact protected input paths required')
    aliases = out / 'aliases'
    aliases.mkdir()
    alias_paths = []
    for group, paths in sorted(groups.items()):
        if not isinstance(paths, list) or not paths:
            raise ValueError('every protected group requires actual paths')
        for i, path in enumerate(paths):
            target = Path(path).resolve(strict=True)
            alias = aliases / f'{group}-{i}'
            alias.symlink_to(target)
            alias_paths.append(str(alias))
    denied_paths = {**groups, 'aliases': alias_paths}
    identities = out / 'identities'
    identities.mkdir()
    frozen_config = out / 'config.json'
    frozen_config.write_text(json.dumps(config, sort_keys=True))
    base_env = {'PYTHONPATH': str(source), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
    main_python = Path(config['runtimes']['main']['python']).resolve(strict=True)
    main_prefix = local / 'pointstream' / 'cache' / ENV_SHAS['main'] / 'tree'
    if not main_python.is_relative_to(main_prefix.resolve(strict=True)):
        raise ValueError('MAIN interpreter outside frozen cache')
    command = [str(main_python), str(Path(__file__).resolve()), '--assets', str(frozen_config),
               '--assets-output', str(identities / 'assets.json'), '--identity-cpu-seconds', str(identity_cpu_limit)]
    identity_wall = (wall_allowance(deadline_monotonic, time.monotonic(), identity_wall_limit, 2)
                     if deadline_monotonic is not None else identity_wall_limit)
    asset_usage = _invoke(command, base_env, source, identities, identity_cpu_limit, identity_wall)
    parent_user = os.readlink('/proc/self/ns/user')
    parent_net = os.readlink('/proc/self/ns/net')
    results = {}
    for name in ('main', 'jcv'):
        runtime = config['runtimes'][name]
        if set(runtime) != {'python', 'allow_paths'}:
            raise ValueError('runtime requires exact python and explicit allow paths')
        python = Path(runtime['python']).resolve(strict=True)
        prefix = local / 'pointstream' / 'cache' / ENV_SHAS[name] / 'tree'
        if not python.is_relative_to(prefix.resolve(strict=True)):
            raise ValueError('runtime interpreter outside declared cached environment')
        stage = out / name
        stage.mkdir()
        worker_out = stage / 'worker'
        worker_out.mkdir()
        worker_spec = worker_config(source, worker_out, runtime['allow_paths'], denied_paths,
                                    parent_user, parent_net, identities / 'assets.json')
        config_file = stage / 'worker-config.json'
        config_file.write_text(json.dumps(worker_spec, sort_keys=True))
        argv = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', str(python), '-m',
                'experiments.tennis.w8_receiver_bootstrap', '--config', str(config_file),
                '--config-sha256', sha(config_file), '--cpu-seconds', '12']
        remaining_workers = 1 if name == 'main' else 0
        wall = (wall_allowance(deadline_monotonic, time.monotonic(), 120, remaining_workers)
                if deadline_monotonic is not None else 120)
        usage = _invoke(argv, base_env, source, stage, 12, wall)
        receipt = json.loads((worker_out / 'preflight.json').read_text())
        if (receipt['torch'] != EXPECTED_TORCH[name] or receipt['cuda_initialized'] is not False
                or receipt['visible_devices'] != '' or Path(receipt['executable']).resolve() != python
                or not Path(receipt['prefix']).resolve().is_relative_to(prefix.resolve())):
            raise ValueError(f'runtime identity mismatch: {name}')
        closure = receipt['source_modules']
        if not closure:
            raise ValueError('empty loaded source closure')
        for module in closure.values():
            if source_index.get(module['path']) != module['sha256']:
                raise ValueError(f'loaded source differs from frozen archive: {module["path"]}')
        isolation = receipt['isolation']
        wanted = {p for paths in denied_paths.values() for p in paths}
        if (set(isolation['denials']) != wanted or any(code not in (1, 13) for code in isolation['denials'].values())
                or isolation['pre_sandbox_threads'] != 1 or isolation['extra_fds'] is not False
                or isolation['network_interfaces'] != ['lo'] or isolation['landlock_abi'] < 1
                or isolation['namespaces']['user'] == parent_user or isolation['namespaces']['net'] == parent_net
                or isolation['cpu_bootstrap']['before_landlock_sha256'] != isolation['cpu_bootstrap']['after_landlock_sha256']
                or isolation['cpu_bootstrap']['cuda_initialized_before_isolation'] is not False):
            raise ValueError(f'isolation receipt mismatch: {name}')
        results[name] = {'usage': usage, 'receipt_sha256': sha(worker_out / 'preflight.json')}
    return {'source_archive_sha256': archive_sha, 'source_files': len(source_index),
            'assets': asset_usage, 'runtimes': results,
            'scope': 'CPU isolation/runtime probes; no GPU, model run, or cached-environment equivalence'}


def run_cheap(config_path: Path, config_sha256: str, output: Path, local_root: Path) -> None:
    """Write only new bounded AA/SAM receipts under a verified host-local root."""
    local = local_root.resolve(strict=True)
    if not output.is_absolute() or not output.parent.resolve(strict=True).is_relative_to(local):
        raise ValueError('cheap result must be in existing host-local directory')
    raw = config_path.read_bytes()
    if len(raw) > 100_000 or hashlib.sha256(raw).hexdigest() != config_sha256:
        raise ValueError('cheap config identity mismatch')
    checkpoint_path = output.with_name(output.name + '.checkpoint.jsonl')
    if os.path.lexists(output) or os.path.lexists(checkpoint_path):
        raise ValueError('cheap output and checkpoint must be new')
    checkpoint = start_checkpoint(checkpoint_path)
    result = cheap_assets(json.loads(raw), checkpoint)
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--source-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--assets-output', type=Path)
    parser.add_argument('--identity-cpu-seconds', type=int, default=20)
    parser.add_argument('--identity-wall-seconds', type=int, default=120)
    parser.add_argument('--deadline-monotonic', type=float)
    parser.add_argument('--cheap-config', type=Path)
    parser.add_argument('--cheap-config-sha256')
    parser.add_argument('--cheap-output', type=Path)
    args = parser.parse_args()
    if args.cheap_config or args.cheap_output or args.cheap_config_sha256:
        if not args.cheap_config or not args.cheap_output or not args.cheap_config_sha256 or any(
                (args.assets, args.assets_output, args.config, args.source_sha256, args.output)):
            parser.error('cheap mode requires only cheap-config, cheap-config-sha256 and cheap-output')
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        run_cheap(args.cheap_config, args.cheap_config_sha256, args.cheap_output, Path('/local/users/emanuele'))
        return 0
    if args.assets and args.assets_output and not any((args.config, args.source_sha256, args.output)):
        if not 1 <= args.identity_cpu_seconds <= 125:
            raise ValueError('identity CPU limit must be between 1 and 125 seconds')
        resource.setrlimit(resource.RLIMIT_CPU, (args.identity_cpu_seconds, 165))
        asset_phase(json.loads(args.assets.read_text()), args.assets_output, args.identity_cpu_seconds)
        return 0
    if not args.config or not args.source_sha256 or not args.output or args.assets or args.assets_output:
        parser.error('config, source-sha256 and output required')
    config = json.loads(args.config.read_text())
    result = run(config, args.source_sha256, args.output, args.identity_cpu_seconds, args.identity_wall_seconds,
                 args.deadline_monotonic)
    (args.output / 'result.json').write_text(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
