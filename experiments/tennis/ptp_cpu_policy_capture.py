"""Capture a strict CPU-only derivative of the reviewed gpu6 PTP policy.

No traversal, ldd, model read, inference, downloads or remote calls. Only the
explicit reviewed logical OS roles below may resolve to host-version targets.
The resulting new path/hash inventory requires root review before qualification.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import resource
import re
import signal
import stat
import sys
import time
from pathlib import Path
from typing import Any

BASE_SHA = '4fa67547b1288e46f979040f0de5315288bd8bfc5d6065e3b6abaeeee32bea1c'
RUNTIME_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
PACKED = '/local/users/emanuele/pointstream/cache/' + RUNTIME_SHA + '/tree'
HOSTS = {'gpu1', 'gpu3', 'gpu5'}
OS_ROOT = '/usr/lib/x86_64-linux-gnu/'
OS_LIBRARIES = tuple(OS_ROOT + name for name in (
    'ld-linux-x86-64.so.2', 'libc.so.6', 'libdl.so.2', 'libgcc_s.so.1', 'libm.so.6',
    'libpthread.so.0', 'librt.so.1', 'libstdc++.so.6.0.30', 'libutil.so.1', 'libz.so.1.2.11'))
# Ten paths; no new soname/version role is discovered when a host differs.
OS_TARGET_ROOTS = ('/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu', '/lib64')
PACKED_EXECUTABLES = (PACKED + '/bin/python3.12', PACKED + '/bin/dav1d')
LOADER = OS_ROOT + 'ld-linux-x86-64.so.2'
UNSHARE_ROLES = ('/usr/bin/unshare', '/bin/unshare')
DROP_SYSTEM = (
    '/local/users/emanuele/envs/20261006b/root/pointstream-build/lib/libgcc_s.so.1',
    '/local/users/emanuele/envs/20261006b/root/pointstream-build/lib/libstdc++.so.6.0.36',
    '/local/users/emanuele/envs/20261006b/root/pointstream-build/targets/x86_64-linux/lib/libcudart.so.12.8.90',
    '/local/users/emanuele/pointstream/cache/934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3/cvpr2026_video_htl.pth.tar',
    '/local/users/emanuele/pointstream/cache/b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302/cvpr2026_image.pth.tar',
    *[OS_ROOT + name for name in ('libcuda.so.535.183.01', 'libnvidia-ml.so.535.183.01',
                                  'libnvidia-nvvm.so.535.183.01', 'libnvidia-ptxjitcompiler.so.535.183.01')],
)
CPU_LEAVES = ('/etc/ld.so.cache', '/proc/cpuinfo', '/proc/self/maps', '/proc/self/status', '/proc/stat',
              '/proc/sys/vm/mmap_min_addr', '/sys/devices/system/cpu/online',
              '/sys/devices/system/memory/block_size_bytes', '/dev/null', '/dev/urandom')
DROP_LEAVES = ('/proc/devices', '/proc/driver/nvidia/capabilities/mig/config',
               '/proc/driver/nvidia/capabilities/mig/monitor', '/proc/driver/nvidia/params',
               '/proc/driver/nvidia/version', '/dev/nvidia-modeset', '/dev/nvidia-uvm',
               '/dev/nvidia-uvm-tools', '/dev/nvidia0', '/dev/nvidia1', '/dev/nvidiactl',
               '/dev/nvidia-caps/nvidia-cap1', '/dev/nvidia-caps/nvidia-cap2')


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_sha(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def file_identity(logical: str) -> dict[str, Any]:
    if logical not in {*OS_LIBRARIES, *PACKED_EXECUTABLES, *UNSHARE_ROLES}:
        raise ValueError('unreviewed read path; stop rather than expand policy')
    path = Path(logical)
    resolved = path.resolve(strict=True)
    if logical in PACKED_EXECUTABLES:
        if resolved != path:
            raise ValueError('packed executable retargeted')
    elif logical in UNSHARE_ROLES:
        if resolved.parent not in (Path('/usr/bin'), Path('/bin')):
            raise ValueError('namespace utility resolves outside reviewed binary roots')
    elif not any(resolved.is_relative_to(Path(root)) for root in OS_TARGET_ROOTS):
        raise ValueError('logical OS role resolves outside reviewed OS library roots')
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 64 * 1024**2:
        raise ValueError('unexpected library type/size')
    with resolved.open('rb') as source:
        sha = hashlib.file_digest(source, 'sha256').hexdigest()
    after = resolved.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or path.resolve(strict=True) != resolved):
        raise ValueError('resource changed during capture')
    return {'path': str(resolved), 'logical_path': logical, 'resolved_path': str(resolved),
            'sha256': sha, 'size': before.st_size}


def check_leaf(row: dict[str, Any]) -> None:
    if row['path'] not in CPU_LEAVES:
        raise ValueError('unreviewed CPU leaf')
    info = Path(row['path']).stat()
    if row['kind'] not in ('regular', 'character') or not (
        stat.S_ISREG(info.st_mode) if row['kind'] == 'regular' else stat.S_ISCHR(info.st_mode)
    ):
        raise ValueError('CPU leaf absent or changed type')


def _paths(rows: list[dict[str, Any]]) -> set[str]:
    paths = [row['path'] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate reviewed resource')
    return set(paths)


def kernel_metadata() -> dict[str, Any]:
    if sys.platform != 'linux':
        raise ValueError('Landlock metadata requires Linux')
    library = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    abi = int(library.syscall(444, 0, 0, 1))
    return {'kernel_release': os.uname().release, 'landlock_abi': abi, 'errno': ctypes.get_errno(),
            'namespace_execution': 'not tested; requires real bounded qualification smoke'}


def capture(base_bytes: bytes, host: str) -> dict[str, Any]:
    if digest(base_bytes) != BASE_SHA or host not in HOSTS:
        raise ValueError('reviewed base policy hash or eligible CPU host differs')
    base = json.loads(base_bytes)
    if base.get('schema') != 1 or set(base['profiles']) != {'gpu6'}:
        raise ValueError('expected reviewed gpu6-only policy')
    original = base['profiles']['gpu6']
    if (_paths(original['system_files']) != set(OS_LIBRARIES) | set(DROP_SYSTEM)
            or _paths(original['executables']) != set(PACKED_EXECUTABLES) | {LOADER}
            or _paths(original['system_leaves']) != set(CPU_LEAVES) | set(DROP_LEAVES)
            or original['python'] != PACKED_EXECUTABLES[0]
            or original['unshare']['path'] not in UNSHARE_ROLES):
        raise ValueError('base resources differ from explicit reviewed path lists; stop')
    started, cpu = time.monotonic(), time.process_time()
    identities = {name: file_identity(name) for name in (*OS_LIBRARIES, *PACKED_EXECUTABLES)}
    unshare = file_identity(original['unshare']['path'])
    metadata = kernel_metadata()
    if metadata['landlock_abi'] < 3:
        raise ValueError('Landlock ABI >=3 unavailable; no qualification admitted')
    for old in original['executables']:
        if old['path'] in PACKED_EXECUTABLES:
            new = identities[old['path']]
            if (new['sha256'], new['size']) != (old['sha256'], old['size']):
                raise ValueError('packed Python/dav1d identity differs; no runtime replacement')
    profile = json.loads(json.dumps(original))
    profile['unshare'] = unshare
    profile['system_files'] = [identities[name] for name in OS_LIBRARIES]
    profile['executables'] = [dict(identities[name]) for name in (*PACKED_EXECUTABLES, LOADER)]
    profile['system_leaves'] = [row for row in profile['system_leaves'] if row['path'] in CPU_LEAVES]
    for row in profile['system_leaves']:
        check_leaf(row)
    # Runtime/source/import fields are copied unchanged; unshare is recaptured. Source inventory
    # is subsequently amended by the already reviewed PTP adapter, as before.
    result = json.loads(json.dumps(base))
    result['profiles'] = {host: profile}
    result['cpu_derivation'] = {'schema': 1, 'base_policy_sha256': BASE_SHA, 'runtime_archive_sha256': RUNTIME_SHA,
        'hosts': [host], 'runtime_inventory_sha256': object_sha(original['runtime_files']),
        'source_amendment': 'unchanged ptp_isolated_receiver.amend_policy; no new import/resource discovery',
        'dropped_system_paths': list(DROP_SYSTEM), 'dropped_leaf_paths': list(DROP_LEAVES),
        'retained_logical_OS_roles': list(OS_LIBRARIES), 'retained_CPU_leaves': list(CPU_LEAVES),
        'scope': 'explicit CPU-only generic resource derivative; captured host OS identities require root review; not an OS-denial witness',
        'kernel_preflight': metadata, 'namespace_utility': unshare,
        'capture_wall_seconds': time.monotonic() - started, 'capture_cpu_seconds': time.process_time() - cpu}
    validate_derivation(result, base_bytes)
    return result


def _captured_row(row: dict[str, Any], logical: str, *, packed: bool = False, utility: bool = False) -> None:
    if (not isinstance(row, dict) or set(row) != {'path', 'logical_path', 'resolved_path', 'sha256', 'size'}
            or row['logical_path'] != logical or row['resolved_path'] != row['path']
            or not isinstance(row['path'], str) or not Path(row['path']).is_absolute()
            or str(Path(row['path'])) != row['path'] or '..' in Path(row['path']).parts
            or type(row['size']) is not int or not 0 < row['size'] <= 64 * 1024**2
            or not isinstance(row['sha256'], str) or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None):
        raise ValueError('invalid captured resource identity')
    target = Path(row['path'])
    if packed:
        if row['path'] != logical:
            raise ValueError('packed executable retargeted')
    elif utility:
        if logical not in UNSHARE_ROLES or target.parent not in (Path('/usr/bin'), Path('/bin')) or target.name != 'unshare':
            raise ValueError('namespace utility role expanded')
    else:
        if logical not in OS_LIBRARIES or not any(target.is_relative_to(Path(root)) for root in OS_TARGET_ROOTS):
            raise ValueError('OS library role expanded')
        name = Path(logical).name
        if logical == LOADER:
            allowed = target.name == name or re.fullmatch(r'ld-[0-9.]+\.so', target.name) is not None
        else:
            family, abi = name.split('.so.', 1)
            major = abi.split('.')[0]
            allowed = (re.fullmatch(re.escape(family) + r'\.so\.' + major + r'(?:\.\d+)*', target.name) is not None
                       or re.fullmatch(re.escape(family) + r'-[0-9.]+\.so', target.name) is not None)
        if not allowed:
            raise ValueError('OS logical role resolves to a different resource family')


def validate_derivation(config: dict[str, Any], base_bytes: bytes) -> None:
    """Pure artifact validation; no current-host file reads or path resolution."""
    if digest(base_bytes) != BASE_SHA:
        raise ValueError('reviewed base policy hash differs')
    base = json.loads(base_bytes)
    if base.get('schema') != 1 or set(base['profiles']) != {'gpu6'}:
        raise ValueError('reviewed gpu6-only base required')
    original = base['profiles']['gpu6']
    if (set(config) != set(base) | {'cpu_derivation'} or config.get('schema') != 1
            or any(config[k] != base[k] for k in base if k != 'profiles')
            or not config['profiles'] or not set(config['profiles']) <= HOSTS):
        raise ValueError('derived policy host/schema/metadata expanded')
    derivation = config['cpu_derivation']
    if (derivation.get('schema') != 1 or derivation.get('base_policy_sha256') != BASE_SHA
            or derivation.get('runtime_archive_sha256') != RUNTIME_SHA
            or derivation.get('runtime_inventory_sha256') != object_sha(original['runtime_files'])
            or derivation.get('hosts') != list(config['profiles'])
            or derivation.get('dropped_system_paths') != list(DROP_SYSTEM)
            or derivation.get('dropped_leaf_paths') != list(DROP_LEAVES)
            or derivation.get('retained_logical_OS_roles') != list(OS_LIBRARIES)
            or derivation.get('retained_CPU_leaves') != list(CPU_LEAVES)):
        raise ValueError('CPU derivation provenance/subset differs')
    captures = derivation.get('captures', [derivation])
    if len(captures) != len(config['profiles']):
        raise ValueError('per-host capture metadata incomplete')
    for host, metadata in zip(config['profiles'], captures):
        if (metadata.get('hosts') != [host] or metadata.get('namespace_utility') != config['profiles'][host]['unshare']
                or metadata.get('kernel_preflight', {}).get('landlock_abi', 0) < 3):
            raise ValueError('per-host namespace utility/kernel identity differs')
    changing = {'system_files', 'executables', 'system_leaves', 'unshare'}
    leaves = [r for r in original['system_leaves'] if r['path'] in CPU_LEAVES]
    for profile in config['profiles'].values():
        if set(profile) != set(original) or any(profile[k] != original[k] for k in original if k not in changing):
            raise ValueError('packed runtime/source/import inventory changed')
        if profile['system_leaves'] != leaves or _paths(profile['system_leaves']) != set(CPU_LEAVES):
            raise ValueError('CPU leaf policy expanded or altered')
        system = profile['system_files']
        if len(system) != len(OS_LIBRARIES) or {r.get('logical_path') for r in system} != set(OS_LIBRARIES):
            raise ValueError('OS library logical roles expanded')
        for row in system:
            _captured_row(row, row['logical_path'])
        execute = profile['executables']
        if len(execute) != 3 or {r.get('logical_path') for r in execute} != {*PACKED_EXECUTABLES, LOADER}:
            raise ValueError('executable policy expanded')
        for row in execute:
            logical = row['logical_path']
            _captured_row(row, logical, packed=logical in PACKED_EXECUTABLES)
            if logical in PACKED_EXECUTABLES:
                old = next(r for r in original['executables'] if r['path'] == logical)
                if (row['sha256'], row['size']) != (old['sha256'], old['size']):
                    raise ValueError('packed executable identity changed')
            elif row != next(r for r in system if r['logical_path'] == LOADER):
                raise ValueError('executed loader differs from captured library identity')
        _captured_row(profile['unshare'], original['unshare']['path'], utility=True)


def verify_links(config: dict[str, Any], host: str | None = None) -> None:
    """Refuse any logical-role retargeting after capture, without following new reads."""
    host = host or os.uname().nodename.split('.')[0]
    if host not in HOSTS or host not in config['profiles']:
        raise ValueError('current host absent from captured CPU policy')
    for profile in [config['profiles'][host]]:
        for row in [*profile['system_files'], *profile['executables'], profile['unshare']]:
            logical = row.get('logical_path')
            if logical not in {*OS_LIBRARIES, *PACKED_EXECUTABLES, *UNSHARE_ROLES} or row.get('resolved_path') != row['path']:
                raise ValueError('captured logical role metadata missing/unreviewed')
            if str(Path(logical).resolve(strict=True)) != row['resolved_path']:
                raise ValueError('captured logical role retargeted')


def merge(captures: list[dict[str, Any]], base_bytes: bytes) -> dict[str, Any]:
    if not captures:
        raise ValueError('one or more explicitly captured profiles required')
    for config in captures:
        validate_derivation(config, base_bytes)
    result = json.loads(json.dumps(captures[0]))
    profiles: dict[str, Any] = {}
    for config in captures:
        derivation = config.get('cpu_derivation', {})
        if (derivation.get('base_policy_sha256') != BASE_SHA or derivation.get('runtime_archive_sha256') != RUNTIME_SHA
                or len(config['profiles']) != 1):
            raise ValueError('capture provenance differs')
        host, profile = next(iter(config['profiles'].items()))
        if host not in HOSTS or host in profiles:
            raise ValueError('unknown or duplicate captured host')
        reference = next(iter(captures[0]['profiles'].values()))
        for key in ('runtime_root', 'runtime_prefix', 'runtime_import_paths', 'runtime_files', 'source_root', 'source_files', 'python'):
            if profile[key] != reference[key]:
                raise ValueError('generic runtime/source/import inventory differs across hosts')
        profiles[host] = profile
    result['profiles'] = profiles
    result['cpu_derivation']['hosts'] = list(profiles)
    result['cpu_derivation']['captures'] = [c['cpu_derivation'] for c in captures]
    validate_derivation(result, base_bytes)
    return result


def write_new(path: Path, value: Any) -> None:
    with path.open('x') as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('capture', 'merge', 'verify-links'))
    parser.add_argument('--base', type=Path)
    parser.add_argument('--input', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'capture':
        if args.base is None or args.out is None or sys.platform != 'linux':
            parser.error('Linux capture requires --base and exclusive --out')
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        signal.alarm(120)
        result = capture(args.base.read_bytes(), os.uname().nodename.split('.')[0])
    elif args.command == 'merge':
        if not args.input or args.out is None or args.base is None:
            parser.error('merge requires authenticated --base, --input capture(s) and exclusive --out')
        result = merge([json.loads(p.read_text()) for p in args.input], args.base.read_bytes())
    else:
        if len(args.input) != 1:
            parser.error('verify-links requires one --input policy')
        verify_links(json.loads(args.input[0].read_text()))
        print(json.dumps({'logical_links_match_capture': True}))
        return
    write_new(args.out, result)
    print(json.dumps({'output': str(args.out), 'sha256': digest(args.out.read_bytes()), 'hosts': list(result['profiles'])}))


if __name__ == '__main__':
    main()
