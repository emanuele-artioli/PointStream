"""Fleet supervisor for the registered GVC-RT Clip3 4-to-16-frame pilot.

No job is submitted by this module. Sender, probe, isolated receiver and scorer
run as independent children; this module never imports the authored `src`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import stat
import signal
import shutil
import subprocess
import sys

from experiments.visor.gvcrt_entropy_pilot import kill_owned_tree
from experiments.generative import gvcrt_device_guard as device_guard
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_SHA = 'a7ad708cfaea7d0030cc1d78bd2cd1f1d62a1df999d8eaece292bff510c30f7e'
OLD_SOURCE_SHA = '748591675c350fe3e0214b0a4aca4529b56199419f0c4838b167e6e52440ecba'
DATA_SHA = 'bb45372d011ef4092d78173169c8ad6280d7dbdb9998efaf2b1c72a63ca1dd18'
LABEL_SHA = '247c81967bfb05b53f2b65c0a691a7f14c98802ca93c95a68298b2244b5ba72f'
I_SHA = 'c8a840415cc028307131c66f68076b7e9f6fcc4db7fed423c62e8b202f83a22d'
P_SHA = '7a89bd342a1484b3da7dfc3f03834ff81e0e4180e02f8a286dec0d1025c99f4e'
ENTROPY_SHA = 'f0951974be2a1168f87a67f8d3683844874aa1902b2d135cde46c2e5b915262e'
BASELINE_SHA = 'dc1bc14133cf210589f2452785fbaf6d4905e91aae05fdc6d7ac0f25da8ec066'
MARKER_SHA = 'b211cbcc4280acad546a41283ec95c63c7ca95c8d0fd6fdb07af7120932cbe97'
BRIDGE_SHA = 'cf42d1a115d905ed5faf3fdb01a8c7aa66c728091a18880c8cbf818cbcfbfd61'
RUNTIME_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
HERE = Path(__file__).resolve().parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path: Path, obj: Any) -> None:
    with path.open('x') as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def projection(sender: dict[str, Any], receiver: dict[str, Any], times: dict[str, float],
               remaining: float) -> dict[str, float | bool]:
    """One I + 15 P, including two feature resets; four frames cover every P shift."""
    if (sender.get('encode_full_timing_scope') != 'source_hash_read_rgb_pad_transfer_compress_packet_per_frame'
            or sender.get('decode_full_timing_scope') != 'packet_parse_decode_crop_cpu_transfer_per_frame'
            or receiver.get('decode_full_timing_scope') != 'packet_parse_decode_crop_cpu_transfer_per_frame'):
        raise ValueError('complete per-frame timing scope required')
    encode = sender['encode_full_seconds_per_frame']
    reference = sender['reference_decode_full_seconds_per_frame']
    decode = receiver['decode_full_seconds_per_frame']
    if any(len(x) != 4 or any(not math.isfinite(t) or t < 0 for t in x)
           for x in (encode, reference, decode)):
        raise ValueError('four finite per-frame timings required')
    core = (sender['encode_seconds_per_frame'], sender['reference_decode_seconds_per_frame'],
            receiver['decode_seconds_per_frame'])
    if any(len(x) != 4 or any(not math.isfinite(t) or t < 0 for t in x) for x in core):
        raise ValueError('four finite codec-only timings required')
    if any(full < partial for entire, inside in zip((encode, reference, decode), core)
           for full, partial in zip(entire, inside)):
        raise ValueError('full-frame timing shorter than contained codec timing')
    def estimate(values: list[float]) -> float:
        return values[0] + 13 * max(values[2:4]) + 2 * values[1]
    for key in ('staging', 'prep', 'probe', 'sender', 'receiver', 'score', 'output_copy',
                'validation_reserve', 'release_reserve'):
        if not math.isfinite(times[key]) or times[key] < 0:
            raise ValueError('invalid process or reserve timing')
    sender_arrays = sender['reference_stack_seconds'] + sender['reference_save_seconds']
    receiver_arrays = receiver['stack_seconds'] + receiver['save_seconds']
    if any(not math.isfinite(x) or x < 0 for x in (sender_arrays, receiver_arrays)):
        raise ValueError('invalid reconstruction stack/save timing')
    if sum(encode) + sum(reference) + sender_arrays > times['sender'] or sum(decode) + receiver_arrays > times['receiver']:
        raise ValueError('frame timings exceed enclosing worker wall time')
    variable_four = sum(encode) + sum(reference) + sum(decode)
    variable_sixteen = sum(estimate(x) for x in (encode, reference, decode))
    array_four = sender_arrays + receiver_arrays
    fixed = max(0.0, times['sender'] + times['receiver'] - variable_four - array_four)
    # All measured prep/probe/score/validation/release costs are conservatively
    # repeated; scoring is scaled by 4 because it processes all frames.
    support = (times['staging'] + times['prep'] + times['probe'] + 4 * times['score']
               + 4 * times['output_copy'] + 4 * array_four + times['validation_reserve'] + times['release_reserve'])
    forecast = 1.5 * (fixed + variable_sixteen + support)
    cpu_upper = forecast  # fleet enforces one CPU for the complete campaign
    return {'fixed_seconds': fixed, 'variable_16_seconds': variable_sixteen, 'array_four_seconds': array_four,
            'support_seconds': support,
            'forecast_gpu_seconds': forecast, 'forecast_cpu_seconds': cpu_upper,
            'remaining_seconds': remaining,
            'passes': forecast <= 500 and forecast <= remaining and cpu_upper <= remaining}


def terminate(_signal: int, _frame: Any) -> None:
    raise KeyboardInterrupt('fleet requested termination')


def run_child(command: list[str], name: str, stage: Path, env: dict[str, str], timeout: float) -> dict[str, Any]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with (stage / f'{name}.stdout').open('xb') as stdout, (stage / f'{name}.stderr').open('xb') as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                env=env, close_fds=True)
        timed_out = False
        interrupted = False
        try:
            code = proc.wait(timeout=timeout)
        except BaseException as exc:
            kill_owned_tree(proc.pid)
            code = proc.wait()
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            interrupted = not timed_out
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    ended = time.monotonic()
    r = {'name': name, 'argv': command, 'pid': proc.pid, 'status': code, 'timed_out': timed_out,
         'started_monotonic': started, 'ended_monotonic': ended, 'wall_seconds': ended - started,
         'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
         'stdout_sha256': sha(stage / f'{name}.stdout'), 'stderr_sha256': sha(stage / f'{name}.stderr')}
    r['interrupted'] = interrupted
    save(stage / f'{name}.exit.json', r)
    if interrupted:
        raise KeyboardInterrupt('child terminated with fleet request')
    return r


def staged(dispatch: dict[str, Any], name: str, expected: str) -> Path:
    found = [x for x in dispatch['staged_inputs'] if x.get('name') == name]
    if len(found) != 1 or found[0].get('sha256') != expected:
        raise ValueError(f'staged {name} identity mismatch')
    return Path(found[0]['path']).resolve(strict=True)


def unpack(archive: Path, destination: Path) -> None:
    destination.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter='data')


def env_for(cwd: Path, author: Path, entropy: Path) -> dict[str, str]:
    return {'PATH': '/usr/bin:/bin', 'HOME': str(cwd), 'TMPDIR': str(cwd), 'TMP': str(cwd), 'TEMP': str(cwd),
            'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
            'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'PYTHONPATH': os.pathsep.join([str(author), str(entropy.parent)])}


def aliases_for(protected: dict[str, list[str]], alias_dir: Path) -> dict[str, list[str]]:
    alias_dir.mkdir()
    aliases = []
    for name, paths in protected.items():
        for index, path in enumerate(paths):
            link = alias_dir / f'{name}-{index}'
            link.symlink_to(Path(path).resolve(strict=True))
            aliases.append(str(link))
    return {**protected, 'aliases': aliases}


def device_paths() -> list[str]:
    # Exact c33fa60e-qualified W8 generic runtime/device list; no broad /proc or /sys.
    paths = ['/usr/lib', '/lib', '/lib64', '/dev/nvidia-modeset', '/dev/nvidia-uvm',
             '/dev/nvidia-uvm-tools', '/dev/nvidia0', '/dev/nvidia1', '/dev/nvidiactl',
             '/dev/nvidia-caps/nvidia-cap1', '/dev/nvidia-caps/nvidia-cap2',
             '/dev/null', '/dev/urandom', '/proc/driver/nvidia/version']
    if any(not Path(path).exists() for path in paths):
        raise ValueError('a qualified runtime/device grant is unavailable')
    return paths


def inspect_gpu(visible: str) -> dict[str, str]:
    query = subprocess.run(['/usr/bin/nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True, timeout=10)
    rows = [line.split(', ', 1) for line in query.stdout.splitlines() if line.strip()]
    found = [row for row in rows if len(row) == 2 and row[0] == visible]
    if len(found) != 1:
        raise ValueError('claimed physical GPU identity not found')
    return {'uuid': found[0][0], 'name': found[0][1]}


def tree_identity(tree: Path) -> str:
    files = sorted(p for p in tree.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml'))
    if not files:
        raise ValueError('empty source tree')
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(tree).as_posix().encode() + b'\0' + bytes.fromhex(sha(p)))
    return h.hexdigest()


def archive_tree_identity(archive_path: Path) -> str:
    entries: dict[str, str] = {}
    with tarfile.open(archive_path) as tar:
        for member in tar:
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('unsafe code archive')
            if p.suffix not in ('.py', '.yaml', '.yml'):
                continue
            if not member.isfile() or p.as_posix() in entries:
                raise ValueError('invalid code archive member')
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError('unreadable code archive member')
            entries[p.as_posix()] = hashlib.sha256(stream.read()).hexdigest()
    if not entries:
        raise ValueError('empty code archive')
    h = hashlib.sha256()
    for name, digest in sorted(entries.items()):
        h.update(name.encode() + b'\0' + bytes.fromhex(digest))
    return h.hexdigest()


def verify_metric_sources(dispatch: dict[str, Any], inputs: dict[str, Path]) -> dict[str, str]:
    verified = {}
    for name in ('balltrack_code', 'wasb_code'):
        entry = next(x for x in dispatch['staged_inputs'] if x['name'] == name)
        archive = inputs[name].parent / Path(entry['source']).name
        if sha(archive) != METRIC_INPUTS[name]:
            raise ValueError('metric code archive mismatch')
        found = tree_identity(inputs[name])
        if found != archive_tree_identity(archive):
            raise ValueError('metric cached code tree differs from archive')
        verified[name] = found
    if sha(inputs['wasb_code'] / 'src/configs/dataset/tennis.yaml') != METRIC_INPUTS['wasb_config']:
        raise ValueError('WASB config differs from frozen config')
    if sha(inputs['wasb_weights'] / 'wasb_tennis_best.pth.tar') != '9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb':
        raise ValueError('WASB weight differs from frozen checkpoint')
    return verified


def receiver_config(runtime: Path, author: Path, bridge: Path, entropy: Path, i_model: Path, p_model: Path,
                    wire: Path, output: Path, denied: dict[str, list[str]], canary: Path) -> dict[str, Any]:
    loader = Path('/etc/ld.so.cache').resolve(strict=True)
    info = loader.stat()
    if str(loader) != '/etc/ld.so.cache' or not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & stat.S_IWOTH:
        raise ValueError('unsafe host loader cache')
    loader_identity = {'device': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode, 'uid': info.st_uid}
    return {'policy_profile': 'gvcrt-w8-twelve-leaf-mutation-v1', 'loader_identity': loader_identity,
            'runtime': str(runtime), 'author': str(author), 'bridge': str(bridge), 'entropy': str(entropy),
            'i_model': str(i_model), 'p_model': str(p_model), 'wire': str(wire), 'output': str(output),
            'device_paths': device_paths(), 'denied_paths': denied, 'canary_dir': str(canary),
            'parent_user': os.readlink('/proc/self/ns/user'), 'parent_net': os.readlink('/proc/self/ns/net'),
            'visible_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'cpu_seconds': 180}


def make_canaries(path: Path) -> None:
    path.mkdir()
    for name in ('write', 'truncate', 'remove', 'rename', 'link'):
        (path / name).write_bytes(b'canary')


def check_canaries(path: Path) -> dict[str, str]:
    if ({x.name for x in path.iterdir()} != {'write', 'truncate', 'remove', 'rename', 'link'}
            or any(p.read_bytes() != b'canary' for p in path.iterdir())):
        raise ValueError('mutation changed a task-owned canary')
    return {p.name: sha(p) for p in sorted(path.iterdir())}


def receiver_command(python: Path, worker: Path, mode: str, config: Path, *, isolated: bool) -> list[str]:
    cmd = [str(python), str(worker), mode, '--config', str(config), '--config-sha256', sha(config)]
    return ['/usr/bin/unshare', '--user', '--map-root-user', '--net', *cmd] if isolated else cmd


def check_manifest(manifest: dict[str, Any], frames: int, stream: Path) -> None:
    expected = {'schema': 'gvcrt-native-v1', 'source_revision': 'd0e32bfa3e8e282f9a77437c223c605858eea637',
                'frames': frames, 'fps': 30, 'original_width': 1280, 'original_height': 720,
                'canvas_width': 1920, 'canvas_height': 1088, 'base_qp': 3, 'reset_interval': 8,
                'padding_policy': 'released', 'ec_part': 1,
                'model_sha256': {'I': I_SHA, 'P': P_SHA}}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('native manifest differs from fixed profile')
    if manifest.get('stream') != {'filename': 'stream.bin', 'bytes': stream.stat().st_size, 'sha256': sha(stream)}:
        raise ValueError('native stream identity mismatch')
    placements = manifest.get('placements', [])
    if len(placements) != frames:
        raise ValueError('native frame placement count mismatch')
    offset = 0
    shifts = [0, 2, 0, 1, 0, 1, 0, 1]
    for index, item in enumerate(placements):
        if (item.get('index') != index or item.get('I') != (index == 0)
                or item.get('reset') != (index > 0 and index % 8 == 1)
                or item.get('qp') != 3 + (shifts[index % 8] if index else 0)
                or item.get('offset') != offset or type(item.get('bytes')) is not int or item['bytes'] <= 0):
            raise ValueError('native frame schedule mismatch')
        offset += item['bytes']
    if offset != stream.stat().st_size:
        raise ValueError('native placements do not cover the stream')


def check_isolation(receipt: dict[str, Any], config: dict[str, Any]) -> None:
    expected = {p for paths in config['denied_paths'].values() for p in paths}
    denials = receipt.get('denials', {})
    namespaces = receipt.get('namespaces', {})
    if (not expected or set(denials) != expected or any(type(x) is not int or x not in (1, 13) for x in denials.values())
            or type(receipt.get('landlock_abi')) is not int or receipt['landlock_abi'] < 3
            or receipt.get('network_interfaces') != ['lo'] or receipt.get('pre_sandbox_threads') != 1
            or receipt.get('extra_fds') is not False or receipt.get('cuda_preinitialized') is not False
            or not receipt.get('cpu_before_sha256')
            or receipt.get('cpu_before_sha256') != receipt.get('cpu_after_sha256')
            or not namespaces.get('user') or not namespaces.get('net')
            or namespaces['user'] == config['parent_user'] or namespaces['net'] == config['parent_net']):
        raise ValueError('incomplete receiver isolation evidence')
    abi = receipt['landlock_abi']
    expected_mask = (1 << 15) - 1 | ((1 << 15) if abi >= 5 else 0)
    mutations = receipt.get('mutation_denials', {})
    if (receipt.get('handled_fs_rights') != expected_mask or receipt.get('output_write_probe') is not True
            or set(mutations) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
            or any(type(code) is not int or code not in ((1, 13, 18) if name in ('rename', 'link') else (1, 13))
                   for name, code in mutations.items())):
        raise ValueError('mutation confinement evidence missing')
    expected_canary = hashlib.sha256(b'canary').hexdigest()
    canaries = receipt.get('parent_canary_post_sha256', {})
    if (set(canaries) != {'write', 'truncate', 'remove', 'rename', 'link'}
            or any(value != expected_canary for value in canaries.values())):
        raise ValueError('independent mutation canary check failed')
    identities = receipt.get('protected_identities', {})
    if set(identities) != set(config['denied_paths']):
        raise ValueError('protected identity groups missing')
    for group, paths in config['denied_paths'].items():
        if [x.get('path') for x in identities[group]] != paths:
            raise ValueError('protected identity path mismatch')
    alias_ids = {(x['device'], x['inode']) for x in identities['aliases']}
    for group in set(identities) - {'aliases'}:
        if not all((x['device'], x['inode']) in alias_ids for x in identities[group]):
            raise ValueError('protected identity alias missing')
    root_names = ('runtime', 'author', 'bridge', 'entropy', 'i_model', 'p_model', 'wire', 'output')
    expected_roots = [str(Path(config[name]).resolve()) for name in root_names]
    expected_roots.extend(str(Path(p).resolve()) for p in config['device_paths'])
    leaves = receipt.get('approved_leaves', [])
    if config.get('policy_profile') != 'gvcrt-w8-twelve-leaf-mutation-v1' or receipt.get('policy_profile') != config['policy_profile']:
        raise ValueError('policy profile differs')
    if [row.get('name') for row in leaves] != list(APPROVED_LEAVES):
        raise ValueError('qualified system leaf list differs')
    pid = receipt.get('pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError('isolated worker pid missing')
    loader = config['loader_identity']
    if loader['uid'] != 0 or loader['mode'] & stat.S_IWOTH:
        raise ValueError('loader identity unsafe')
    for leaf in leaves:
        name, resolved = leaf['name'], leaf['resolved']
        expected_leaf = name.replace('/proc/self/', f'/proc/{pid}/')
        if (resolved != expected_leaf or not stat.S_ISREG(leaf['mode'])
                or any(type(leaf[k]) is not int or leaf[k] < 0 for k in ('device', 'inode', 'mode', 'uid'))):
            raise ValueError('approved leaf identity differs')
        if name == '/etc/ld.so.cache' and any(leaf[k] != loader[k] for k in ('device', 'inode', 'mode')):
            raise ValueError('loader identity differs')
        expected_roots.append(resolved)
    if receipt.get('allow_roots') != expected_roots:
        raise ValueError('receiver allowlist differs from frozen config')
    roots = [Path(p) for p in receipt.get('allow_roots', [])]
    if not roots or any(Path(x['resolved']) == root or Path(x['resolved']).is_relative_to(root)
                        for group in identities.values() for x in group for root in roots):
        raise ValueError('source identity overlaps granted root')


APPROVED_LEAVES = (
    '/etc/ld.so.cache', '/proc/cpuinfo', '/proc/devices',
    '/proc/driver/nvidia/capabilities/mig/config', '/proc/driver/nvidia/capabilities/mig/monitor',
    '/proc/driver/nvidia/params', '/proc/self/maps', '/proc/self/status', '/proc/stat',
    '/proc/sys/vm/mmap_min_addr', '/sys/devices/system/cpu/online',
    '/sys/devices/system/memory/block_size_bytes',
)

METRIC_INPUTS = {
    'balltrack_code': '5bec7ca535d2562ce0252a45f3dded0cfd5897571a6e29a10247a87884d961ff',
    'balltrack_weight': 'b052479e6229d840a1a7f42dbc83a6b752a53ef8bc5841a4b957acd519563fec',
    'wasb_code': '657fb5e0c06bbe621f10bdf6d8937f5cb7f8cbef1c4a369e30d9d87f54e87a55',
    'wasb_weights': '4129d4210145dd029e393028deef717a10b59a603f93e5081c1058b8f178f5e6',
    'wasb_config': '6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d',
    'lpips_backbone': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
    'dists_backbone': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0',
}


def baseline_rows(path: Path, count: int) -> list[dict[str, Any]]:
    if sha(path) != BASELINE_SHA:
        raise ValueError('baseline archive mismatch')
    with tarfile.open(path) as archive:
        entry = archive.extractfile('publish/pilot.json')
        if entry is None:
            raise ValueError('missing baseline report')
        raw = entry.read()
        if hashlib.sha256(raw).hexdigest() != 'c46864671036e7435a572b9c05793ad79639927897adce2f89bd0bb62d3fff12':
            raise ValueError('baseline report identity mismatch')
        pilot = json.loads(raw)
        if (pilot.get('frames') != 16 or pilot.get('clip') != 'tracknet/game10/Clip3'
                or pilot.get('conversion', {}).get('yuv_sha256') !=
                'c82d2e12e18e19f0453cb4d1b497581841c423149b153a1fc4ba2de6ff4353f4'):
            raise ValueError('baseline source contract mismatch')
        results = []
        for point in pilot['points']:
            codec, setting = point['codec'], point['point']
            directory = f'publish/{codec}-{setting}'
            base = point['baseline']
            for filename, digest in (('native.bin', base['native_sha256']),
                                     ('baseline.psbl', base['zero_ball_wire_sha256'])):
                data = archive.extractfile(f'{directory}/{filename}')
                if data is None or hashlib.sha256(data.read()).hexdigest() != digest:
                    raise ValueError('baseline physical stream mismatch')
            metrics = base['metrics']
            if len(metrics) != 16:
                raise ValueError('baseline metric frames missing')
            results.append({'codec': codec, 'point': setting, 'native_bytes': base['native_bytes'],
                            'zero_ball_wire_bytes': base['zero_ball_wire_bytes'],
                            'native_sha256': base['native_sha256'], 'wire_sha256': base['zero_ball_wire_sha256'],
                            'decoded_yuv_sha256': point['encoder_decoded_sha256'],
                            'metrics': metrics[:count],
                            'judges': {k: {'rows': v['rows'][:count]} for k, v in base['judges'].items()}})
    if {(x['codec'], x['point']) for x in results} != {('dcvc', 0), ('dcvc', 21), ('dcvc', 42),
                                                        ('svt', 48), ('svt', 55), ('svt', 62)}:
        raise ValueError('baseline point set differs')
    return results


def verify_source_members(archive_path: Path, receipt: dict[str, Any], labels: list[list[Any]]) -> None:
    members = receipt.get('source_members', [])
    if (receipt.get('tracknet_sha256') != DATA_SHA or receipt.get('frames') != 16
            or receipt.get('size') != [1280, 720] or receipt.get('fps') != 30
            or receipt.get('source_yuv_sha256') !=
                'c82d2e12e18e19f0453cb4d1b497581841c423149b153a1fc4ba2de6ff4353f4'
            or not isinstance(receipt.get('conversion_floor_psnr'), (int, float))
            or not math.isclose(receipt['conversion_floor_psnr'], 41.45716184079476, abs_tol=1e-9)
            or len(members) != 16 or len(labels) != 16):
        raise ValueError('source preparation identity or conversion floor mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        raw_label = archive.read('Dataset/game10/Clip3/Label.csv')
        if hashlib.sha256(raw_label).hexdigest() != LABEL_SHA:
            raise ValueError('source label identity mismatch')
        rows = [line.split(',') for line in raw_label.decode().splitlines()[1:] if line.split(',')[0]]
        if len(rows) != 188:
            raise ValueError('source label row count mismatch')
        for index, (row, member, label) in enumerate(zip(rows[:16], members, labels)):
            name = f'Dataset/game10/Clip3/{row[0]}'
            raw = archive.read(name)
            visibility = int(row[1]) if row[1] else 0
            expected_label = [1 if visibility in (1, 2) else 0 if visibility == 0 else -1,
                              float(row[2]) if row[2] else 0.0, float(row[3]) if row[3] else 0.0]
            if (member != {'path': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                    or label != expected_label):
                raise ValueError(f'ordered source member or label mismatch at {index}')


def quality_summary(rows: list[dict[str, Any]], labels: list[list[Any]]) -> dict[str, float]:
    if len(rows) != len(labels) or not rows:
        raise ValueError('quality row count mismatch')
    keys = ('frame_psnr', 'full_lpips', 'full_dists', 'crop_lpips', 'crop_dists')
    values: dict[str, list[float]] = {k: [] for k in keys}
    for index, (row, label) in enumerate(zip(rows, labels)):
        if row.get('frame') != index:
            raise ValueError('quality frame order mismatch')
        for key in keys:
            value = row.get(key)
            if value is None and key.startswith('crop_') and label[0] != 1:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'invalid metric {key}')
            values[key].append(float(value))
    if any(not v for v in values.values()):
        raise ValueError('quality metric has no valid frames')
    return {key: sum(v) / len(v) for key, v in values.items()}


def rate_context(peer_metrics: dict[str, float], baseline: list[dict[str, Any]],
                 peer_native_bytes: int, peer_wire_bytes: int, frames: int) -> dict[str, Any] | None:
    """Single-point nearest and achievable anchor-hull context, with no BD-rate."""
    if frames != 16:
        return None
    from experiments.background.g5 import anchor_quality, envelope
    result: dict[str, Any] = {}
    metrics = ('frame_psnr', 'full_lpips', 'full_dists', 'crop_lpips', 'crop_dists')
    for kind, peer_bytes, baseline_key in (('native', peer_native_bytes, 'native_bytes'),
                                           ('wire', peer_wire_bytes, 'zero_ball_wire_bytes')):
        peer_rate = peer_bytes * 8 * 30 / frames / 1000
        if peer_rate <= 0:
            raise ValueError('invalid peer rate')
        result[kind] = {}
        for metric in metrics:
            sign = 1 if metric == 'frame_psnr' else -1
            candidates = []
            for item in baseline:
                quality = item['mean_metrics'][metric]
                rate = item[baseline_key] * 8 * 30 / frames / 1000
                if rate <= 0 or not math.isfinite(quality):
                    raise ValueError('invalid anchor rate or quality')
                candidates.append((rate, sign * quality, item['codec'], item['point']))
            hull = envelope([(rate, quality) for rate, quality, _, _ in candidates])
            nearest = min(candidates, key=lambda x: abs(math.log(x[0] / peer_rate)))
            interpolated = None
            if hull and hull[0][0] <= peer_rate <= hull[-1][0]:
                value = anchor_quality([(r, q) for r, q, _, _ in candidates], peer_rate)
                interpolated = None if value is None else sign * value
            result[kind][metric] = {'peer_rate_kbps': peer_rate, 'peer_quality': peer_metrics[metric],
                'nearest': {'codec': nearest[2], 'point': nearest[3], 'rate_kbps': nearest[0],
                            'quality': sign * nearest[1]},
                'anchor_envelope_quality_at_peer_rate': interpolated,
                'anchor_envelope': [{'rate_kbps': r, 'quality': sign * q} for r, q in hull],
                'scope': 'single point; equal-rate quality context, not BD-rate'}
    return result


def paired_comparison(score: dict[str, Any], baseline: list[dict[str, Any]],
                      labels: list[list[Any]], frames: int, native_bytes: int, wire_bytes: int) -> dict[str, Any]:
    from experiments.tennis.k5d_analysis import score_frames

    def judges(value: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for name in ('wasb', 'balltrack'):
            rows = value[name]['rows']
            if len(rows) != frames:
                raise ValueError('judge row count mismatch')
            predictions = [(row['detected'], row['x'], row['y']) for row in rows]
            out[name] = score_frames([tuple(x) for x in labels], predictions)
        return out
    if len(labels) != frames:
        raise ValueError('label count mismatch')
    peer_judges = judges(score)
    if any(peer_judges[k] != score[k]['scores'] for k in peer_judges):
        raise ValueError('peer detector summary differs from saved frame predictions')
    peer_metrics = quality_summary(score['metrics'], labels)
    anchor_summary = [{**x, 'mean_metrics': quality_summary(x['metrics'], labels)} for x in baseline]
    anchors = [{'codec': x['codec'], 'point': x['point'],
                'native_bytes_16': x['native_bytes'], 'wire_bytes_16': x['zero_ball_wire_bytes'],
                'native_kbps': x['native_bytes'] * 8 * 30 / 16 / 1000 if frames == 16 else None,
                'wire_kbps': x['zero_ball_wire_bytes'] * 8 * 30 / 16 / 1000 if frames == 16 else None,
                'metrics': x['mean_metrics'], 'judges': judges(x['judges'])} for x in anchor_summary]
    return {'frames': frames, 'rate_scope': 'matched_16_frames' if frames == 16 else '4-frame smoke; baseline streams cover 16',
            'peer': {'metrics': peer_metrics, 'judges': peer_judges,
                     'native_kbps': native_bytes * 8 * 30 / frames / 1000 if frames == 16 else None,
                     'wire_kbps': wire_bytes * 8 * 30 / frames / 1000 if frames == 16 else None},
            'baselines': anchors, 'rate_quality_context': rate_context(peer_metrics, anchor_summary,
                                                                        native_bytes, wire_bytes, frames),
            'bd_rate': None}



def need_success(result: dict[str, Any]) -> None:
    if result['status'] != 0 or result['timed_out']:
        raise RuntimeError(f"{result['name']} failed; retained receipts")


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f'invalid JSON object: {path}')
    return result


def run(args: argparse.Namespace) -> int:
    stage_started = time.monotonic()
    signal.signal(signal.SIGTERM, terminate)
    if args.frames not in (4, 16) or os.environ.get('PS_STAGE') != ('smoke' if args.frames == 4 else 'full'):
        raise ValueError('registered frame count and phase mismatch')
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get('PS_CLAIMED_GPU_UUID') != os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('one CPU and claimed GPU UUID required')
    stage_root = Path(os.environ['PS_STAGE_DIR']).resolve(strict=True)
    dispatch = read_json(stage_root / 'dispatch.json')
    if dispatch.get('gpu_sharing', {}).get('mode') != 'exclusive':
        raise ValueError('exclusive fleet GPU required')
    packed = dispatch['staged_environment']
    if packed['sha256'] != RUNTIME_SHA or Path(packed['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('runtime dispatch identity mismatch')
    clock = dispatch.get('campaign_clock', {})
    if not isinstance(clock.get('budget_end'), (int, float)):
        raise ValueError('fleet campaign deadline missing')
    stage_budget = 200.0 if args.frames == 4 else 500.0
    def child(command: list[str], name: str, stage: Path, env: dict[str, str], timeout: float) -> dict[str, Any]:
        reserve = 35.0 if name == 'isolated-probe' else 5.0
        available = min(stage_started + stage_budget, clock['budget_end']) - time.monotonic() - reserve
        if available <= 0:
            raise RuntimeError(f'insufficient remaining stage budget for {name}')
        return run_child(command, name, stage, env, min(timeout, available))
    if args.frames == 16:
        smoke = read_json(stage_root.parent / 'smoke/gvcrt-peer/result.json')
        if (not smoke['projection']['passes'] or smoke['projection']['forecast_gpu_seconds'] + 30
                > clock['budget_end'] - time.monotonic()):
            raise ValueError('smoke cost gate or remaining campaign budget failed')
    known = {'source': SOURCE_SHA, 'historic': OLD_SOURCE_SHA, 'archive': DATA_SHA, 'baseline': BASELINE_SHA,
             'i_model': I_SHA, 'p_model': P_SHA, 'entropy': ENTROPY_SHA, **METRIC_INPUTS}
    inputs = {name: staged(dispatch, name, digest) for name, digest in known.items()}
    for name, expected in (('source', SOURCE_SHA), ('historic', OLD_SOURCE_SHA), ('baseline', BASELINE_SHA)):
        if sha(inputs[name]) != expected:
            raise ValueError(f'{name} bytes differ from staged identity')
    runtime = Path(packed['prefix']).resolve(strict=True)
    claimed = os.environ.get('PS_CLAIMED_GPU_UUID', '')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != claimed:
        raise ValueError('CUDA visibility differs from fleet claim')
    expected_uuid = device_guard.canonical_uuid(claimed)
    dispatched = (dispatch.get('environment', {}).get('gpu') or {}).get('uuid')
    if device_guard.canonical_uuid(dispatched) != expected_uuid:
        raise ValueError('fleet dispatch and claimed GPU differ')
    gpu_identity = inspect_gpu(claimed)
    ffmpeg = runtime / 'bin/ffmpeg'
    if sha(ffmpeg) != '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2':
        raise ValueError('calibration ffmpeg identity mismatch')
    ffmpeg_version = subprocess.run([str(ffmpeg), '-version'], capture_output=True, text=True,
                                    check=True, timeout=10).stdout.splitlines()[0]
    python = Path(packed['python']).resolve(strict=True)
    work = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True) / 'gvcrt-peer'
    work.mkdir()
    stage = stage_root / 'gvcrt-peer'
    stage.mkdir()
    author, historic, minimal = work / 'author', work / 'historic', work / 'minimal'
    unpack(inputs['source'], author)
    unpack(inputs['historic'], historic)
    marker = HERE / 'gvcrt_source_marker.json'
    if sha(marker) != MARKER_SHA or sha(HERE / 'gvcrt_native_bridge.py') != BRIDGE_SHA:
        raise ValueError('bridge or author marker identity mismatch')
    shutil.copyfile(marker, author / '.gvcrt-source.json')
    for item in author.rglob('*'):
        if not item.is_symlink():
            item.chmod(0o555 if item.is_dir() else 0o444)
    author.chmod(0o555)
    minimal.mkdir()
    for name in ('gvcrt_native_bridge.py', 'gvcrt_receiver_worker_w8leaves.py', 'gvcrt_peer_scorer.py',
                 'gvcrt_device_guard.py', 'gvcrt_guarded_scorer.py'):
        shutil.copyfile(HERE / name, minimal / name)
        (minimal / name).chmod(0o444)
    bridge, worker, scorer = (minimal / name for name in
                              ('gvcrt_native_bridge.py', 'gvcrt_receiver_worker_w8leaves.py', 'gvcrt_peer_scorer.py'))
    env = env_for(work, author, inputs['entropy'])
    historical_env = {**env, 'PYTHONPATH': str(historic), 'PS_SCRATCH_DIR': str(work)}
    prep = work / 'prep'
    prepare = [str(python), str(scorer), 'prepare', '--archive', str(inputs['archive']),
               '--clip', 'tracknet/game10/Clip3', '--output', str(prep), '--ffmpeg', str(runtime / 'bin/ffmpeg')]
    prep_result = child(prepare, 'prepare', stage, historical_env, 70)
    need_success(prep_result)
    prepare_receipt = read_json(prep / 'prepare_receipt.json')
    prepared_labels = read_json(prep / 'labels.json')['labels']
    verify_source_members(inputs['archive'], prepare_receipt, prepared_labels)
    frames = json.loads((prep / 'frames.json').read_text())
    if len(frames) != 16:
        raise ValueError('prepared 16-frame manifest missing')
    selected = work / 'frames.json'
    save(selected, frames[:args.frames])
    baseline = baseline_rows(inputs['baseline'], args.frames)
    placeholder = work / 'probe-reference-placeholder'
    placeholder.write_bytes(b'protected reference placeholder')
    original_archive = next(x['source'] for x in dispatch['staged_inputs'] if x['name'] == 'archive')
    protected = {'source_archive': list(dict.fromkeys([str(inputs['archive']), str(Path(original_archive).resolve(strict=True))])),
                 'source_rgb': [str(prep / 'source_rgb.npy'), str(prep / 'converted_rgb.npy'),
                                *(f['path'] for f in frames)],
                 'source_yuv': [str(prep / 'source.yuv')], 'sender_reference': [str(placeholder)],
                 'ground_truth': [str(prep / 'labels.json'), str(inputs['baseline']), str(inputs['historic']),
                                  str(historic / 'experiments/tennis/k5d_encoder.py')]}
    denied = aliases_for(protected, work / 'probe-aliases')
    fakewire = work / 'probe-wire'
    fakewire.mkdir()
    (fakewire / 'stream.bin').write_bytes(b'probe')
    (fakewire / 'manifest.json').write_text('{}')
    probeout = work / 'probe-output'
    probeout.mkdir()
    probe_canary = work / 'probe-canary'
    make_canaries(probe_canary)
    cfg = receiver_config(runtime, author, bridge, inputs['entropy'], inputs['i_model'], inputs['p_model'],
                          fakewire, probeout, denied, probe_canary)
    cfgpath = work / 'probe-config.json'
    save(cfgpath, cfg)
    shutil.copyfile(cfgpath, stage / 'probe-config.json')
    probe = child(receiver_command(python, worker, 'probe', cfgpath, isolated=True), 'isolated-probe', stage, env, 40)
    probe_error = None
    try:
        need_success(probe)
        isolation = read_json(probeout / 'isolation.json')
        isolation['parent_canary_post_sha256'] = check_canaries(probe_canary)
        check_isolation(isolation, cfg)
        if isolation['pid'] != probe['pid']:
            raise ValueError('isolated probe PID differs from worker receipt')
        probe_cuda = read_json(probeout / 'probe.json')
        device_guard.check_device(probe_cuda, gpu_identity['uuid'])
        if (probe_cuda.get('sum') != 1048576.0 or probe_cuda.get('name') != gpu_identity['name']
                or isolation.get('cuda_preinitialized') is not False):
            raise ValueError('invalid probe receipt')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        probe_error = {'type': type(exc).__name__, 'message': str(exc)}
    if probe_error is not None:
        for name in ('isolation.json', 'probe.json'):
            if (probeout / name).is_file():
                shutil.copyfile(probeout / name, stage / f'failed-probe-{name}')
        generic_out = work / 'generic-output'
        generic_out.mkdir()
        generic_canary = work / 'generic-canary'
        make_canaries(generic_canary)
        generic_cfg = {**cfg, 'output': str(generic_out), 'canary_dir': str(generic_canary)}
        generic_path = work / 'generic-config.json'
        save(generic_path, generic_cfg)
        shutil.copyfile(generic_path, stage / 'generic-config.json')
        child(receiver_command(python, worker, 'generic', generic_path, isolated=False), 'generic-probe', stage, env, 30)
        if (generic_out / 'generic.json').is_file():
            shutil.copyfile(generic_out / 'generic.json', stage / 'generic-probe.json')
        save(stage / 'probe-failure.json', probe_error)
        raise RuntimeError('isolated CUDA/file-denial probe failed; generic control retained; no model run')
    sender_dir = work / 'sender'
    sender_cmd = [str(python), str(bridge), 'encode', '--disable-fused', '--official-repo', str(author),
                  '--i-checkpoint', str(inputs['i_model']), '--i-sha256', I_SHA, '--p-checkpoint', str(inputs['p_model']),
                  '--p-sha256', P_SHA, '--frames-json', str(selected), '--frames', str(args.frames),
                  '--width', '1280', '--height', '720', '--fps', '30', '--qp', '3', '--reset-interval', '8',
                  '--padding-policy', 'released', '--output-dir', str(sender_dir)]
    sender = child(sender_cmd, 'sender', stage, env, 150 if args.frames == 4 else 350)
    need_success(sender)
    sender_receipt = read_json(sender_dir / 'sender_receipt.json')
    manifest = read_json(sender_dir / 'manifest.json')
    check_manifest(manifest, args.frames, sender_dir / 'stream.bin')
    env_record = sender_receipt['qualification']['environment']
    if (env_record.get('disable_fused') is not True or env_record.get('customized_cuda_inference') is not False
            or env_record.get('cuda_visible_devices') != os.environ['CUDA_VISIBLE_DEVICES']
            or env_record.get('device_name') != gpu_identity['name']
            or env_record.get('bridge_sha256') != BRIDGE_SHA or env_record.get('entropy_sha256') != ENTROPY_SHA
            or len(sender_receipt['qualification'].get('coverage', [])) != 2
            or not all(x.get('strict_complete_coverage') for x in sender_receipt['qualification']['coverage'])):
        raise ValueError('sender backend or GPU visibility mismatch')
    wire = work / 'receiver-wire'
    wire.mkdir()
    for name in ('stream.bin', 'manifest.json'):
        shutil.copyfile(sender_dir / name, wire / name)
    receiverout = work / 'receiver-output'
    receiverout.mkdir()
    protected['sender_reference'] = [str(sender_dir / name) for name in
                                     ('sender_reconstruction.npy', 'sender_receipt.json', 'stream.bin', 'manifest.json')]
    denied_decode = aliases_for(protected, work / 'receiver-aliases')
    receiver_canary = work / 'receiver-canary'
    make_canaries(receiver_canary)
    rcfg = receiver_config(runtime, author, bridge, inputs['entropy'], inputs['i_model'], inputs['p_model'],
                           wire, receiverout, denied_decode, receiver_canary)
    rcfg_path = work / 'receiver-config.json'
    save(rcfg_path, rcfg)
    receiver = child(receiver_command(python, worker, 'decode', rcfg_path, isolated=True),
                         'receiver', stage, env, 150 if args.frames == 4 else 350)
    need_success(receiver)
    receiver_receipt = read_json(receiverout / 'decoded/receiver_receipt.json')
    receiver_isolation = read_json(receiverout / 'isolation.json')
    renv = receiver_receipt['qualification']['environment']
    if (renv.get('disable_fused') is not True or renv.get('customized_cuda_inference') is not False
            or renv.get('entropy_sha256') != ENTROPY_SHA or renv.get('bridge_sha256') != BRIDGE_SHA
            or renv.get('cuda_visible_devices') != os.environ['CUDA_VISIBLE_DEVICES']
            or renv.get('device_name') != gpu_identity['name']):
        raise ValueError('receiver backend identity mismatch')
    receiver_isolation['parent_canary_post_sha256'] = check_canaries(receiver_canary)
    check_isolation(receiver_isolation, rcfg)
    if receiver_isolation['pid'] != receiver['pid']:
        raise ValueError('isolated receiver PID differs from worker receipt')
    received = receiverout / 'decoded/reconstruction.npy'
    reference = sender_dir / 'sender_reconstruction.npy'
    if sha(received) != sha(reference) or receiver_receipt['reconstruction_sha256'] != sha(received):
        raise ValueError('fresh decoder pixels differ from sender reference')
    if args.frames == 16:
        import numpy as np
        smoke_pixels = np.load(stage_root.parent / 'smoke/gvcrt-peer/reconstruction.npy', allow_pickle=False, mmap_mode='r')
        full_pixels = np.load(received, allow_pickle=False, mmap_mode='r')
        if not np.array_equal(smoke_pixels, full_pixels[:4]):
            raise ValueError('full first-four reconstructions differ from smoke')
    metric_tree_hashes = verify_metric_sources(dispatch, inputs)
    config = work / 'scorer-config.json'
    save(config, {name: str(inputs[name]) for name in METRIC_INPUTS if name != 'wasb_config'})
    score_path = work / 'score.json'
    score_cmd = [str(python), str(minimal / 'gvcrt_guarded_scorer.py'), '--expected-uuid', claimed, '--source', str(prep / 'source_rgb.npy'),
                 '--decoded', str(received), '--labels', str(prep / 'labels.json'), '--config', str(config),
                 '--count', str(args.frames), '--output', str(score_path)]
    score = child(score_cmd, 'score', stage, historical_env, 150 if args.frames == 4 else 350)
    need_success(score)
    scores = read_json(score_path)
    device_guard.check_device(scores.get('device', {}), gpu_identity['uuid'], scorer=True)
    if scores['device']['name'] != gpu_identity['name']:
        raise ValueError('scorer GPU identity mismatch')
    metrics_record = scores.get('metric_models', {})
    if (metrics_record.get('lpips', {}).get('heads_sha256') != 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
            or metrics_record.get('lpips', {}).get('backbone_sha256') != METRIC_INPUTS['lpips_backbone']
            or metrics_record.get('dists', {}).get('weights_sha256') != 'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218'
            or metrics_record.get('dists', {}).get('backbone_sha256') != METRIC_INPUTS['dists_backbone']):
        raise ValueError('scorer metric model differs from retained calibration')
    if scores.get('frames') != args.frames or len(scores.get('metrics', [])) != args.frames:
        raise ValueError('missing per-frame scores')
    if any(len(scores[name]['rows']) != args.frames for name in ('wasb', 'balltrack')):
        raise ValueError('missing per-frame detector scores')
    label_data = read_json(prep / 'labels.json')['labels'][:args.frames]
    native_bytes = (wire / 'stream.bin').stat().st_size
    wire_bytes = native_bytes + (wire / 'manifest.json').stat().st_size
    comparison = paired_comparison(scores, baseline, label_data, args.frames, native_bytes, wire_bytes)
    output_copy_start = time.monotonic()
    for source, name in ((rcfg_path, 'receiver-config.json'),
                         (wire / 'stream.bin', 'stream.bin'), (wire / 'manifest.json', 'manifest.json'),
                         (received, 'reconstruction.npy'), (reference, 'sender_reference.npy'),
                         (prep / 'prepare_receipt.json', 'prepare_receipt.json'), (prep / 'labels.json', 'labels.json'),
                         (config, 'scorer-config.json'), (score_path, 'score.json'),
                         (sender_dir / 'sender_receipt.json', 'sender_receipt.json'),
                         (receiverout / 'decoded/receiver_receipt.json', 'receiver_receipt.json')):
        shutil.copyfile(source, stage / name)
    artifact_names = ('stream.bin', 'manifest.json', 'reconstruction.npy', 'sender_reference.npy',
                      'prepare_receipt.json', 'labels.json', 'scorer-config.json', 'score.json',
                      'sender_receipt.json', 'receiver_receipt.json', 'probe-config.json', 'receiver-config.json')
    artifact_hashes = {name: sha(stage / name) for name in artifact_names}
    output_copy_seconds = time.monotonic() - output_copy_start
    times = {'staging': max(0.0, clock['phase_started'] - clock['started']), 'output_copy': output_copy_seconds, 'prep': prep_result['wall_seconds'], 'probe': probe['wall_seconds'],
             'sender': sender['wall_seconds'], 'receiver': receiver['wall_seconds'], 'score': score['wall_seconds'],
             'validation_reserve': 30.0, 'release_reserve': 30.0}
    accounted = sum(times[k] for k in ('prep', 'probe', 'sender', 'receiver', 'score', 'output_copy'))
    unmeasured = max(0.0, time.monotonic() - stage_started - accounted)
    times['staging'] += unmeasured
    remaining = min(500.0, clock['budget_end'] - time.monotonic() - 60.0)
    cost = projection(sender_receipt, receiver_receipt, times, remaining) if args.frames == 4 else None
    result = {'frames': args.frames, 'clip': 'tracknet/game10/Clip3', 'qp': 3, 'comparison': comparison,
              'source_yuv_sha256': 'c82d2e12e18e19f0453cb4d1b497581841c423149b153a1fc4ba2de6ff4353f4',
              'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'gpu_identity': gpu_identity, 'probe_cuda': probe_cuda,
              'ffmpeg': {'sha256': sha(ffmpeg), 'version': ffmpeg_version},
              'metric_input_sha256': METRIC_INPUTS, 'metric_source_tree_sha256': metric_tree_hashes,
              'probe_isolation': isolation, 'receiver_isolation': receiver_isolation,
              'source_sha256': SOURCE_SHA, 'runtime_sha256': RUNTIME_SHA, 'entropy_sha256': ENTROPY_SHA,
              'i_model_sha256': I_SHA, 'p_model_sha256': P_SHA, 'cold_model_bytes': 541107049,
              'canonical_models': ['/home/itec/emanuele/Models/GVC-RT/GVC_RT-I.pt',
                                   '/home/itec/emanuele/Models/GVC-RT/GVC_RT-P.pt'],
              'baseline_archive_sha256': BASELINE_SHA, 'baseline': baseline,
              'stream_bytes': (stage / 'stream.bin').stat().st_size,
              'manifest_bytes': (stage / 'manifest.json').stat().st_size,
              'native_kbps': (stage / 'stream.bin').stat().st_size * 8 * 30 / args.frames / 1000,
              'wire_kbps': ((stage / 'stream.bin').stat().st_size + (stage / 'manifest.json').stat().st_size)
                           * 8 * 30 / args.frames / 1000,
              'snapshot': dispatch['code'], 'worker_sha256': sha(worker), 'scorer_sha256': sha(scorer),
              'device_guard_sha256': sha(minimal / 'gvcrt_device_guard.py'),
              'guarded_scorer_sha256': sha(minimal / 'gvcrt_guarded_scorer.py'),
              'files': artifact_hashes,
              'times': times, 'projection': cost, 'training_overlap': 'unknown',
              'limitations': 'one exploratory peer point; no BD-rate or full-clip quality claim'}
    save(stage / 'result.json', result)
    if args.frames == 4 and (cost is None or not cost['passes']):
        raise RuntimeError('16-frame cost gate failed')
    return 0


def validate_phase(stage: Path, validation: Path) -> int:
    root = stage / 'gvcrt-peer'
    result = read_json(root / 'result.json')
    frames = result.get('frames')
    if type(frames) is not int or frames not in (4, 16):
        raise ValueError('invalid saved frame count')
    checks = {'frames_and_qp': frames in (4, 16) and result.get('qp') == 3,
              'fixed_sources': result.get('source_sha256') == SOURCE_SHA and result.get('runtime_sha256') == RUNTIME_SHA
                  and result.get('entropy_sha256') == ENTROPY_SHA and result.get('baseline_archive_sha256') == BASELINE_SHA,
              'input_checkpoint_identity': result.get('i_model_sha256') == I_SHA and result.get('p_model_sha256') == P_SHA,
              'ffmpeg_identity': result.get('ffmpeg', {}).get('sha256') ==
                  '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2',
              'artifacts': all((root / name).is_file() and sha(root / name) == digest
                               for name, digest in result.get('files', {}).items())
                  and set(result.get('files', {})) == {'stream.bin', 'manifest.json', 'reconstruction.npy',
                      'sender_reference.npy', 'prepare_receipt.json', 'labels.json', 'scorer-config.json', 'score.json',
                      'sender_receipt.json', 'receiver_receipt.json', 'probe-config.json', 'receiver-config.json'},
              'actual_bytes': (root / 'stream.bin').stat().st_size == result.get('stream_bytes')
                  and (root / 'manifest.json').stat().st_size == result.get('manifest_bytes'),
              'probe': result.get('probe_cuda', {}).get('sum') == 1048576.0
                  and result.get('probe_isolation', {}).get('cuda_preinitialized') is False,
              'receiver_denials': bool(result.get('receiver_isolation', {}).get('denials')),
              'baseline': len(result.get('baseline', [])) == 6,
              'metric_source_identity': result.get('metric_input_sha256') == METRIC_INPUTS
                  and set(result.get('metric_source_tree_sha256', {})) == {'balltrack_code', 'wasb_code'},
              'gate': frames != 4 or bool(result.get('projection', {}).get('passes'))}
    try:
        check_manifest(read_json(root / 'manifest.json'), frames, root / 'stream.bin')
        check_isolation(result['probe_isolation'], read_json(root / 'probe-config.json'))
        receiver_config_doc = read_json(root / 'receiver-config.json')
        probe_config_doc = read_json(root / 'probe-config.json')
        check_isolation(result['receiver_isolation'], receiver_config_doc)
        if (receiver_config_doc['loader_identity'] != probe_config_doc['loader_identity']
                or receiver_config_doc['policy_profile'] != probe_config_doc['policy_profile']):
            raise ValueError('probe and receiver policy identity differs')
        for name in ('source_archive', 'source_rgb', 'source_yuv', 'ground_truth'):
            if receiver_config_doc['denied_paths'][name] != probe_config_doc['denied_paths'][name]:
                raise ValueError('probe and receiver protected inputs differ')
        checks['native_schedule_and_isolation'] = True
    except (ValueError, KeyError, TypeError):
        checks['native_schedule_and_isolation'] = False
    checks['current_code'] = (result.get('worker_sha256') == sha(HERE / 'gvcrt_receiver_worker_w8leaves.py')
                              and result.get('scorer_sha256') == sha(HERE / 'gvcrt_peer_scorer.py')
                              and result.get('device_guard_sha256') == sha(HERE / 'gvcrt_device_guard.py')
                              and result.get('guarded_scorer_sha256') == sha(HERE / 'gvcrt_guarded_scorer.py')
                              and sha(HERE / 'gvcrt_native_bridge.py') == BRIDGE_SHA)
    try:
        names = ('prepare', 'isolated-probe', 'sender', 'receiver', 'score')
        receipts = [read_json(root / f'{name}.exit.json') for name in names]
        for name, rec in zip(names, receipts):
            if (rec.get('name') != name or rec.get('status') != 0 or rec.get('timed_out')
                    or rec.get('interrupted') or rec.get('started_monotonic', -1) < 0
                    or rec.get('ended_monotonic', -1) < rec['started_monotonic']
                    or sha(root / f'{name}.stdout') != rec.get('stdout_sha256')
                    or sha(root / f'{name}.stderr') != rec.get('stderr_sha256')):
                raise ValueError('child exit or log receipt mismatch')
        if any(a['ended_monotonic'] > b['started_monotonic'] for a, b in zip(receipts, receipts[1:])):
            raise ValueError('probe/sender/receiver/scorer ordering mismatch')
        if (result['probe_isolation']['pid'] != receipts[1]['pid']
                or result['receiver_isolation']['pid'] != receipts[3]['pid']):
            raise ValueError('isolated receipt PID differs from child')
        for rec, config_name in ((receipts[1], 'probe-config.json'), (receipts[3], 'receiver-config.json')):
            argv = rec['argv']
            if '--config-sha256' not in argv or argv[argv.index('--config-sha256') + 1] != sha(root / config_name):
                raise ValueError('child config identity mismatch')
        checks['child_receipts_and_probe_order'] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        checks['child_receipts_and_probe_order'] = False
    score = read_json(root / 'score.json')
    checks['score_rows'] = score.get('frames') == frames and len(score.get('metrics', [])) == frames and all(
        len(score.get(name, {}).get('rows', [])) == frames for name in ('wasb', 'balltrack'))
    try:
        labels = read_json(root / 'labels.json')['labels'][:frames]
        checks['paired_comparison'] = paired_comparison(score, result['baseline'], labels, frames,
            result['stream_bytes'], result['stream_bytes'] + result['manifest_bytes']) == result['comparison']
    except (ValueError, KeyError, TypeError):
        checks['paired_comparison'] = False
    receiver = read_json(root / 'receiver_receipt.json')
    sender = read_json(root / 'sender_receipt.json')
    def expected_backend(x: dict[str, Any]) -> bool:
        return bool(x.get('disable_fused') is True and x.get('customized_cuda_inference') is False
                    and x.get('entropy_sha256') == ENTROPY_SHA and x.get('bridge_sha256') == BRIDGE_SHA
                    and x.get('cuda_visible_devices') == result.get('gpu_uuid')
                    and x.get('device_name') == result['gpu_identity']['name'])
    checks['backend'] = all(expected_backend(x['qualification']['environment']) for x in (sender, receiver))
    checks['model_coverage'] = all(len(x['qualification']['coverage']) == 2 and all(
        item.get('strict_complete_coverage') for item in x['qualification']['coverage']) for x in (sender, receiver))
    metric_record = score.get('metric_models', {})
    checks['scorer_models'] = (metric_record.get('lpips', {}).get('heads_sha256') ==
        'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
        and metric_record.get('lpips', {}).get('backbone_sha256') == METRIC_INPUTS['lpips_backbone']
        and metric_record.get('dists', {}).get('weights_sha256') ==
        'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218'
        and metric_record.get('dists', {}).get('backbone_sha256') == METRIC_INPUTS['dists_backbone'])
    try:
        device_guard.check_device(result['probe_cuda'], result['gpu_uuid'])
        device_guard.check_device(score['device'], result['gpu_uuid'], scorer=True)
        checks['scorer_device'] = (score['device']['name'] == result['gpu_identity']['name']
                                   and result['probe_cuda']['name'] == result['gpu_identity']['name'])
    except (ValueError, TypeError, KeyError):
        checks['scorer_device'] = False
    if frames == 4:
        try:
            recomputed = projection(sender, receiver, result['times'], result['projection']['remaining_seconds'])
            checks['cost_recomputed'] = recomputed == result['projection'] and recomputed['passes']
        except (ValueError, KeyError, TypeError):
            checks['cost_recomputed'] = False
    checks['parity'] = (sha(root / 'reconstruction.npy') == receiver.get('reconstruction_sha256')
                        and sha(root / 'sender_reference.npy') == sha(root / 'reconstruction.npy'))
    prep = read_json(root / 'prepare_receipt.json')
    checks['input_transform'] = (prep.get('source_yuv_sha256') == result.get('source_yuv_sha256')
                                 and score.get('inputs_sha256', {}).get('source') == prep.get('source_rgb_sha256')
                                 and score.get('inputs_sha256', {}).get('labels') == prep.get('labels_sha256')
                                 and score.get('inputs_sha256', {}).get('decoded') == sha(root / 'reconstruction.npy')
                                 and score.get('inputs_sha256', {}).get('config') == sha(root / 'scorer-config.json'))
    checks['native_timing'] = (len(sender.get('encode_seconds_per_frame', [])) == frames
        and len(sender.get('encode_full_seconds_per_frame', [])) == frames
        and len(sender.get('reference_decode_full_seconds_per_frame', [])) == frames
        and len(receiver.get('decode_seconds_per_frame', [])) == frames
        and len(receiver.get('decode_full_seconds_per_frame', [])) == frames
        and sender.get('encode_full_timing_scope') == 'source_hash_read_rgb_pad_transfer_compress_packet_per_frame'
        and sender.get('decode_full_timing_scope') == 'packet_parse_decode_crop_cpu_transfer_per_frame'
        and receiver.get('decode_full_timing_scope') == 'packet_parse_decode_crop_cpu_transfer_per_frame')
    checks['source_denied'] = receiver.get('source_open_attempts_during_decode') == []
    if frames == 16:
        try:
            import numpy as np
            smoke_root = stage.parent / 'smoke/gvcrt-peer'
            smoke = read_json(smoke_root / 'result.json')
            smoke_sender = read_json(smoke_root / 'sender_receipt.json')
            smoke_receiver = read_json(smoke_root / 'receiver_receipt.json')
            expected = projection(smoke_sender, smoke_receiver, smoke['times'], smoke['projection']['remaining_seconds'])
            a = np.load(smoke_root / 'reconstruction.npy', allow_pickle=False, mmap_mode='r')
            b = np.load(root / 'reconstruction.npy', allow_pickle=False, mmap_mode='r')
            checks['smoke_promotion_and_prefix'] = (expected == smoke['projection'] and bool(expected['passes'])
                and sha(smoke_root / 'reconstruction.npy') == smoke['files']['reconstruction.npy']
                and np.array_equal(a, b[:4]))
        except (OSError, ValueError, KeyError, TypeError):
            checks['smoke_promotion_and_prefix'] = False
    dispatch = read_json(stage / 'dispatch.json')
    try:
        cfg = read_json(root / 'receiver-config.json')
        denied_paths = cfg['denied_paths']
        archive_entry = next(x for x in dispatch['staged_inputs'] if x['name'] == 'archive')
        expected_archives = {str(Path(archive_entry['path']).resolve()), str(Path(archive_entry['source']).resolve())}
        source_rgb = denied_paths['source_rgb']
        references = denied_paths['sender_reference']
        ground = denied_paths['ground_truth']
        checks['protected_input_coverage'] = (set(denied_paths['source_archive']) == expected_archives
            and len(source_rgb) == 18 and {Path(p).name for p in source_rgb[:2]} == {'source_rgb.npy', 'converted_rgb.npy'}
            and len(denied_paths['source_yuv']) == 1 and Path(denied_paths['source_yuv'][0]).name == 'source.yuv'
            and len(references) == 4 and {Path(p).name for p in references} ==
                {'sender_reconstruction.npy', 'sender_receipt.json', 'stream.bin', 'manifest.json'}
            and len(ground) == 4 and 'labels.json' in {Path(p).name for p in ground}
            and 'k5d_encoder.py' in {Path(p).name for p in ground}
            and len(denied_paths['aliases']) == sum(len(v) for k, v in denied_paths.items() if k != 'aliases'))
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        checks['protected_input_coverage'] = False
    try:
        archive_input = staged(dispatch, 'archive', DATA_SHA)
        verify_source_members(archive_input, prep, read_json(root / 'labels.json')['labels'])
        checks['source_members_and_floor'] = True
        baseline = staged(dispatch, 'baseline', BASELINE_SHA)
        checks['baseline_recomputed'] = baseline_rows(baseline, frames) == result['baseline']
        actual_inputs = {name: staged(dispatch, name, digest) for name, digest in METRIC_INPUTS.items()}
        checks['metric_sources_recomputed'] = verify_metric_sources(dispatch, actual_inputs) == result['metric_source_tree_sha256']
        expected_score_config = {name: str(actual_inputs[name]) for name in METRIC_INPUTS if name != 'wasb_config'}
        checks['scorer_config'] = read_json(root / 'scorer-config.json') == expected_score_config
        checks['metric_checkpoint_bytes'] = all(sha(actual_inputs[name]) == METRIC_INPUTS[name]
            for name in ('balltrack_weight', 'wasb_config', 'lpips_backbone', 'dists_backbone'))
        checks['checkpoint_bytes'] = (sha(staged(dispatch, 'i_model', I_SHA)) == I_SHA
                                      and sha(staged(dispatch, 'p_model', P_SHA)) == P_SHA
                                      and sha(staged(dispatch, 'entropy', ENTROPY_SHA)) == ENTROPY_SHA)
    except (OSError, ValueError, KeyError, TypeError):
        checks['baseline_recomputed'] = False
    passed = all(checks.values())
    save(validation, {'passed': passed, 'checks': checks})
    return 0 if passed else 1


def make_spec(deadline: str) -> dict[str, Any]:
    base = HERE.parents[1] / 'docs/experiments/ego-peer-inputs/20261104/gvcrt-native16-fleet-draft.json'
    spec = read_json(base)
    spec['entrypoint'] = ['-m', 'experiments.generative.gvcrt_peer_job_w8leaves']
    spec['validator'] = ['{python}', '-m', 'experiments.generative.gvcrt_peer_job_w8leaves', 'validate']
    spec['hosts'] = ['gpu5', 'gpu6']
    spec['gpu_models'] = ['RTX 6000 Ada']
    spec['deadline'] = deadline
    return spec


def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('run', 'validate', 'audit', 'make-spec'))
    p.add_argument('--stage', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--frames', type=int, choices=(4, 16))
    p.add_argument('--deadline')
    args = p.parse_args()
    if args.mode == 'make-spec':
        if args.output is None or args.deadline is None:
            p.error('make-spec needs --deadline and --output')
        spec = make_spec(args.deadline)
        from experiments.jobs.inbox import validate_spec
        validate_spec(spec)
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return 0
    if args.mode == 'validate':
        return validate_phase(Path(os.environ['PS_STAGE_DIR']), Path(os.environ['PS_VALIDATION_PATH']))
    if args.mode == 'audit':
        if args.stage is None or args.output is None:
            p.error('audit needs --stage and --output')
        return validate_phase(args.stage, args.output)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
