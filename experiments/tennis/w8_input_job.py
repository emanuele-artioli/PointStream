"""CPU qualification of the fixed W8 packet against its original inputs.

The fixed worker receives the paid bundle, independent profile and installed
generic guider. Only the separate validator reads fitted or golden inputs.
Linux network namespaces and Landlock are required; local tests do not prove
that the original runtime or a receiver run passes these gates.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import subprocess
import sys
import time
from typing import Any

# Legacy diagnostic process limits are retained below. The fixed path requires
# a fleet-wide one-core campaign of at most 300 seconds in fixed_inputs().
if len(sys.argv) > 1 and sys.argv[1] in {"run", "worker", "validate"}:
    _cpu = {"run": 120, "worker": 60, "validate": 120}[sys.argv[1]]
    resource.setrlimit(resource.RLIMIT_CPU, (_cpu, 650))

# Set before importing NumPy/OpenCV, including in the fleet validator process.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np  # noqa: E402

from experiments.tennis.w8_inputs import archive_members, decode_inputs, validate_golden  # noqa: E402

WINDOW = "nadal_kyrgios/000@120"
UNITS = tuple(f"nadal_kyrgios__000_at120__person_{i:012d}" for i in range(2))
PREFIX = "publish/packets/nadal_kyrgios__000@120/"
RECIPE = "0:240:3"
PREPARED_SHA = "94a5d6318fc113557fb24c8a32f2edffdf368f7a289484fddd4b19a749302bfa"
PACKET_SHAS = {
    "keypoints.kps": "2e8452e4c458cf0998b6b8e004812575bdb2a638dd927b087dbc330d11d67070",
    f"{UNITS[0]}.png.ref": "63d214f5631fd5557d86a9c8c7c2936e2ef45afe539e882001d60e6115896559",
    f"{UNITS[1]}.png.ref": "c8a4d5355f06d72281751a0d936d5e28c1d80d189021b20000ccfc1dc62e7d6a",
}
WINDOW_SHA = 'eff1aca84a63e870a2994bd227b3dacb5583fee5c54a420235d828fa1a31bbf4'
FIT_ARCHIVE_SHA = 'cceb49b3d593c770e5b0d8dcd5d69b5b358bf88d76030cef8034674417907584'
GUIDER_SHA = '1a8b7c1b4db92980fd977b4fd003c1396bbae9a9cdea00c35d452136d5e4f488'
PROFILE_SHA = 'bec98d67280e9f777a0d3cc110baa51ae2bf2c876aefe6b247d9691432f61107'
FIT_MEMBER_SHA = '525830aa39c14f3549084a422cda6cfc61cf1ad76b07b531dd899034e6f57570'
FIXED_INPUTS = {
    'packet': WINDOW_SHA, 'profile': PROFILE_SHA, 'prepared': PREPARED_SHA,
    'fits': FIT_ARCHIVE_SHA, 'guider': GUIDER_SHA,
    'base': '075b001a3ae9dcc9fdda9fb81e4240aaf3a9da228d707ac58733db094d983285',
    'source': 'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d',
    'masks': '4eb91e79c287a49accc17fcd115819e7345bc078d49e1477ce43ccdf87cd549b',
    'golden_generated': 'cce89567a57bd202db8dbd6a39599688bce8a52e048d53f2b38fd7fda7da4985',
}
PREP_ENV_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def judged_bitmap(recipe: str) -> bytes:
    if recipe != RECIPE:
        raise ValueError("Unexpected paid judged-frame recipe")
    ticks = np.zeros(240, dtype=np.uint8)
    ticks[0:240:3] = 1
    return np.packbits(ticks).tobytes()


def decode_frozen_window(blob: bytes, profile: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], bytes]:
    """Decode the fixed paid envelope; caller independently pins the profile.

    This function does not establish isolation or validate against goldens.
    """
    from experiments.tennis.w8_payload import unpack_window

    if digest(blob) != WINDOW_SHA:
        raise ValueError('fixed window payload identity mismatch')
    code_root = Path(__file__).resolve().parent
    for key, filename in (('payload_decoder_code', 'w8_payload.py'), ('conditioning_decoder_code', 'w8_inputs.py'),
                          ('geometry_code', 'k6b.py'), ('packet_code', 'k6b_rate_streams.py')):
        if file_sha(code_root / filename) != profile['models'][key]:
            raise ValueError(f'payload-declared decoder code mismatch: {key}')
    sections, _ = unpack_window(blob, profile)
    if profile['window'] != WINDOW or sections[profile['judged']] != judged_bitmap(RECIPE):
        raise ValueError('window or paid judged bitmap mismatch')
    units = profile['units']
    if {unit['id'] for unit in units} != set(UNITS):
        raise ValueError('payload unit identity mismatch')
    references = {unit['id']: sections[unit['reference']] for unit in units}
    arrays, bindings = decode_inputs(sections[profile['kps']], references, WINDOW, sections[profile['judged']])
    for unit in units:
        name = unit['id']
        if (not np.array_equal(arrays[name]['t'], np.arange(240))
                or not np.array_equal(np.flatnonzero(arrays[name]['judged']), np.arange(0, 240, 3))
                or bindings['reference_frames'][name] != unit['reference_tick']
                or not np.array_equal(arrays[name]['reference_box'], np.asarray(unit['box']))):
            raise ValueError(f'payload time, reference or scored-frame binding mismatch: {name}')
    return arrays, bindings, sections[profile['match']['section']]


def tensor_fingerprints(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fingerprint names, shape, dtype and exact contiguous tensor bytes."""
    result = {}
    for part in ('reference_unet', 'denoising_unet', 'pose_guider'):
        tensors = state[part]
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError(f'missing tensor group: {part}')
        for name, tensor in sorted(tensors.items()):
            if not isinstance(name, str) or not name:
                raise ValueError('invalid tensor name')
            value = tensor.detach().cpu().contiguous().numpy() if hasattr(tensor, 'detach') else np.asarray(tensor)
            if value.dtype.hasobject:
                raise ValueError('object tensor unsupported')
            result[f'{part}.{name}'] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                                       'sha256': digest(value.tobytes(order='C'))}
    return result


def check_worker_match(worker: dict[str, Any], expected: dict[str, Any]) -> None:
    fingerprints = worker.get('match_tensors')
    if not isinstance(fingerprints, dict) or len(fingerprints) != 480 or fingerprints != expected.get('tensor_fingerprints'):
        raise ValueError('isolated worker match tensor fingerprints differ')


def check_installed_guider(guider: Path, profile: dict[str, Any]) -> None:
    if file_sha(guider) != GUIDER_SHA or profile['models']['pose_guider'] != GUIDER_SHA:
        raise ValueError('installed generic guider identity mismatch')


def verify_match_payload(match: bytes, profile: dict[str, Any], fits: Path, guider: Path) -> dict[str, Any]:
    """Independently reproduce match wire symbols and quantized tensors from pinned originals."""
    import io
    import tarfile

    if digest(fits.read_bytes()) != FIT_ARCHIVE_SHA or digest(guider.read_bytes()) != GUIDER_SHA:
        raise ValueError('original checkpoint or guider identity mismatch')
    if profile['models']['pose_guider'] != GUIDER_SHA:
        raise ValueError('declared guider identity mismatch')
    import torch

    from experiments.tennis.k6b_fit import quantize_adapters
    from experiments.tennis.w8_payload import compare_checkpoint, pack_match, unpack_match
    with tarfile.open(fits) as archive:
        members = [m for m in archive if m.name == 'publish/nk-C@fit-100-fp16.pt']
        if len(members) != 1 or not members[0].isfile() or not 0 < members[0].size <= 100_000_000:
            raise ValueError('missing or ambiguous original checkpoint')
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError('unreadable original checkpoint')
        raw_checkpoint = stream.read()
        if len(raw_checkpoint) != members[0].size:
            raise ValueError('truncated original checkpoint')
    if digest(raw_checkpoint) != FIT_MEMBER_SHA:
        raise ValueError('original checkpoint member identity mismatch')
    torch.set_num_threads(1)
    state = torch.load(io.BytesIO(raw_checkpoint), map_location='cpu', weights_only=True)
    base = torch.load(guider, map_location='cpu', weights_only=True)
    expected, _ = quantize_adapters(state, 4, base)
    regenerated = pack_match(state, base, GUIDER_SHA)
    if regenerated != match or digest(match) != profile['match']['sha256']:
        raise ValueError('match wire symbols, scales or metadata differ')
    decoded = unpack_match(match, profile['match']['sha256'], GUIDER_SHA, base)
    counts = compare_checkpoint(decoded, expected)
    if counts != {'reference_unet': 208, 'denoising_unet': 256, 'pose_guider': 16}:
        raise ValueError('unexpected checkpoint tensor counts')
    return {'tensor_counts_exact': counts, 'tensor_fingerprints': tensor_fingerprints(expected),
            'checkpoint_sha256': digest(raw_checkpoint), 'guider_sha256': GUIDER_SHA,
            'match_sha256': digest(match)}


def verify_base_archive(path: Path, profile: dict[str, Any]) -> None:
    import tarfile

    expected = '85e3c65c9777b799f9ed0df3a5c8da567fe8a92e4ac83d115719527acb739eb4'
    if profile['models']['tennis_base'] != expected:
        raise ValueError('declared tennis base mismatch')
    with tarfile.open(path) as archive:
        members = [item for item in archive if item.name == 'publish/tennis-base@fit-100.pt']
        if len(members) != 1 or not members[0].isfile() or not 0 < members[0].size <= 2 * 1024**3:
            raise ValueError('missing or ambiguous tennis base')
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError('unreadable tennis base')
        checksum = hashlib.sha256()
        count = 0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            count += len(block)
            checksum.update(block)
        if count != members[0].size or checksum.hexdigest() != expected:
            raise ValueError('tennis base member identity mismatch')


def cpu_limit(seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, 650))


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def pinned_profile(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if digest(raw) != PROFILE_SHA:
        raise ValueError('independent profile identity mismatch')
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError('invalid independent profile')
    return result


def synthetic_torch_bootstrap(torch: Any) -> str:
    """Initialize CPU dispatch with synthetic decoder operations, without model data."""
    fp16 = np.array([0.5, -0.0, 1.5], dtype=np.float16)
    symbols = np.array([-2, 0, 2], dtype=np.int8)
    scales = torch.from_numpy(fp16.copy()).float()
    q = torch.from_numpy(symbols.copy()).float()
    value = q * scales[0]
    value = torch.where(torch.tensor([False, True, False]), -torch.zeros_like(value), value)
    actual = value.contiguous().numpy().tobytes()
    expected = np.array([-1.0, -0.0, 1.0], dtype=np.float32).tobytes()
    if actual != expected or scales.half().contiguous().numpy().tobytes() != fp16.tobytes():
        raise RuntimeError('synthetic CPU decoder arithmetic changed')
    return digest(actual + fp16.tobytes())


def check_bootstrap_receipt(record: dict[str, Any]) -> None:
    expected = digest(np.array([-1.0, -0.0, 1.0], dtype=np.float32).tobytes()
                      + np.array([0.5, -0.0, 1.5], dtype=np.float16).tobytes())
    if (record.get('before_landlock_sha256') != expected or record.get('after_landlock_sha256') != expected
            or record.get('cpuinfo_denied_errno') not in (errno.EACCES, errno.EPERM)):
        raise ValueError('synthetic CPU bootstrap or cpuinfo denial not established')


def fixed_worker(bundle: Path, profile_path: Path, guider: Path, output: Path, forbidden: list[Path], parent_net: str) -> None:
    """Consume only the fixed payload after namespace and read isolation."""
    if sys.platform != 'linux':
        raise RuntimeError('a distinct Linux network namespace is required')
    child_net = os.readlink('/proc/self/ns/net')
    if child_net == parent_net:
        raise RuntimeError('a distinct Linux network namespace is required')
    import socket

    interfaces = sorted(name for _, name in socket.if_nameindex())
    if interfaces != ['lo']:
        raise RuntimeError('network namespace must contain only loopback')
    if any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise RuntimeError('unexpected inherited file descriptor')
    import cv2
    import torch

    if any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise RuntimeError('unexpected descriptor after runtime import')
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if np.__version__ != '1.26.4' or cv2.__version__ != '4.10.0':
        raise ValueError('original preparation NumPy/OpenCV versions required')
    with Path('/proc/cpuinfo').open('rb') as cpuinfo:
        if not cpuinfo.read(1):
            raise RuntimeError('cpuinfo pre-isolation probe is empty')
    before_bootstrap = synthetic_torch_bootstrap(torch)
    output.mkdir(parents=True, exist_ok=False)
    allowed = _allowed_read_paths(bundle, output, forbidden)
    if any(p.resolve() == profile_path.resolve() or profile_path.resolve().is_relative_to(p.resolve()) for p in forbidden):
        raise ValueError('profile overlaps a forbidden input')
    for model_path in (profile_path, guider):
        if any(model_path.resolve() == p.resolve() or model_path.resolve().is_relative_to(p.resolve()) for p in forbidden):
            raise ValueError('allowed model or profile overlaps a forbidden input')
        allowed.append(model_path.resolve())
    from experiments.tennis.k6b_receiver_sam import restrict_reads

    abi = restrict_reads(allowed)
    try:
        with Path('/proc/cpuinfo').open('rb') as cpuinfo:
            cpuinfo.read(1)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EPERM):
            raise RuntimeError('cpuinfo probe did not prove read denial') from exc
        cpuinfo_denied = exc.errno
    else:
        raise RuntimeError('cpuinfo remained readable after isolation')
    bootstrap = {'before_landlock_sha256': before_bootstrap,
                 'after_landlock_sha256': synthetic_torch_bootstrap(torch),
                 'cpuinfo_denied_errno': cpuinfo_denied}
    check_bootstrap_receipt(bootstrap)
    denied = {}
    for path in forbidden:
        try:
            with path.open('rb') as stream:
                stream.read(1)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise RuntimeError(f'forbidden read did not prove denial: {path}') from exc
            denied[str(path)] = exc.errno
        else:
            raise RuntimeError(f'forbidden read succeeded: {path}')
    profile = pinned_profile(profile_path)
    arrays, bindings, match = decode_frozen_window(bundle.read_bytes(), profile)
    check_installed_guider(guider, profile)
    from experiments.tennis.w8_payload import unpack_match

    base = torch.load(guider, map_location='cpu', weights_only=True)
    decoded_match = unpack_match(match, profile['match']['sha256'], GUIDER_SHA, base)
    fingerprints = tensor_fingerprints(decoded_match)
    if len(fingerprints) != 480:
        raise ValueError('unexpected decoded match tensor count')
    hashes = {}
    for name in UNITS:
        path = output / f'{name}.npz'
        np.savez_compressed(path, **arrays[name])
        hashes[name] = digest(path.read_bytes())
    write_json(output / 'worker.json', {'bundle_sha256': WINDOW_SHA, 'profile_sha256': PROFILE_SHA,
               'hashes': hashes, 'bindings': bindings, 'match_tensors': fingerprints, 'cpu_bootstrap': bootstrap,
               'numpy': np.__version__, 'opencv': cv2.__version__,
               'cpu_s': cpu_seconds(), 'isolation': {'landlock_abi': abi, 'denied': denied,
               'allowed': [str(p) for p in allowed], 'network_namespace_distinct': True,
               'parent_network_namespace': parent_net, 'worker_network_namespace': child_net,
               'network_interfaces': interfaces,
               'extra_inherited_fds': False},
               'scope': 'input decode only; unused generator, SAM and native installations remain unverified'})


def fixed_inputs(stage: Path) -> tuple[dict[str, Path], list[Path]]:
    """Validate the fleet budget contract and bind staged/original identities."""
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (spec.get('device') != 'cpu' or spec.get('cpu_threads') != 1
            or spec.get('single_cpu_campaign') is not True or spec.get('defer_failed_publication') is not True
            or type(spec.get('budget_seconds')) not in (int, float) or not 0 < spec['budget_seconds'] <= 300
            or spec.get('environment', {}).get('sha256') != PREP_ENV_SHA
            or len(dispatch.get('cpu_affinity', [])) != 1):
        raise ValueError('one-core, deferred-publication, 300-second original preparation environment required')
    declared = spec['staged_inputs']
    staged = staging['inputs']
    if len({item['name'] for item in declared}) != len(declared) or len({item['name'] for item in staged}) != len(staged):
        raise ValueError('duplicate staged identity')
    lookup = {item['name']: item for item in declared}
    actual = {item['name']: item for item in staged}
    paths: dict[str, Path] = {}
    for name, sha in FIXED_INPUTS.items():
        if (name not in lookup or name not in actual or lookup[name]['sha256'] != sha
                or actual[name]['sha256'] != sha or lookup[name].get('extract', False)
                or actual[name].get('extract', False)):
            raise ValueError(f'missing or changed staged input: {name}')
        path = Path(actual[name]['path'])
        if file_sha(path) != sha:
            raise ValueError(f'staged input hash mismatch: {name}')
        paths[name] = path.absolute()
    forbidden_names = ('prepared', 'fits', 'base', 'source', 'masks', 'golden_generated')
    forbidden = list(dict.fromkeys([paths[name] for name in forbidden_names]
                    + [Path(lookup[name]['path']).absolute() for name in forbidden_names]))
    for path in forbidden:
        if not path.is_file():
            raise ValueError(f'forbidden probe is missing: {path}')
        with path.open('rb') as stream:
            if not stream.read(1):
                raise ValueError(f'forbidden probe is empty: {path}')
    return paths, forbidden


def validate_fixed(stage: Path, decoded_dir: Path, temporary_root: Path) -> dict[str, Any]:
    """Fresh offline validation; receiver outputs never grant access to goldens."""
    from experiments.tennis.w8_payload import unpack_window
    import cv2

    cv2.setNumThreads(1)
    if np.__version__ != '1.26.4' or cv2.__version__ != '4.10.0':
        raise ValueError('original preparation NumPy/OpenCV versions required')

    paths, forbidden = fixed_inputs(stage)
    profile = pinned_profile(paths['profile'])
    result = json.loads((decoded_dir / 'worker.json').read_text())
    check_bootstrap_receipt(result['cpu_bootstrap'])
    launch = json.loads((stage / 'fixed-launch.json').read_text())
    isolation = result['isolation']
    if (result['bundle_sha256'] != WINDOW_SHA or result['profile_sha256'] != PROFILE_SHA
            or isolation['landlock_abi'] < 1 or isolation['network_namespace_distinct'] is not True
            or not isolation.get('parent_network_namespace') or not isolation.get('worker_network_namespace')
            or isolation['parent_network_namespace'] == isolation['worker_network_namespace']
            or isolation['parent_network_namespace'] != launch.get('parent_network_namespace')
            or isolation.get('network_interfaces') != ['lo']
            or isolation['extra_inherited_fds'] is not False
            or set(isolation['denied']) != {str(p) for p in forbidden}
            or any(code not in (errno.EPERM, errno.EACCES) for code in isolation['denied'].values())):
        raise ValueError('worker isolation receipt mismatch')
    blob = paths['packet'].read_bytes()
    sections, _ = unpack_window(blob, profile)
    if sections[profile['judged']] != judged_bitmap(RECIPE):
        raise ValueError('paid judged bitmap mismatch')
    independently_decoded, bindings, _ = decode_frozen_window(blob, profile)
    if result['bindings'] != bindings:
        raise ValueError('decoded profile and worker bindings differ')
    golden_bytes = archive_members(paths['prepared'], [f'publish/units/{name}.npz' for name in UNITS])
    golden_root = Path(tempfile.mkdtemp(prefix='w8-independent-golden-', dir=temporary_root))
    decoded = {}
    golden = {}
    for name in UNITS:
        path = decoded_dir / f'{name}.npz'
        if path.is_symlink() or file_sha(path) != result['hashes'][name]:
            raise ValueError('decoded output hash mismatch')
        with np.load(path, allow_pickle=False) as arrays:
            decoded[name] = {key: arrays[key] for key in arrays.files}
        if (set(decoded[name]) != {'reference', 'reference_box', 't', 'run', 'box', 'pose', 'judged'}
                or not np.array_equal(decoded[name]['t'], np.arange(240))
                or not np.array_equal(np.flatnonzero(decoded[name]['judged']), np.arange(0, 240, 3))):
            raise ValueError('conditioning or scored-tick coverage mismatch')
        if any(not np.array_equal(decoded[name][key], independently_decoded[name][key]) for key in decoded[name]):
            raise ValueError('sealed worker output differs from independent payload decode')
        golden[name] = golden_root / f'{name}.npz'
        golden[name].write_bytes(golden_bytes[f'publish/units/{name}.npz'])
    validate_golden(decoded, golden, sections[profile['kps']])
    verify_base_archive(paths['base'], profile)
    match = verify_match_payload(sections[profile['match']['section']], profile, paths['fits'], paths['guider'])
    check_worker_match(result, match)
    return {'passed': True, 'checks': {'payload_identity': True, 'isolation': True,
            'exact_conditioning_and_golden': True, 'exact_match': True},
            'bundle_sha256': WINDOW_SHA, 'profile_sha256': PROFILE_SHA, 'match': match,
            'worker_sha256': file_sha(decoded_dir / 'worker.json'),
            'decoded_sha256': result['hashes'], 'cpu_seconds': cpu_seconds(),
            'temporary_golden_directory': str(golden_root),
            'scope': 'CPU input oracle only; unused generator, SAM and native installations require a separate pre-GPU gate'}


def fixed_run(stage: Path, arguments: dict[str, Path]) -> None:
    """Launch a source-free decoder, then a separate offline validator."""
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Linux CPU-only fleet runtime required')
    paths, forbidden = fixed_inputs(stage)
    if set(arguments) != set(FIXED_INPUTS) or any(arguments[n].resolve() != paths[n] for n in paths):
        raise ValueError('command arguments do not match the staged identities')
    namespace = os.readlink('/proc/self/ns/net')
    unshare = '/usr/bin/unshare'
    if not os.access(unshare, os.X_OK):
        raise RuntimeError('network namespace launcher unavailable')
    scratch = checked_scratch()
    env = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'TEMP', 'TMP', 'PS_SCRATCH_DIR') if key in os.environ}
    env.update({'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'XDG_CACHE_HOME': os.environ['TMPDIR'], 'TORCH_HOME': os.environ['TMPDIR']})
    command = [unshare, '--user', '--map-root-user', '--net', sys.executable, '-m',
               'experiments.tennis.w8_input_job', 'fixed-worker', '--bundle', str(paths['packet']),
               '--profile', str(paths['profile']), '--guider', str(paths['guider']), '--out', str(scratch / 'fixed-decoded'),
               '--parent-net', namespace, *[value for path in forbidden for value in ('--deny', str(path))]]
    started = time.monotonic()
    try:
        worker_result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                       env=env, close_fds=True, timeout=240, check=False)
    except subprocess.TimeoutExpired:
        write_json(stage / 'fixed-launch.json', {'timed_out': True, 'command': command,
                   'worker_wall_seconds': time.monotonic() - started, 'close_fds': True,
                   'parent_cpu_seconds': cpu_seconds()})
        raise
    (stage / 'fixed-worker.stdout').write_text(worker_result.stdout)
    (stage / 'fixed-worker.stderr').write_text(worker_result.stderr)
    write_json(stage / 'fixed-launch.json', {'returncode': worker_result.returncode,
               'worker_wall_seconds': time.monotonic() - started, 'command': command,
               'parent_network_namespace': namespace,
               'stdin': 'devnull', 'close_fds': True, 'sanitized_environment': sorted(env),
               'parent_cpu_seconds': cpu_seconds()})
    if worker_result.returncode:
        raise RuntimeError('isolated fixed-packet worker failed; logs retained')
    try:
        validation = subprocess.run([sys.executable, '-m', 'experiments.tennis.w8_input_job',
                                     'fixed-validate', '--stage', str(stage), '--output', str(stage / 'fixed-validation.json')],
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                    env=env, close_fds=True, timeout=240, check=False)
    except subprocess.TimeoutExpired:
        write_json(stage / 'fixed-validation-timeout.json', {'timed_out': True, 'cpu_seconds': cpu_seconds()})
        raise
    (stage / 'fixed-validator.stdout').write_text(validation.stdout)
    (stage / 'fixed-validator.stderr').write_text(validation.stderr)
    if validation.returncode:
        raise RuntimeError('independent fixed-packet validation failed; logs retained')
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    write_json(stage / 'fixed-resource.json', {'parent_cpu_seconds': cpu_seconds(),
               'children_cpu_seconds': children.ru_utime + children.ru_stime,
               'semantics': 'process-lifetime cumulative; subtract earlier snapshots, do not sum cumulative snapshots'})
    publish_fixed_outputs(stage, scratch)


def checked_scratch() -> Path:
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    if not scratch.is_absolute() or scratch.is_symlink() or not scratch.is_dir() or scratch.resolve() != scratch:
        raise ValueError('a real absolute fleet scratch directory is required')
    return scratch


def fixed_publication_files(stage: Path, scratch: Path) -> dict[str, Path]:
    decoded_dir = scratch / 'fixed-decoded'
    if decoded_dir.is_symlink() or not decoded_dir.is_dir() or decoded_dir.resolve() != decoded_dir:
        raise ValueError('invalid local decoded directory')
    files = {'worker.json': decoded_dir / 'worker.json',
             'validation.json': stage / 'fixed-validation.json',
             'launch.json': stage / 'fixed-launch.json', 'resources.json': stage / 'fixed-resource.json'}
    files.update({f'{name}.npz': decoded_dir / f'{name}.npz' for name in UNITS})
    total = 0
    tar_upper = 10240 + len(files) * 2048
    for path in files.values():
        parent = decoded_dir if path.parent == decoded_dir else stage
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(parent.resolve()):
            raise ValueError('invalid fixed publication file')
        size = path.stat().st_size
        total += size
        tar_upper += ((size + 511) // 512) * 512
    if max(total, tar_upper) > 20 * 1024 * 1024:
        raise ValueError('fixed publication exceeds 20 MiB')
    return files


def publish_fixed_outputs(stage: Path, scratch: Path) -> None:
    files = fixed_publication_files(stage, scratch)
    publish = scratch / 'publish'
    publish.mkdir(parents=False, exist_ok=False)
    for name, source in files.items():
        shutil.copyfile(source, publish / name)


def recover_fixed_publication(stage: Path, temporary_root: Path) -> Path:
    """Check a sealed archive and recover its bounded outputs to host-local temporary storage."""
    import tarfile

    staging = json.loads((stage / 'staging.json').read_text())
    archive_path = stage / 'published.tar'
    if (archive_path.is_symlink() or archive_path.stat().st_size > 20 * 1024 * 1024
            or file_sha(archive_path) != staging['published']['sha256']):
        raise ValueError('fixed publication identity or size mismatch')
    names = {'worker.json', 'validation.json', 'launch.json', 'resources.json', *(f'{name}.npz' for name in UNITS)}
    recovered = Path(tempfile.mkdtemp(prefix='w8-published-audit-', dir=temporary_root))
    with tarfile.open(archive_path) as archive:
        entries = archive.getmembers()
        expected = {f'publish/{name}' for name in names}
        regular = [item for item in entries if item.isfile()]
        if ({item.name for item in regular} != expected or len(regular) != len(expected)
                or sum(item.size for item in regular) > 20 * 1024 * 1024
                or any(not item.isfile() and not (item.isdir() and item.name.rstrip('/') == 'publish') for item in entries)):
            raise ValueError('unexpected fixed publication contents')
        for item in regular:
            stream = archive.extractfile(item)
            if stream is None:
                raise ValueError('unreadable fixed publication')
            with (recovered / Path(item.name).name).open('xb') as destination:
                shutil.copyfileobj(stream, destination, 1024 * 1024)
    for published, original in (('validation.json', 'fixed-validation.json'), ('launch.json', 'fixed-launch.json'),
                                ('resources.json', 'fixed-resource.json')):
        if file_sha(recovered / published) != file_sha(stage / original):
            raise ValueError('fixed publication receipt mismatch')
    return recovered


def fixed_validate_command(stage: Path, output: Path) -> int:
    try:
        sealed_path = stage / 'fixed-validation.json'
        if output == sealed_path:
            temporary_root = checked_scratch()
            decoded_dir = temporary_root / 'fixed-decoded'
            if decoded_dir.is_symlink() or not decoded_dir.is_dir() or decoded_dir.resolve() != decoded_dir:
                raise ValueError('invalid local decoded directory')
        else:
            staging = json.loads((stage / 'staging.json').read_text())
            temporary_root = Path(os.environ['TMPDIR'])
            if (not temporary_root.is_absolute() or temporary_root.is_symlink() or not temporary_root.is_dir()
                    or str(temporary_root) != staging['temporary_environment']['TMPDIR']):
                raise ValueError('fleet host-local temporary directory required')
            decoded_dir = recover_fixed_publication(stage, temporary_root)
        result = validate_fixed(stage, decoded_dir, temporary_root)
        if output != sealed_path:
            sealed = json.loads(sealed_path.read_text())
            if (sealed.get('passed') is not True or sealed.get('worker_sha256') != result['worker_sha256']
                    or sealed.get('decoded_sha256') != result['decoded_sha256']):
                raise ValueError('independent full validation seal mismatch')
            if stage.name != 'smoke':
                raise ValueError('fleet promotion validation must run on smoke')
            spec = json.loads((stage.parent / 'spec.json').read_text())
            clock = json.loads((stage / 'dispatch.json').read_text())['campaign_clock']
            now = time.monotonic()
            start, phase, end = clock['started'], clock['phase_started'], clock['budget_end']
            full = spec['full']['seconds']
            if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                        for v in (start, phase, end, now, full))
                    or not start <= phase <= now < end or not 0 < full <= 300
                    or end - start > spec['budget_seconds'] + 1):
                raise ValueError('invalid fleet projection clock or full allowance')
            staging = max(10.0, 1.25 * (phase - start))
            measured = now - phase
            if 1.25 * measured > full or end - now < staging + full + 10:
                raise ValueError('remaining budget cannot cover staging, full and publication')
            result['checks']['remaining_budget'] = True
            result['projection'] = {'remaining_seconds': end - now, 'staging_reserve_seconds': staging,
                                    'measured_smoke_seconds': measured, 'full_allowance_seconds': full,
                                    'publication_reserve_seconds': 10}
        write_json(output, result)
        return 0
    except Exception as exc:
        write_json(output, {'passed': False, 'checks': {'independent_fixed_validation': False},
                            'error': f'{type(exc).__name__}: {exc}', 'cpu_seconds': cpu_seconds()})
        return 1


def _verified_denials(paths: list[Path]) -> None:
    if len(paths) < 2 or len({str(path.resolve()) for path in paths}) != len(paths):
        raise ValueError("Distinct existing golden and probe files are required")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Denied-read probe is not a file: {path}")
        with path.open("rb") as handle:
            if not handle.read(1):
                raise ValueError(f"Denied-read probe is empty: {path}")


def _allowed_read_paths(packet_dir: Path, output: Path, forbidden: list[Path]) -> list[Path]:
    """Narrow code/runtime trees; fail when any forbidden file is within one."""
    candidates = [Path(sys.prefix), Path(sys.base_prefix), Path(__file__).resolve().parents[2],
                  packet_dir, output, Path("/usr/lib"), Path("/lib"), Path("/lib64"),
                  Path("/etc/ld.so.cache"), Path("/dev/null"), Path("/dev/zero"), Path("/dev/urandom")]
    denied_roots = {Path("/"), Path("/usr"), Path("/home"), Path("/local"), Path("/tmp"), Path("/var"), Path("/dev")}
    allowed = list(dict.fromkeys(p.resolve() for p in candidates if p.exists()))
    if not Path(sys.prefix).resolve().is_dir() or not Path(__file__).resolve().parents[2].is_dir():
        raise ValueError("Runtime and source trees must exist")
    for path in allowed:
        if path in denied_roots:
            raise ValueError(f"Overbroad read allowance: {path}")
        for target in forbidden:
            if target.resolve() == path or target.resolve().is_relative_to(path):
                raise ValueError(f"Read allowance includes forbidden input: {target}")
    return allowed


def worker(packet_dir: Path, output: Path, recipe: str, forbidden: list[Path]) -> None:
    cpu_limit(60)
    started = time.monotonic()
    import cv2

    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=False)
    allowed = _allowed_read_paths(packet_dir, output, forbidden)
    from experiments.tennis.k6b_receiver_sam import restrict_reads

    abi = restrict_reads(allowed)
    denials = {}
    for path in forbidden:
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise RuntimeError(f"Forbidden probe did not prove access denial: {path}") from exc
            denials[str(path)] = exc.errno
        else:
            raise RuntimeError(f"Worker read forbidden file: {path}")
    blob = (packet_dir / "keypoints.kps").read_bytes()
    references = {n: (packet_dir / f"{n}.png.ref").read_bytes() for n in UNITS}
    arrays, bindings = decode_inputs(blob, references, WINDOW, judged_bitmap(recipe))
    hashes = {}
    for name in UNITS:
        path = output / f"{name}.npz"
        np.savez_compressed(path, **arrays[name])
        hashes[name] = digest(path.read_bytes())
    write_json(output / "worker.json", {"bindings": bindings, "hashes": hashes, "numpy": np.__version__,
                                      "opencv": cv2.__version__, "cpu_s": cpu_seconds(),
                                      "wall_s": time.monotonic() - started, "recipe": recipe,
                                      "isolation": {"landlock_abi": abi, "allowed": [str(p) for p in allowed],
                                                    "denied": denials, "packet_read_sha256": digest(blob)}})


def run(transport: Path, prepared: Path, stage: Path, scratch: Path, recipe: str, source: Path) -> None:
    cpu_limit(120)
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only environment required")
    judged_bitmap(recipe)
    stage.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=False)
    packet_dir = stage / "packets"
    packet_dir.mkdir()
    requested = [PREFIX + "keypoints.kps", *(PREFIX + f"{n}.png.ref" for n in UNITS)]
    data = archive_members(transport, requested)
    for member, content in data.items():
        (packet_dir / Path(member).name).write_bytes(content)
    archived_golden = stage / "prepared.tar"
    shutil.copyfile(prepared, archived_golden)
    if digest(archived_golden.read_bytes()) != PREPARED_SHA:
        raise ValueError("Prepared archive identity mismatch")
    probe = stage / "forbidden-probe.bin"
    probe.write_bytes(b"landlock-negative-read-probe")
    forbidden = [prepared.resolve(), archived_golden.resolve(), probe.resolve(), source.resolve()]
    _verified_denials(forbidden)
    out = stage / "decoded"
    cmd = [sys.executable, "-m", "experiments.tennis.w8_input_job", "worker", "--packet-dir", str(packet_dir),
           "--out", str(out), "--recipe", recipe, *[v for path in forbidden for v in ("--deny", str(path))]]
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, check=False)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr += "\nWorker exceeded 120-second wall bound.\n"
    (stage / "worker.stdout").write_text(stdout)
    (stage / "worker.stderr").write_text(stderr)
    receipt = {"worker_returncode": returncode, "packet_dir": "packets", "prepared": "prepared.tar",
               "packets": {Path(k).name: digest(v) for k, v in data.items()}, "recipe": recipe,
               "parent_cpu_s": cpu_seconds(), "wall_s": time.monotonic() - started,
               "forbidden": [str(p) for p in forbidden], "access_scope": "Landlock required in worker; parent verifies probes before launch"}
    write_json(stage / "receipt.json", receipt)
    if returncode:
        raise ValueError("Packet decoder failed; logs and receipt retained")
    result = validate(stage)
    result["manifest"] = evidence_manifest(stage)
    write_json(stage / "full_validation.json", result)
    if not result["passed"]:
        raise ValueError("Strict in-stage validation failed; evidence retained")


def evidence_manifest(stage: Path) -> dict[str, str]:
    paths = ["receipt.json", "prepared.tar", "decoded/worker.json", "packets/keypoints.kps"]
    paths += [f"packets/{n}.png.ref" for n in UNITS]
    paths += [f"decoded/{n}.npz" for n in UNITS]
    result = {}
    for name in paths:
        path = stage / name
        if path.is_symlink() or not path.resolve().is_relative_to(stage.resolve()) or not path.is_file():
            raise ValueError(f"Invalid stage evidence path: {name}")
        result[name] = digest(path.read_bytes())
    return result


def validate(stage: Path) -> dict[str, Any]:
    cpu_limit(120)
    started = time.monotonic()
    import cv2

    cv2.setNumThreads(1)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        receipt = json.loads((stage / "receipt.json").read_text())
        worker_result = json.loads((stage / "decoded" / "worker.json").read_text())
        checks["cpu_only"] = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        checks["worker_success"] = receipt["worker_returncode"] == 0
        isolation = worker_result["isolation"]
        checks["landlock_active"] = (isinstance(isolation["landlock_abi"], int) and isolation["landlock_abi"] >= 1 and
                                     set(isolation["denied"]) == set(receipt["forbidden"]) and len(isolation["denied"]) == 4 and
                                     all(code in (errno.EACCES, errno.EPERM) for code in isolation["denied"].values()))
        details["isolation"] = isolation
        checks["declared_recipe"] = receipt["recipe"] == RECIPE == worker_result["recipe"]
        manifest = evidence_manifest(stage)
        checks["relative_paths"] = receipt["packet_dir"] == "packets" and receipt["prepared"] == "prepared.tar"
        packets = stage / "packets"
        checks["packet_hashes"] = receipt["packets"] == PACKET_SHAS and all(manifest["packets/" + n] == sha for n, sha in PACKET_SHAS.items())
        checks["prepared_hash"] = manifest["prepared.tar"] == PREPARED_SHA
        golden_bytes = archive_members(stage / "prepared.tar", [f"publish/units/{n}.npz" for n in UNITS])
        tmpdir = Path(os.environ["TMPDIR"])
        if not tmpdir.is_dir():
            raise ValueError("Host-local TMPDIR is unavailable")
        golden_dir = Path(tempfile.mkdtemp(prefix="w8-golden-validation-", dir=tmpdir))
        details["golden_directory"] = str(golden_dir)
        golden = {}
        decoded = {}
        for name in UNITS:
            golden[name] = golden_dir / f"{name}.npz"
            golden[name].write_bytes(golden_bytes[f"publish/units/{name}.npz"])
            path = stage / "decoded" / f"{name}.npz"
            checks[name + ":output_hash"] = digest(path.read_bytes()) == worker_result["hashes"][name]
            with np.load(path, allow_pickle=False) as data:
                decoded[name] = {k: data[k] for k in data.files}
            checks[name + ":fields"] = set(decoded[name]) == {"reference", "reference_box", "t", "run", "box", "pose", "judged"}
            checks[name + ":coverage"] = (len(decoded[name]["t"]) == 240 and int(decoded[name]["judged"].sum()) == 80)
        try:
            validate_golden(decoded, golden, (packets / "keypoints.kps").read_bytes())
            checks["exact_golden_and_native_equivalence"] = True
        except (ValueError, KeyError) as exc:
            checks["exact_golden_and_native_equivalence"] = False
            details["mismatch"] = str(exc)
        details["golden_hashes"] = {name: digest(golden_bytes[f"publish/units/{name}.npz"]) for name in UNITS}
        details["versions"] = {"numpy": np.__version__, "opencv": cv2.__version__, "worker_numpy": worker_result["numpy"], "worker_opencv": worker_result["opencv"]}
        checks["same_environment"] = details["versions"]["numpy"] == worker_result["numpy"] and details["versions"]["opencv"] == worker_result["opencv"]
        details["worker_cpu_s"] = worker_result["cpu_s"]
        details["parent_cpu_s"] = receipt["parent_cpu_s"]
        checks["process_cpu_limits"] = (0 <= details["parent_cpu_s"] <= 120 and
                                        0 <= details["worker_cpu_s"] <= 60 and cpu_seconds() <= 120)
    except Exception as exc:
        checks["validator_completed"] = False
        details["error"] = f"{type(exc).__name__}: {exc}"
    return {"passed": bool(checks) and all(checks.values()), "checks": checks, "details": details,
            "cpu_seconds": cpu_seconds(), "wall_seconds": time.monotonic() - started, "gpu_hours": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--transport", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--windows", type=int, required=True, choices=[1])
    p.add_argument("--deny-source", type=Path, required=True)
    p = sub.add_parser("worker")
    p.add_argument("--packet-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--deny", type=Path, action="append", required=True)
    p.add_argument("--recipe", required=True)
    sub.add_parser("validate")
    p = sub.add_parser('fixed-run')
    p.add_argument('--windows', type=int, choices=[1], required=True)
    for name in FIXED_INPUTS:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p = sub.add_parser('fixed-worker')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--guider', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parent-net', required=True)
    p.add_argument('--deny', type=Path, action='append', required=True)
    p = sub.add_parser('fixed-validate')
    p.add_argument('--stage', type=Path)
    p.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'fixed-worker':
        fixed_worker(args.bundle, args.profile, args.guider, args.out, args.deny, args.parent_net)
    elif args.command == 'fixed-run':
        fixed_run(Path(os.environ['PS_STAGE_DIR']), {name: getattr(args, name) for name in FIXED_INPUTS})
    elif args.command == 'fixed-validate':
        stage = args.stage or Path(os.environ['PS_STAGE_DIR'])
        output = args.output or Path(os.environ['PS_VALIDATION_PATH'])
        return fixed_validate_command(stage, output)
    elif args.command == "worker":
        raise RuntimeError('legacy transport worker is disabled pending fixed-window isolation integration')
    elif args.command == "run":
        raise RuntimeError('legacy transport oracle is disabled pending fixed-window isolation integration')
    else:
        stage = Path(os.environ["PS_STAGE_DIR"])
        result = validate(stage)
        try:
            sealed = json.loads((stage / "full_validation.json").read_text())
            result["checks"]["sealed_pass"] = sealed.get("passed") is True
            result["checks"]["sealed_manifest"] = sealed.get("manifest") == evidence_manifest(stage)
        except Exception as exc:
            result["checks"]["sealed_manifest"] = False
            result["details"]["seal_error"] = f"{type(exc).__name__}: {exc}"
        result["passed"] = all(result["checks"].values())
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
        print(json.dumps(result))
        return int(not result["passed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
