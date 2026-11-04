"""CPU-only GVC-RT native entropy build/interface pilot; no models or video.

Draft for review: execute only in a fresh task-owned scratch directory, using
an authenticated packed Python runtime and source archive. No installation is
made into the supplied runtime. This is not codec or GPU qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

SOURCE_SHA = 'a7ad708cfaea7d0030cc1d78bd2cd1f1d62a1df999d8eaece292bff510c30f7e'
RUNTIME_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
SOURCES = ('src/cpp/py_rans/py_rans.cpp', 'src/cpp/py_rans/py_rans.h',
           'src/cpp/py_rans/rans.cpp', 'src/cpp/py_rans/rans.h', 'src/cpp/py_rans/rans_byte.h')
HEADER_SHA = 'e51728f9a5c0b446d6aebf8101f73d86c245000dec2f57822f9f697d6459efe0'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_test(extension_dir: Path) -> dict[str, Any]:
    import numpy as np

    sys.path.insert(0, str(extension_dir))
    import MLCodec_extensions_cpp as native

    cdf = np.array([[0, 32768, 65535, 65536]], dtype=np.int32)
    lengths = np.array([4], dtype=np.int32)
    offsets = np.array([0], dtype=np.int32)
    symbols = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int16)
    indexes = np.zeros(len(symbols), dtype=np.uint8)
    cases = []
    for dual in (False, True):
        encoder = native.RansEncoder()
        decoder = native.RansDecoder()
        encoder.set_use_two_encoders(dual)
        decoder.set_use_two_decoders(dual)
        eidx = encoder.add_cdf(cdf, lengths, offsets)
        didx = decoder.add_cdf(cdf, lengths, offsets)
        if eidx != didx:
            raise ValueError('encoder/decoder CDF indexes differ')
        encoder.reset()
        encoder.encode_y(symbols << 8, eidx)
        encoder.flush()
        stream = encoder.get_encoded_stream()
        decoder.set_stream(stream)
        decoded = decoder.decode_and_get_y(indexes, didx)
        if not np.array_equal(decoded, symbols.astype(np.int8)):
            raise ValueError('native entropy roundtrip mismatch')
        cases.append({'two_entropy_coders': dual, 'exact_roundtrip': True,
                      'stream_bytes': int(stream.size), 'stream_hex': stream.tobytes().hex(),
                      'sha256': hashlib.sha256(stream.tobytes()).hexdigest()})
    return {'cases': cases, 'numpy': np.__version__}


def kill_owned_tree(parent: int) -> None:
    # Remain in the fleet's process group. Stop only the timed-out child's
    # descendants (g++ can spawn cc1plus), then the child itself.
    children: dict[int, list[int]] = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / 'stat').read_text()
            ppid = int(stat[stat.rfind(')') + 2:].split()[1])
            children.setdefault(ppid, []).append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    def kill(pid: int) -> None:
        for child in children.get(pid, []):
            kill(child)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    kill(parent)


def run_owned(command: list[str], *, cwd: Path, env: dict[str, str],
              stdout: Any, stderr: Any, timeout: float) -> int:
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                            stdin=subprocess.DEVNULL)
    try:
        return proc.wait(timeout=timeout)
    except BaseException:
        kill_owned_tree(proc.pid)
        proc.wait()
        raise


def _terminate(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt('fleet terminated the pilot')


def build(args: argparse.Namespace) -> int:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('Linux and one-CPU affinity required')
    source = args.source.resolve(strict=True)
    root = args.runtime.resolve(strict=True)
    work = args.work.resolve(strict=True)
    if any(work.iterdir()):
        raise ValueError('fresh empty work directory required')
    if digest(source) != SOURCE_SHA:
        raise ValueError('source archive identity mismatch')
    if root.name != 'tree' or root.parent.name != RUNTIME_SHA:
        raise ValueError('runtime is not the qualified staged environment path')
    python = root / 'bin/python'
    python_header = root / 'include/python3.12/Python.h'
    torch_include = root / 'lib/python3.12/site-packages/torch/include'
    header = torch_include / 'pybind11/pybind11.h'
    if not python.is_file() or not python_header.is_file() or digest(header) != HEADER_SHA:
        raise ValueError('Python or vendored pybind11 headers differ')
    source_dir = work / 'source'
    source_dir.mkdir()
    with tarfile.open(source, 'r:gz') as archive:
        for name in SOURCES:
            member = archive.getmember(name)
            if not member.isfile() or member.size > 2_000_000:
                raise ValueError('invalid source member')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('missing source member')
            (source_dir / Path(name).name).write_bytes(stream.read())
    command = ['/usr/bin/g++', '-shared', '-fPIC', '-std=c++17', '-O3', '-Wall', '-Wextra', '-Werror',
               '-pthread', f'-I{python_header.parent}', f'-I{torch_include}',
               str(source_dir / 'py_rans.cpp'), str(source_dir / 'rans.cpp'),
               '-o', str(work / 'MLCodec_extensions_cpp.so')]
    env = {'PATH': '/usr/bin:/bin', 'TMPDIR': str(work), 'TMP': str(work), 'TEMP': str(work),
           'XDG_CACHE_HOME': str(work / '.cache'),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
    signal.signal(signal.SIGTERM, _terminate)
    compiler_version = subprocess.run(['/usr/bin/g++', '--version'], check=True, capture_output=True,
                                      text=True, timeout=5).stdout.splitlines()[0]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    status: dict[str, Any] = {'source_sha256': SOURCE_SHA, 'runtime_archive_sha256': RUNTIME_SHA,
                              'header_sha256': HEADER_SHA, 'command': command, 'compiler_version': compiler_version,
                              'compiler_sha256': digest(Path('/usr/bin/g++').resolve()), 'passed': False}
    try:
        with (work / 'compiler.stdout').open('xb') as stdout, (work / 'compiler.stderr').open('xb') as stderr:
            compiler_status = run_owned(command, cwd=work, env=env, stdout=stdout, stderr=stderr, timeout=90)
        status['compiler_status'] = compiler_status
        if compiler_status:
            raise RuntimeError('native compile failed')
        remaining = 110 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('no budget remains for interface test')
        probe = [str(python), str(Path(__file__).resolve()), 'probe', '--work', str(work)]
        with (work / 'probe.stdout').open('xb') as stdout, (work / 'probe.stderr').open('xb') as stderr:
            probe_status = run_owned(probe, cwd=work, env=env, stdout=stdout, stderr=stderr, timeout=min(20, remaining))
        status['probe_status'] = probe_status
        if probe_status:
            raise RuntimeError('native interface test failed')
        status['probe'] = json.loads((work / 'probe.stdout').read_text())
        status['extension_sha256'] = digest(work / 'MLCodec_extensions_cpp.so')
        status['source_members_sha256'] = {name: digest(source_dir / Path(name).name) for name in SOURCES}
        status['probe_stdout_sha256'] = digest(work / 'probe.stdout')
        status['passed'] = True
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        status['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        status['wall_seconds'] = time.monotonic() - started
        status['child_cpu_seconds'] = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        with (work / 'pilot.json').open('x') as f:
            json.dump(status, f, sort_keys=True, indent=2)
            f.write('\n')
    return 0 if status['passed'] else 1


def validate(artifact: Path, source: Path) -> dict[str, Any]:
    receipt = json.loads((artifact / 'pilot.json').read_text())
    if (receipt.get('passed') is not True or receipt.get('source_sha256') != SOURCE_SHA
            or receipt.get('runtime_archive_sha256') != RUNTIME_SHA or receipt.get('header_sha256') != HEADER_SHA
            or receipt.get('compiler_status') != 0 or receipt.get('probe_status') != 0
            or not receipt.get('compiler_version') or len(receipt.get('compiler_sha256', '')) != 64
            or set(receipt.get('source_members_sha256', {})) != set(SOURCES)
            or digest(artifact / 'MLCodec_extensions_cpp.so') != receipt.get('extension_sha256')
            or digest(artifact / 'probe.stdout') != receipt.get('probe_stdout_sha256')):
        raise ValueError('invalid pilot identity or exit receipt')
    if digest(source) != SOURCE_SHA or digest(Path('/usr/bin/g++').resolve()) != receipt['compiler_sha256']:
        raise ValueError('source archive or compiler differs from receipt')
    with tarfile.open(source, 'r:gz') as archive:
        for name in SOURCES:
            member = archive.extractfile(name)
            if member is None or hashlib.sha256(member.read()).hexdigest() != receipt['source_members_sha256'][name]:
                raise ValueError('compiled source member mismatch')
    result = json.loads((artifact / 'probe.stdout').read_text())
    if result != receipt.get('probe'):
        raise ValueError('probe result mismatch')
    cases = result.get('cases', [])
    if len(cases) != 2 or [c.get('two_entropy_coders') for c in cases] != [False, True]:
        raise ValueError('missing single/dual entropy results')
    for case in cases:
        raw = bytes.fromhex(case['stream_hex'])
        if (not case.get('exact_roundtrip') or len(raw) != case.get('stream_bytes') or not raw
                or hashlib.sha256(raw).hexdigest() != case.get('sha256')):
            raise ValueError('invalid native entropy stream receipt')
    replay = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'probe', '--work', str(artifact)],
                            check=True, capture_output=True, text=True, timeout=20,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                                 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    if json.loads(replay.stdout) != result:
        raise ValueError('fresh-process native roundtrip differs')
    return {'passed': True, 'checks': {'source_and_runtime_identity': True, 'compiler_identity': True,
                                      'extension_hash': True, 'single_and_dual_roundtrip_receipts': True, 'fresh_process_replay': True}}


def fleet_run(args: argparse.Namespace) -> int:
    if os.environ.get('PS_STAGE') != args.phase:
        raise ValueError('fleet phase mismatch')
    if args.phase == 'full':
        return 0
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PS_CLAIMED_GPU_UUID'):
        raise ValueError('CPU-only fleet claim required')
    stage = Path(os.environ['PS_STAGE_DIR']).resolve(strict=True)
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    packed = dispatch['staged_environment']
    if (packed['sha256'] != RUNTIME_SHA or Path(packed['python']).resolve() != Path(sys.executable).resolve()
            or not any(x['sha256'] == SOURCE_SHA and Path(x['path']).resolve() == args.source.resolve()
                       for x in dispatch['staged_inputs'])):
        raise ValueError('fleet staged identity mismatch')
    clock = dispatch.get('campaign_clock', {})
    if not isinstance(clock.get('budget_end'), (int, float)) or time.monotonic() + 140 > clock['budget_end']:
        raise ValueError('insufficient campaign time for build, probe, validation and release')
    work = Path(os.environ['PS_SCRATCH_DIR']) / 'gvcrt-entropy'
    work.mkdir()
    args.work = work
    args.runtime = Path(packed['prefix'])
    result = build(args)
    artifact = stage / 'gvcrt-entropy'
    artifact.mkdir()
    for name in ('pilot.json', 'compiler.stdout', 'compiler.stderr', 'probe.stdout', 'probe.stderr',
                 'MLCodec_extensions_cpp.so'):
        source = work / name
        if source.is_file():
            shutil.copyfile(source, artifact / name)
    return result


def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('fleet', 'probe', 'validate'))
    p.add_argument('--phase', choices=('smoke', 'full'))
    p.add_argument('--work', type=Path)
    p.add_argument('--source', type=Path)
    args = p.parse_args()
    if args.mode == 'probe':
        if args.work is None:
            p.error('probe requires --work')
        print(json.dumps(run_test(args.work), sort_keys=True))
        return 0
    if args.mode == 'validate':
        stage = Path(os.environ['PS_STAGE_DIR']) / 'gvcrt-entropy'
        dispatch = json.loads((stage.parent / 'dispatch.json').read_text())
        inputs = [x for x in dispatch['staged_inputs'] if x.get('sha256') == SOURCE_SHA]
        if len(inputs) != 1:
            raise ValueError('historical source input missing or ambiguous')
        verdict = validate(stage, Path(inputs[0]['path']))
        with Path(os.environ['PS_VALIDATION_PATH']).open('x') as f:
            json.dump(verdict, f, sort_keys=True)
            f.write('\n')
        return 0
    if args.source is None or args.phase is None:
        p.error('fleet requires --source and --phase')
    return fleet_run(args)


if __name__ == '__main__':
    raise SystemExit(main())
