"""CPU-only native decoder inventory. No receiver isolation claim is made.

Run only after a reviewed aggregate CPU/wall reservation. The launcher must
start one CPU-affined process with CUDA hidden and retain its task handle.
This utility does not invoke a model, GPU, shell, package manager or network.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time
import zlib
from typing import Any

from experiments.tennis.w8_receiver_native_bootstrap import parent_death_signal

PACKET_SHA = 'eff1aca84a63e870a2994bd227b3dacb5583fee5c54a420235d828fa1a31bbf4'
IVF_SHA = 'f9622b6af4cddea3d70226c0b22e12374f4c77aadf2120e54d5d21ba55bfbbb8'
FFMPEG_SHA = '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'
YUV_SHA = '7490edf9b678db805c23baae49a2a3b5e9d7b037b18dd216320aad47a061beff'
ENV_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
LOCAL = Path('/local/users/emanuele/pointstream')
MAIN = LOCAL / 'cache' / ENV_SHA / 'tree'
DATA = Path('/home/itec/emanuele/Datasets/pointstream-data')
SYSTEM = tuple(Path(p).resolve() for p in ('/usr/lib', '/usr/lib64', '/lib', '/lib64'))
LD_CACHE = Path('/etc/ld.so.cache')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
        after = os.fstat(stream.fileno())
    def stamp(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if stamp(before) != stamp(after) or stamp(after) != stamp(path.stat()):
        raise ValueError(f'mapped file changed while hashing: {path}')
    return h.hexdigest()


def extract_ivf(blob: bytes) -> bytes:
    if sha(blob) != PACKET_SHA or blob[:5] != b'PSW8\x02' or len(blob) < 13:
        raise ValueError('paid packet mismatch')
    coded, raw_size = struct.unpack('>II', blob[5:13])
    if coded > 4_194_304 or raw_size > 4_194_304 or 13 + coded > len(blob):
        raise ValueError('packet header size')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(blob[13:13+coded], raw_size+1)
    if len(raw) != raw_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('packet header encoding')
    header = json.loads(raw)
    entries = header['sections']
    if not isinstance(entries, list) or not entries or len(entries) > 10000:
        raise ValueError('packet section directory')
    target, found, pos = header['profile']['ivf'], None, 13+coded
    names = set()
    for entry in entries:
        name, size = entry['name'], entry['length']
        if name in names or type(size) is not int or size < 0 or pos + size > len(blob):
            raise ValueError('packet section boundary')
        names.add(name)
        body = blob[pos:pos+size]
        if sha(body) != entry['sha256']:
            raise ValueError('packet section hash')
        if name == target:
            found = body
        pos += size
    if pos != len(blob) or found is None or sha(found) != IVF_SHA:
        raise ValueError('packet IVF binding')
    return found


def mapped_files(text: str, roots: tuple[Path, ...]) -> list[Path]:
    result = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or fields[5].startswith('['):
            continue
        name = fields[5]
        if name.endswith(' (deleted)') or not name.startswith('/'):
            raise ValueError('deleted or unrecognized file-backed mapping')
        path = Path(name).resolve(strict=True)
        cache = Path(name) == LD_CACHE and path == LD_CACHE
        if not path.is_file() or not (cache or any(path == root or path.is_relative_to(root) for root in roots)):
            raise ValueError(f'unapproved mapped file: {path}')
        result.add(path)
    return sorted(result)


def elf_interpreter(binary: Path) -> Path:
    with binary.open('rb') as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:6] != b'\x7fELF\x02\x01':
            raise ValueError('expected little-endian ELF64 executable')
        phoff = struct.unpack_from('<Q', header, 32)[0]
        phentsize, phnum = struct.unpack_from('<HH', header, 54)
        if phentsize != 56 or not 0 < phnum < 512 or phoff > 10_000_000:
            raise ValueError('invalid ELF program header')
        for i in range(phnum):
            stream.seek(phoff+i*phentsize)
            entry = stream.read(phentsize)
            if len(entry) != phentsize:
                raise ValueError('truncated ELF program header')
            kind, _, offset, _, _, size, _, _ = struct.unpack('<IIQQQQQQ', entry)
            if kind == 3:
                if not 1 < size < 4096:
                    raise ValueError('invalid ELF interpreter')
                stream.seek(offset)
                name = stream.read(size)
                if not name.endswith(b'\x00') or b'\x00' in name[:-1]:
                    raise ValueError('invalid ELF interpreter name')
                return Path(os.fsdecode(name[:-1])).resolve(strict=True)
    raise ValueError('ELF interpreter missing')


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)


def run(config: dict[str, Any]) -> None:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1 or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('Linux, one CPU and hidden GPU required')
    parent_cpu, version_cpu, decode_cpu, wall = (config[k] for k in ('parent_cpu', 'version_cpu', 'decode_cpu', 'wall_seconds'))
    if (parent_cpu != 6 or version_cpu != 1 or decode_cpu != 9 or wall > 120 or wall < 1
            or resource.getrlimit(resource.RLIMIT_CPU) != (6, 9)):
        raise ValueError('unapproved preflight limits')
    output = Path(config['output'])
    if not output.is_absolute() or output.parent.resolve(strict=True) != (LOCAL / 'audits').resolve(strict=True):
        raise ValueError('fresh host-local audits directory required')
    if LOCAL.stat().st_dev == DATA.stat().st_dev or os.statvfs(LOCAL).f_bavail * os.statvfs(LOCAL).f_frsize < 200_000_000:
        raise ValueError('local filesystem or free space requirement')
    output.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    packet, binary = (Path(config[k]).resolve(strict=True) for k in ('packet', 'ffmpeg'))
    if not binary.is_relative_to(MAIN.resolve(strict=True)) or file_sha(binary) != FFMPEG_SHA:
        raise ValueError('FFmpeg does not match installed MAIN identity')
    if packet.stat().st_size > 5_000_000:
        raise ValueError('packet too large')
    ivf = extract_ivf(packet.read_bytes())
    (output/'packet.ivf').write_bytes(ivf)
    loader = elf_interpreter(binary)
    roots = (MAIN.resolve(strict=True), *SYSTEM)
    if not any(loader.is_relative_to(root) for root in roots):
        raise ValueError('ELF interpreter outside allowed installation')
    library_dirs = [Path(p).resolve(strict=True) for p in config['library_dirs']]
    if not library_dirs or any(not any(p == root or p.is_relative_to(root) for root in roots) for p in library_dirs):
        raise ValueError('unapproved library path')
    env = {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': ':'.join(map(str, library_dirs)),
           'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'TMPDIR': str(output),
           'TMP': str(output), 'TEMP': str(output), 'XDG_CACHE_HOME': str(output)}
    ppid = os.getpid()
    def child_limit(seconds: int) -> Any:
        def set_limit() -> None:
            parent_death_signal(ppid)
            resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        return set_limit
    with (output/'ffmpeg-version.txt').open('xb') as stdout, (output/'ffmpeg-version.stderr').open('xb') as stderr:
        version = subprocess.Popen([str(binary), '-version'], stdin=subprocess.DEVNULL, stdout=stdout,
                                   stderr=stderr, env=env, start_new_session=True,
                                   preexec_fn=child_limit(version_cpu), close_fds=True)
        version_handle = {'pid': version.pid, 'pgid': version.pid, 'started_monotonic': time.monotonic()}
        (output/'version-handle.json').write_text(json.dumps(version_handle, sort_keys=True))
        version_status = 'completed'
        try:
            version_code = version.wait(timeout=10)
        except subprocess.TimeoutExpired:
            version_status = 'timeout'
            _kill_group(version)
            version_code = version.returncode
        (output/'version-execution.json').write_text(json.dumps(
            {**version_handle, 'status': version_status, 'returncode': version_code}, sort_keys=True))
        if version_status != 'completed' or version_code:
            raise ValueError('FFmpeg version command failed')
    argv = [str(binary), '-nostdin', '-v', 'error', '-re', '-threads', '1', '-c:v', 'libdav1d',
            '-i', str(output/'packet.ivf'), '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-n', str(output/'native.yuv')]
    with (output/'decode.stdout').open('xb') as stdout, (output/'decode.stderr').open('xb') as stderr:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env,
                                close_fds=True, start_new_session=True, preexec_fn=child_limit(decode_cpu))
        handle = {'pid': proc.pid, 'pgid': proc.pid, 'started_monotonic': time.monotonic(), 'argv': argv}
        (output/'handle.json').write_text(json.dumps(handle, sort_keys=True))
        try:
            deadline = min(time.monotonic()+2, started+wall)
            maps: list[Path] = []
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise ValueError('decoder exited before maps inventory')
                maps = mapped_files(Path(f'/proc/{proc.pid}/maps').read_text(), roots)
                raw_path = output/'native.yuv'
                if (raw_path.exists() and raw_path.stat().st_size > 0 and binary in maps and loader in maps
                        and any('libdav1d.so' in p.name for p in maps)):
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError('decoder shared-library mappings did not appear')
            dav1d = [p for p in maps if 'libdav1d.so' in p.name]
            if len(dav1d) != 1:
                raise ValueError('ambiguous libdav1d mappings')
            if sum(p.stat().st_size for p in maps) > 1_073_741_824:
                raise ValueError('mapped library hashing exceeds byte cap')
            hashes = {str(path): {'sha256': file_sha(path), 'bytes': path.stat().st_size} for path in maps}
            library = ctypes.CDLL(str(dav1d[0]))
            library.dav1d_version.restype = ctypes.c_char_p
            observed_version = library.dav1d_version()
            if observed_version != b'1.5.4':
                raise ValueError(f'wrong actual libdav1d version: {observed_version!r}')
            remaining = started + wall - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('preflight wall limit')
            code = proc.wait(timeout=remaining)
            raw = output/'native.yuv'
            if code or not raw.is_file() or raw.stat().st_size != 47_001_600 or file_sha(raw) != YUV_SHA:
                raise ValueError('native decode or original YUV hash mismatch')
            receipt = {'passed': True, 'scope': 'unisolated CPU native inventory; not RGB or receiver qualification',
                       'packet_sha256': PACKET_SHA, 'ivf_sha256': IVF_SHA, 'ffmpeg_sha256': FFMPEG_SHA,
                       'ffmpeg_path': str(binary), 'elf_interpreter': str(loader), 'mapped_files': hashes,
                       'libdav1d_path': str(dav1d[0]), 'libdav1d_version': '1.5.4', 'yuv_sha256': YUV_SHA,
                       'original_manifest_sha256': '8dd22cf9320b0d04809a8f34d0f4961d812f8488909043ef2bbeeba14090c15b',
                       'yuv_bytes': raw.stat().st_size, 'process': handle, 'returncode': code,
                       'parent_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
                       'children_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2]),
                       'wall_seconds': time.monotonic()-started}
            (output/'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2))
            print(json.dumps({'passed': True, 'output': str(output), 'wall_seconds': receipt['wall_seconds']}))
        except BaseException as exc:
            if proc.poll() is None:
                _kill_group(proc)
            (output/'failure.json').write_text(json.dumps({'error': repr(exc), 'process': handle,
                'returncode': proc.returncode, 'wall_seconds': time.monotonic()-started}, sort_keys=True))
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (6, 9))
    if args.config.stat().st_size > 1_000_000:
        raise ValueError('preflight configuration too large')
    data = args.config.read_bytes()
    if sha(data) != args.config_sha256:
        raise ValueError('preflight configuration identity')
    run(json.loads(data))


if __name__ == '__main__':
    main()
