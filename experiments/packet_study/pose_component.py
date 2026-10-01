"""Lossless framed packing of retained pose bytes; no RGB/task-accuracy claim."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
import zlib

HEADER = struct.Struct('>4sBBQQ32s')
MAGIC = b'PPZ1'
MAX_BYTES = 16 * 1024 * 1024
EXPECTED = {'dwpose.bin': '54d7a5867f2381e8a6ab4d771444f1d7b0566b67108716720fb28ad4ff63d9e3',
            'dwpose_hands.pk.bin': 'ba355843b94b463d8f476e6f0065793d5f09649d3f42aed7c44c03dd06978ae3'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pack(data, level=6):
    if level not in (1, 6, 9) or len(data) > MAX_BYTES:
        raise ValueError('unsupported level or input size')
    compressed = zlib.compress(data, level)
    return HEADER.pack(MAGIC, 1, level, len(data), len(compressed), hashlib.sha256(data).digest()) + compressed


def unpack(blob):
    if len(blob) < HEADER.size:
        raise ValueError('truncated header')
    magic, flags, level, original, encoded, checksum = HEADER.unpack_from(blob)
    if magic != MAGIC or flags != 1 or level not in (1, 6, 9):
        raise ValueError('unsupported format/codec flags')
    if original > MAX_BYTES or encoded != len(blob) - HEADER.size:
        raise ValueError('invalid framed size')
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(blob[HEADER.size:], original + 1)
    except zlib.error as exc:
        raise ValueError('invalid compressed payload') from exc
    if len(raw) != original or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('decoded size/trailing data mismatch')
    if hashlib.sha256(raw).digest() != checksum:
        raise ValueError('decoded checksum mismatch')
    return raw


def read_pk_records(blob):
    records = []; offset = 0
    while offset < len(blob):
        if offset + 4 > len(blob):
            raise ValueError('truncated PK length')
        size = struct.unpack_from('<I', blob, offset)[0]; offset += 4
        if offset + size > len(blob):
            raise ValueError('truncated PK record')
        record = blob[offset:offset + size]; offset += size
        if len(record) < 3 or record[:2] != b'PK' or len(record) != 3 + 47 * record[2]:
            raise ValueError('invalid PK instance count/length')
        records.append(record)
    return records


def pose_interpretation(root):
    path = root / 'pose_delta.py'
    spec = importlib.util.spec_from_file_location('pinned_pose_decoder', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    dwb = (root / 'dwpose.bin').read_bytes()
    pk = read_pk_records((root / 'dwpose_hands.pk.bin').read_bytes())
    metadata = json.loads((root / 'dwpose.json').read_text())
    frames = module.decode_pose_stream(dwb)
    count = metadata['n_frames']
    if len(frames) != count or len(pk) != count:
        raise ValueError('declared pose record denominator mismatch')
    if [parts[0] for parts in frames] != pk:
        raise ValueError('DWB2 decoded hands differ from retained PK records')
    return {'decoder_source': str(path), 'decoder_source_sha256': digest(path.read_bytes()),
            'decoder_entry': 'decode_pose_stream', 'declared_records': count,
            'decoded_dwb2_records': len(frames), 'decoded_pk_records': len(pk),
            'dwb2_hand_packets_equal_retained_pk': True,
            'hand_instances': sum(record[2] for record in pk),
            'sidecar_reported_hand_instances': metadata.get('n_hands_detected'),
            'sidecar_hand_count_matches_wire': metadata.get('n_hands_detected') == sum(record[2] for record in pk),
            'ordinal_records_are_original_media_frame_ids': False,
            'source_rgb_alignment_verified': False, 'independent_task_truth': False}


def receipt(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data), 'sha256': digest(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--unpack', type=Path)
    parser.add_argument('--decoded', type=Path)
    args = parser.parse_args()
    if args.unpack:
        raw = unpack(args.unpack.read_bytes())
        with args.decoded.open('xb') as stream:
            stream.write(raw)
        print(json.dumps({'bytes': len(raw), 'sha256': digest(raw)})); return
    started = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.input_root
    inputs = [receipt(root / name) for name in [*EXPECTED, 'dwpose.json', 'pose_delta.py']]
    changed = [item['path'] for item in inputs if Path(item['path']).name in EXPECTED and item['sha256'] != EXPECTED[Path(item['path']).name]]
    report = {'scope': 'lossless retained pose-component transport; no RGB/full-codec/ground-truth claim',
              'started_at_utc': started, 'smoke': args.smoke, 'inputs': inputs,
              'previously_reported_input_hashes': EXPECTED, 'input_changed_since_inventory': changed,
              'worker': receipt(Path(__file__)),
              'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'code_revision_scope': 'base checkout revision; new worker frozen separately by SHA-256',
              'decoder_source_origin': '/Users/manu/Desktop/PointStream/demo/pipeline/maps/pose_delta.py read-only snapshot',
              'command': sys.argv, 'python': sys.version, 'platform': platform.platform(),
              'zlib_compile_version': zlib.ZLIB_VERSION, 'zlib_runtime_version': zlib.ZLIB_RUNTIME_VERSION,
              'thread_environment': {key: os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']},
              'gpu_uuid': None, 'native_encoder_decoder': 'not used', 'header_bytes': HEADER.size,
              'pose_interpretation': pose_interpretation(root), 'rows': []}
    for name in ['dwpose.bin'] if args.smoke else list(EXPECTED):
        data = (root / name).read_bytes()
        for level in [6] if args.smoke else [1, 6, 9]:
            package = args.out / f'{name}.z{level}.ppz'; package.write_bytes(pack(data, level))
            decoded = args.out / f'{name}.z{level}.decoded'
            command = [sys.executable, '-m', 'experiments.packet_study.pose_component', '--unpack', str(package), '--decoded', str(decoded)]
            child = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
            if decoded.read_bytes() != data:
                raise ValueError('fresh process byte parity failed')
            report['rows'].append({'input_name': name, 'level': level, 'original_bytes': len(data),
                'framed_bytes': package.stat().st_size, 'header_bytes': HEADER.size,
                'compressed_payload_bytes': package.stat().st_size - HEADER.size,
                'net_bytes_saved': len(data) - package.stat().st_size,
                'net_fraction_saved': 1 - package.stat().st_size / len(data),
                'package': receipt(package), 'decoded': receipt(decoded),
                'fresh_process_byte_parity': True, 'receiver_command': command,
                'receiver_stdout': json.loads(child.stdout)})
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['elapsed_seconds'] = time.monotonic() - clock
    out = args.out / 'report.json'; out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(receipt(out)))

if __name__ == '__main__':
    main()
