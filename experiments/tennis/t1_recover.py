"""Read-only recovery of completed native RGB decodes from T1's stopped pilot."""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4
from experiments.tennis.t1_oracle import FG, GRID
from experiments.visor.b1 import file_sha256, stage_dir, write_json
from src.pointstream import bitstream as bs


def pack(args: argparse.Namespace) -> int:
    source = args.source_stage
    state = json.loads((source.parent / 'state.json').read_text())
    if state['status'] not in {'failed', 'complete', 'budget_exhausted'} or state['host_alias'] != 'gpu6':
        raise ValueError('source job is not terminal on the recovery host')
    staging = json.loads((source / 'staging.json').read_text())
    scratch = Path(staging['scratch'])
    work = scratch / k4.safe(args.window)
    published = scratch / 'publish' / k4.safe(args.window)
    original = json.loads((args.inputs / 'inputs.json').read_text())['windows'][args.window]
    w = original['window']
    ball = next(r for r in json.loads(args.ball_report.read_text())['windows'] if r['id'] == args.window)
    classes = np.load(work / 'classes.npy', mmap_mode='r')
    truth = np.load(work / 'truth.npy', mmap_mode='r')
    n = int(classes.shape[0])
    if classes.shape != (240, 1080, 1920) or truth.shape != (240, 1080, 1920, 3):
        raise ValueError('incomplete original reference arrays')
    if classes.dtype != np.uint8 or truth.dtype != np.uint8:
        raise ValueError('wrong original pixel type')
    paths = {'truth.npy': work / 'truth.npy', 'classes.npy': work / 'classes.npy'}
    records = []
    for kind, points in [('foreground', FG), ('background', GRID), ('anchor', GRID)]:
        for codec, value, scale in points:
            name = f'{kind}-{codec}-{value}-s{scale}'
            native = published / (name + ('.ivf' if codec == 'svt' else '.psdc'))
            pixel_file = work / name / 'shown.npy'
            pixels = np.load(pixel_file, mmap_mode='r')
            if pixels.shape != truth.shape or pixels.dtype != np.uint8:
                raise ValueError(f'{name}: incomplete native decode')
            data = native.read_bytes()
            payload = bs.ivf_to_obus(data)[0] if codec == 'svt' else data
            checks: dict[str, Any] = {}
            if codec == 'dcvc':
                enc = json.loads((published / (name + '-encode.json')).read_text())
                dec = json.loads((published / (name + '-decode.json')).read_text())
                checks = {'deterministic': dec['deterministic'],
                          'decoder_matches_encoder_intra': dec['passes'][0]['frame_sha256'][0] == enc['i_recon_sha256'],
                          'environment': enc.get('environment')}
                if not checks['deterministic'] or not checks['decoder_matches_encoder_intra']:
                    raise ValueError(f'{name}: native decode checks failed')
            from experiments.tennis.k6 import scaled_size
            records.append({'kind': kind, 'name': name, 'codec': codec, 'value': value, 'scale': scale,
                            'size': list(scaled_size(scale)), 'native_stream': native.name,
                            'payload_bytes': len(payload), 'checks': checks})
    selected = records[:args.max_decodes]
    for record in selected:
        paths[record['name'] + '.npy'] = work / record['name'] / 'shown.npy'
    ready = json.loads((source.parent / 'ready.json').read_text())
    smoke = json.loads((source.parent / 'smoke' / 't1-oracle.json').read_text())
    manifest = {'id': args.window, 'frames': n, 'fps': w['fps'], 'source_sha256': w['sha256'],
                'ball_sha256': ball['sha256'], 'native_job': source.parent.name,
                'native_code': ready['git_head'], 'native_gpu': state.get('gpu'),
                'reference_yuv_sha256': file_sha256(work / 'reference.yuv'),
                'tools': smoke['tools'], 'torch': smoke['torch'], 'streams': records,
                'judged': [t for t in range(n) if str(int(w['start']) + t) in original['fits'] and (classes[t] > 0).any() and t % 3 == 0],
                'arrays': {name: {'sha256': file_sha256(path), 'bytes': path.stat().st_size} for name, path in paths.items()}}
    archive_path = stage_dir() / 'native-decoded.tar'
    with tarfile.open(archive_path, 'w') as archive:
        body = json.dumps(manifest).encode()
        item = tarfile.TarInfo('manifest.json')
        item.size = len(body)
        archive.addfile(item, io.BytesIO(body))
        for name, path in paths.items():
            archive.add(path, arcname=name)
        for path in sorted(published.iterdir()):
            archive.add(path, arcname='streams/' + path.name)
    index = {'archive_sha256': file_sha256(archive_path), 'archive_bytes': archive_path.stat().st_size,
             'native_job': manifest['native_job'], 'selected_decodes': len(selected),
             'native_checks': all(r['checks'].get('deterministic', True) and r['checks'].get('decoder_matches_encoder_intra', True) for r in records),
             'source_read_only': True}
    write_json(stage_dir() / 'recovery.json', index)
    from experiments.jobs.monitor import publish_progress
    publish_progress('decoded-pixel-recovery', len(selected))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['pack', 'validate'])
    p.add_argument('--source-stage', type=Path)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--ball-report', type=Path)
    p.add_argument('--window')
    p.add_argument('--max-decodes', type=int, default=15)
    args = p.parse_args()
    if args.command == 'validate':
        r = json.loads((stage_dir() / 'recovery.json').read_text())
        checks = {'cpu': os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'native': r['native_checks'],
                  'some_decodes': r['selected_decodes'] > 0,
                  'archive_size': (stage_dir() / 'native-decoded.tar').stat().st_size == r['archive_bytes']}
        with tarfile.open(stage_dir() / 'native-decoded.tar') as archive:
            f = archive.extractfile('manifest.json')
            assert f is not None
            manifest = json.loads(f.read())
            checks['array_identities'] = len(manifest['arrays']) == r['selected_decodes'] + 2
        write_json(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks})
        return 0 if all(checks.values()) else 1
    return pack(args)


if __name__ == '__main__':
    raise SystemExit(main())
