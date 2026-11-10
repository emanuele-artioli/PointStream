"""Replay and score all six prepared points with one shared set of judges."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k5d_checkpoint as cp, k5d_encoder as enc, k5d_source_oracle as oracle
from experiments.tennis.k5d_timing import Timer


def score_all(args: argparse.Namespace, checkpoint_sha: str, identity: dict[str, str]) -> None:
    import torch
    from experiments.tennis import k5b_trackers as trk, windows as tw
    from src.codecs import quality

    root, output = Path(args.checkpoint), Path(args.output)
    same_job = getattr(args, 'same_job_receiver', False) or (root / 'same-job-receiver.json').exists()
    if same_job:
        cp.verify(root, checkpoint_sha, identity, require_same_job_receiver=True)
    else:
        cp.verify(root, checkpoint_sha, identity)
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get('PS_CPU_ALLOWANCE') != '1' or torch.cuda.device_count() != 1:
        raise ValueError('one CPU and one CUDA device required')
    report = json.loads((root / 'pilot.json').read_text())
    source = json.loads((root / 'source-qualification.json').read_text())
    marker = json.loads((root / 'same-job-receiver.json').read_text()) if same_job else None
    expected_producer = getattr(args, 'receiver_producer_job', None)
    if expected_producer and (marker is None or marker['producer_job'] != expected_producer):
        raise ValueError('retained receiver producer differs from released producer')
    oracle.validate_report(source)
    if list(torch.cuda.get_device_capability(0)) != source['device']['capability']:
        raise ValueError('GPU class differs from prepared source qualification')
    output.mkdir(parents=True, exist_ok=False)
    timer = Timer(output / 'timing.json')
    timer.measure('shared.verify_inputs_before', lambda: cp.verify_score_inputs(args, source, report))
    game = int(report['clip'].split('/')[1][4:])
    clips = [c for c in trk.tracknet_clips(Path(args.archive), [game]) if c['id'] == report['clip']]
    if len(clips) != 1:
        raise ValueError('clip missing or ambiguous')
    clip = {**clips[0], 'rows': clips[0]['rows'][:report['frames']]}
    labels = trk.tracknet_labels(clip)
    if (labels != [(r['label_kind'], r['label_x'], r['label_y']) for r in source['rows']]
            or [f"{clip['dir']}/{r[0]}" for r in clip['rows']] != [r['member'] for r in source['rows']]):
        raise ValueError('source labels or frame ordering changed')
    images = timer.measure('shared.source.decode', lambda: trk.tracknet_frames(Path(args.archive), clip))
    bt = timer.measure('model.balltrack', lambda: trk.BallTrack(Path(args.balltrack_code), Path(args.balltrack_weight)))
    zoo = timer.measure('model.wasb', lambda: trk.Zoo(Path(args.wasb_code), Path(args.wasb_weights), names=('wasb-step1',)))
    nets = (timer.measure('model.lpips', lambda: quality.load_lpips(Path(args.lpips_backbone), 'cuda')),
            timer.measure('model.dists', lambda: quality.load_dists(Path(args.dists_backbone), 'cuda')))
    if next(bt.net.parameters()).device.type != 'cuda' or next(zoo.settings['wasb-step1']['net'].parameters()).device.type != 'cuda':
        raise ValueError('judge device mismatch')

    def judges(data: np.ndarray) -> dict[str, Any]:
        return {'wasb': enc.prediction_rows(labels, zoo.run('wasb-step1', data)),
                'balltrack': enc.prediction_rows(labels, bt.run(data))}

    receipts = []
    for p in report['points']:
        codec, point = p['codec'], p['point']
        tag = f'{codec}.{point}'
        work = Path(os.environ['PS_SCRATCH_DIR']) / f'coalesced-{codec}-{point}'
        if same_job:
            work.mkdir(parents=True, exist_ok=False)
            saved = root / f'{codec}-{point}' / 'receiver-evidence'
            receipt = json.loads((saved / 'receiver.json').read_text())
            native_yuv = saved / 'native.yuv'
            rendered_rgb = saved / 'rendered.rgb'
        else:
            wire = root / f'{codec}-{point}' / 'stream.psbl'
            cmd = [sys.executable, '-m', 'experiments.tennis.k5d_worker', 'decode', '--wire', str(wire),
                   '--resources', str(root / 'resources.json'), '--work-dir', str(work)]
            timer.measure(f'receiver.decode.{tag}', lambda: subprocess.run(cmd, check=True, capture_output=True, timeout=2400))
            receipt = json.loads((work / 'receiver.json').read_text())
            native_yuv, rendered_rgb = work / 'native.yuv', work / 'rendered.rgb'
        if (receipt['decoded_yuv_sha256'] != p['encoder_decoded_sha256']
                or receipt['rendered_rgb_sha256'] != p['encoder_rendered_sha256']
                or receipt['wire_sha256'] != p['wire_sha256']
                or receipt['resource_map_sha256'] != report['resource_map_sha256']):
            raise ValueError('receiver pixels differ from prepare')
        n = report['frames']
        w, h = report['size']
        decoded = timer.measure(f'receiver.rgb.{tag}', lambda: tw.rgb_frames(native_yuv, w, h, n, work / 'native-rgb.npy'))
        rendered = np.memmap(rendered_rgb, dtype=np.uint8, mode='r', shape=(n, h, w, 3))
        bm, bj, sm, sj, reuse = timer.measure(f'point.{tag}', lambda: enc.score_arms(
            images, decoded, rendered, labels, nets, judges, p['streak']['guard_off'], timer, tag))
        p['baseline'].update(metrics=bm, judges=bj)
        p['streak'].update(metrics=sm, judges=sj, reuse=reuse)
        path = output / f'point-{codec}-{point}.json'
        cp.write_new(path, {'schema': 1, 'checkpoint_sha256': checkpoint_sha,
                            'execution_identity': identity, 'point': p, 'receiver_replay': receipt,
                            **({'receiver_origin': marker, 'receiver_origin_role': 'producer job; no later native decode'} if same_job else {})})
        receipts.append({'path': str(path), 'sha256': oracle.sha(path)})
    if same_job:
        cp.write_new(output / 'receiver-provenance.json', {'checkpoint_sha256': checkpoint_sha, 'producer': marker,
                                                         'scope': 'producer job receiver; scoring performs no native decode'})
    timer.measure('shared.verify_inputs_after', lambda: cp.verify_score_inputs(args, source, report))
    result = timer.measure('shared.assemble', lambda: cp.assemble(root, checkpoint_sha, receipts, output / 'assembled', identity))
    cp.write_new(output / 'coalesced-validation.json', {'result': result, 'checkpoint_sha256': checkpoint_sha,
                 'execution_identity': identity, 'points': [{'name': Path(x['path']).name, 'sha256': x['sha256']} for x in receipts]})
