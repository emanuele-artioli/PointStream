"""Fresh native encode/decode comparison of legacy and default CUDA streams."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from experiments.tennis.k6b_clean_background_prep import INPUTS, SOURCE_YUV
from experiments.tennis.k6b_ideal_player import digest, write
from experiments.tennis.parallel_baselines import CHECKPOINTS


def run(a: argparse.Namespace) -> None:
    from experiments.jobs.monitor import publish_progress
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6 import downscaled_source
    from src.codecs.dcvc_uf_worker import dcvc_command, unpack_container

    began = time.monotonic()
    expected = {'source': INPUTS['source'], **CHECKPOINTS}
    if {k: digest(getattr(a, k)) for k in expected} != expected:
        raise ValueError('Frozen input identity changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    out = scratch/'publish'
    out.mkdir()
    source = scratch/'source.yuv'
    subprocess.run(convert_command('ffmpeg', a.source, source, 9, 240), check=True, capture_output=True, timeout=120)
    if digest(source) != SOURCE_YUV:
        raise ValueError('Original native source changed')
    raw = downscaled_source(source, a.frames, 2, scratch/'source-half.yuv')
    rows = []
    for policy in (('legacy',) if a.legacy_only else ('legacy', 'default')):
        work = out/policy
        work.mkdir()
        plan: dict[str, Any] = {'structure': 'htl', 'qp': 21, 'frame_count': a.frames, 'width': 960,
            'height': 544, 'src_type': 'yuv420', 'frames_file': str(raw), 'container': str(work/'stream.psdc'),
            'out_file': str(work/'decoded.yuv'), 'image_ckpt': str(a.image_ckpt), 'image_sha256': CHECKPOINTS['image_ckpt'],
            'video_ckpt': str(a.video_ckpt), 'video_sha256': CHECKPOINTS['video_ckpt'],
            'profile': False, 'stream_policy': policy}
        reports, commands = {}, []
        measured_variable = 0.
        elapsed = 0.
        errors = []
        for action in ('encode', 'decode', 'fresh-decode'):
            actual = 'decode' if action == 'fresh-decode' else action
            selected = dict(plan)
            if action == 'fresh-decode':
                selected['out_file'] = str(work/'fresh.yuv')
            pp = work/(action+'-plan.json')
            write(pp, selected)
            command, env, cwd = dcvc_command(Path(sys.prefix), actual, pp, work/(action+'.json'))
            start = time.monotonic()
            result = subprocess.run(command, cwd=cwd, env={**os.environ, **env}, capture_output=True, timeout=120)
            elapsed += time.monotonic()-start
            commands.append(command)
            (work/(action+'.log')).write_bytes(result.stdout+result.stderr)
            if result.returncode:
                errors.append({'action': action, 'returncode': result.returncode})
                break
            report = json.loads((work/(action+'.json')).read_text())
            reports[action] = report
            measured_variable += report['encode_seconds'] if actual == 'encode' else sum(p['decode_seconds'] for p in report['passes'])
        checks: dict[str, bool] = {'all_native_commands': not errors and len(reports) == 3}
        if checks['all_native_commands']:
            encoder, decoder, fresh = (reports[k] for k in ('encode', 'decode', 'fresh-decode'))
            header, _ = unpack_container((work/'stream.psdc').read_bytes())
            sha = digest(work/'stream.psdc')
            checks.update(container=header == {'structure': 'htl', 'frame_count': a.frames}
                and all(r['container_sha256'] == sha for r in reports.values()),
                frame_count=all(len(r['passes'][0]['frame_sha256']) == a.frames for r in (decoder, fresh)),
                two_pass=decoder['deterministic'] and fresh['deterministic'],
                fresh_process=decoder['passes'][0]['frame_sha256'] == fresh['passes'][0]['frame_sha256'],
                intra=encoder['i_recon_sha256'] == decoder['passes'][0]['frame_sha256'][0] == fresh['passes'][0]['frame_sha256'][0],
                bytes_equal=digest(work/'decoded.yuv') == digest(work/'fresh.yuv'),
                memory=0 < encoder['peak_allocated_mib'] < 5000,
                declared_policy=all(r['checkpoints']['stream_policy'] == policy for r in reports.values()),
                gpu=all('A6000' in r['environment']['device_name'] and r['environment']['capability'] == [8, 6] for r in reports.values()))
        rows.append({'policy': policy, 'checks': checks, 'passed': all(checks.values()),
            'native_seconds': measured_variable, 'fixed_seconds': max(0, elapsed-measured_variable),
            'errors': errors, 'commands': commands,
            'reports_sha256': {k: digest(work/(k+'.json')) for k in reports}})
        publish_progress(os.environ['PS_STAGE'], len(rows))
    summary = {'frames': a.frames, 'source_yuv_sha256': SOURCE_YUV,
        'rows': rows, 'legacy_only': a.legacy_only, 'legacy_passed': rows[0]['passed'],
        'default_passed': rows[1]['passed'] if len(rows) > 1 else None, 'seconds': time.monotonic()-began,
        'scope': 'Fresh encoding on same claimed A6000; no stored-stream decode or quality evidence.',
        'inputs': expected, 'timing_secondary': True}
    write(out/'diagnosis.json', summary)
    write(Path(os.environ['PS_STAGE_DIR'])/'diagnosis-summary.json', summary)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    s = json.loads((stage/'diagnosis-summary.json').read_text())
    spec = json.loads((stage.parent/'spec.json').read_text())
    projection = 1.35*(sum(r['fixed_seconds']+r['native_seconds']*6 for r in s['rows'])
                       + max(0, s['seconds']-sum(r['fixed_seconds']+r['native_seconds'] for r in s['rows'])))
    checks = {'frames': s['frames'] == 4, 'source': s['source_yuv_sha256'] == SOURCE_YUV,
        'selected_variant_correct': s['legacy_passed'] if s['legacy_only'] else s['default_passed'],
        'variants': [r['policy'] for r in s['rows']] == (['legacy'] if s['legacy_only'] else ['legacy', 'default']),
        'cost': projection < spec['full']['seconds'], 'budget': spec['budget_seconds'] <= 600,
        'timing_secondary': s['timing_secondary']}
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks,
        'projected_full_seconds': projection})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    for k in ('source', 'image_ckpt', 'video_ckpt'):
        p.add_argument('--'+k.replace('_', '-'), type=Path)
    p.add_argument('--frames', type=int, choices=(4, 24), default=4)
    p.add_argument('--legacy-only', action='store_true', help='Complete original supported-path reproduction only')
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, k) is None for k in ('source', 'image_ckpt', 'video_ckpt')):
        p.error('Frozen source/checkpoints required')
    else:
        run(a)


if __name__ == '__main__':
    main()
