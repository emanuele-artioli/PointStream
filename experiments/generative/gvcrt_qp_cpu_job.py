"""Fleet CPU-only GVC checkpoint/table qualification; never runs a codec."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.generative import gvcrt_peer_job_w8leaves as peer

HERE = Path(__file__).resolve().parent


def run():
    stage = Path(os.environ['PS_STAGE_DIR']).resolve(strict=True)
    if os.environ['PS_STAGE'] == 'full':
        (stage / 'noop.json').write_text('{"no_additional_work": true}\n')
        return
    if os.environ['PS_STAGE'] != 'smoke' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU smoke with CUDA hidden required')
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('single CPU affinity required')
    dispatch = peer.read_json(stage / 'dispatch.json')
    packed = dispatch['staged_environment']
    if packed['sha256'] != peer.RUNTIME_SHA or Path(packed['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('runtime identity mismatch')
    inputs = {name: peer.staged(dispatch, name, digest) for name, digest in (
        ('source', peer.SOURCE_SHA), ('i_model', peer.I_SHA), ('p_model', peer.P_SHA), ('entropy', peer.ENTROPY_SHA))}
    if any(peer.sha(inputs[name]) != digest for name, digest in (
        ('source', peer.SOURCE_SHA), ('i_model', peer.I_SHA), ('p_model', peer.P_SHA), ('entropy', peer.ENTROPY_SHA))):
        raise ValueError('input bytes mismatch')
    if peer.sha(HERE / 'gvcrt_source_marker.json') != peer.MARKER_SHA or peer.sha(HERE / 'gvcrt_native_bridge.py') != peer.BRIDGE_SHA:
        raise ValueError('source marker or bridge mismatch')
    work = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True) / 'gvcrt-qp-cpu'
    work.mkdir()
    author = work / 'author'
    peer.unpack(inputs['source'], author)
    shutil.copyfile(HERE / 'gvcrt_source_marker.json', author / '.gvcrt-source.json')
    env = peer.env_for(work, author, inputs['entropy'])
    env['PYTHONPATH'] = os.pathsep.join([str(HERE.parents[1]), env['PYTHONPATH']])
    cmd = [sys.executable, '-m', 'experiments.generative.gvcrt_qp_coverage', '--official-repo', str(author),
           '--i-checkpoint', str(inputs['i_model']), '--i-sha256', peer.I_SHA,
           '--p-checkpoint', str(inputs['p_model']), '--p-sha256', peer.P_SHA,
           '--output', str(stage / 'qp-coverage.json')]
    subprocess.run(cmd, env=env, cwd=work, check=True, timeout=35, stdin=subprocess.DEVNULL)


def validate():
    stage = Path(os.environ['PS_STAGE_DIR'])
    receipt = peer.read_json(stage / 'qp-coverage.json')
    proof, env = receipt['cpu_proof'], receipt['environment']
    tables = receipt['tables']
    checks = {
        'cpu_only': proof['cuda_visible_devices'] == '' and proof['cuda_initialized_before'] is False
        and proof['cuda_initialized_after'] is False and len(proof['affinity']) == 1,
        'devices': all(all(d == 'cpu' for d in entry['devices']) for entry in receipt['devices'].values())
        and receipt['devices']['I.parameters']['count'] > 0 and receipt['devices']['P.parameters']['count'] > 0,
        'coverage': len(receipt['coverage']) == 2 and all(x['strict_complete_coverage'] for x in receipt['coverage']),
        'table_shapes': all(tables['shapes'][f'{role}.{name}'][0] == size for role, names, size in (
            ('I', ('q_scale_enc', 'q_scale_dec', 'q_scale_recon'), 10),
            ('P', ('q_scale_enc', 'q_scale_dec', 'q_scale_feature', 'q_scale_recon'), 12)) for name in names),
        'shifts': tables['p_shift_indices'] == {'0': [0, 2, 1], '3': [3, 5, 4], '9': [9, 11, 10]},
        'identities': env['source_revision'] == 'd0e32bfa3e8e282f9a77437c223c605858eea637'
        and env['i_sha256'] == peer.I_SHA and env['p_sha256'] == peer.P_SHA
        and env['entropy_sha256'] == peer.ENTROPY_SHA and env['bridge_sha256'] == peer.BRIDGE_SHA
        and proof['source_manifest_sha256'] == peer.MARKER_SHA
        and proof['helper_sha256'] == peer.sha(HERE / 'gvcrt_qp_coverage.py')
        and proof['interpreter_sha256'] == peer.sha(Path(sys.executable).resolve()),
        'process_and_usage': proof['pid'] > 0 and proof['pgid'] > 0 and proof['elapsed_seconds'] > 0
        and proof['usage_after']['self']['user_seconds'] >= proof['usage_before']['self']['user_seconds'],
    }
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': all(checks.values()), 'checks': checks}))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['validate']:
        sys.exit(validate())
    if len(sys.argv) != 3 or sys.argv[1] != 'run' or sys.argv[2] != os.environ.get('PS_STAGE'):
        raise SystemExit('expected run with matching fleet stage, or validate')
    run()
