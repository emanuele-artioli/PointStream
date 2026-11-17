"""CPU audit of actual frozen JPEG references; no generation or quality scores.

Run from the repository root with PYTHONPATH=. and its pinned check environment.
Inputs remain read-only; choose a new output directory outside the checkout and
both input roots. Do not run Python with assertions disabled.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

import cv2
import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b, k6b_rate_streams as rates, w8_payload as wire
from experiments.tennis.player_requalification import array_sha


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    protected = (root, args.inputs.resolve(), args.prepared.resolve())
    if any(args.output.resolve().is_relative_to(path) for path in protected):
        raise ValueError('raw output must be outside the checkout and input roots')
    args.output.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.monotonic(), time.process_time()
    record_root = root / 'docs/experiments/common-player-packet-20261117'
    manifest_bytes = (args.prepared / 'manifest.json').read_bytes()
    frozen = root / 'docs/experiments/player-reference-pilot-20261117/prepared-manifest.json'
    assert manifest_bytes == frozen.read_bytes()
    manifest = json.loads(manifest_bytes)
    old_blob = (args.inputs / 'packet-eff1aca8.psw8').read_bytes()
    old_profile_bytes = (args.inputs / 'profile-bec98d67.json').read_bytes()
    assert digest(old_blob) == manifest['70']['packet_sha256'] == manifest['40']['packet_sha256']
    assert digest(old_profile_bytes) == manifest['70']['profile_sha256'] == manifest['40']['profile_sha256']
    old_profile = json.loads(old_profile_bytes)
    sections, _ = wire.unpack_window(old_blob, old_profile)
    _, units = rates.decode_keypoints(sections[old_profile['kps']])
    by_name = {u['name']: u for u in units}
    order = [k6b.unit_name(old_profile['window'], p) for p in old_profile['recipe']['compositing_order']]
    keypoints = rates.encode_keypoints_kps2({'native_frames': 240}, [by_name[n] for n in order])
    base = sections[old_profile['ivf']]
    profile = json.loads((record_root / 'cpu-profile.json').read_text())
    assert profile['order'] == order
    assert digest(Path('experiments/tennis/k6b_generate.py').read_bytes()) == profile['resources']['generator_code']
    native = Path('/opt/homebrew/Cellar/ffmpeg/8.1/bin/ffmpeg')
    assert digest(native.read_bytes()) == profile['resources']['native_decoder']
    refs = {u['reference']: sections[u['reference']] for u in old_profile['units']}
    png = packet.build(profile, base, keypoints, refs)
    prior_audit = json.loads((record_root / 'retained-input-final-audit.json').read_text())
    assert digest(png) == prior_audit['new_packet']['sha256']
    assert len(png) == prior_audit['new_packet']['bytes'] == 391981
    build_sha = digest(cv2.getBuildInformation().encode())
    variants = []
    for quality in (70, 40):
        info = manifest[str(quality)]
        assert build_sha == info['decoder']['build_information_sha256']
        folder = args.prepared / f'q{quality}'
        assert digest((folder / 'conditioning.tar').read_bytes()) == info['archive_sha256']
        assert digest((folder / 'config.json').read_bytes()) == info['config_sha256']
        rows = {row['unit']: row for row in info['units']}
        jpeg_profile = copy.deepcopy(profile)
        jpeg_profile['recipe'] = copy.deepcopy(packet.JPEG_RECIPES[quality])
        jpeg_profile['resources']['reference_decoder'] = build_sha
        jpeg_profile['reference_rgb_sha256'] = {
            run['reference']: rows[run['unit']]['decoded_rgb_sha256'] for run in profile['runs']}
        jpeg_refs, checks = {}, []
        for run in profile['runs']:
            name, refname = run['unit'], run['reference']
            row = rows[name]
            payload = (folder / f'{name}.ref1').read_bytes()
            assert digest(payload) == row['encoded_ref1_sha256'] and len(payload) == row['encoded_ref1_bytes']
            jpeg_refs[refname] = payload
        blob = packet.build(jpeg_profile, base, keypoints, jpeg_refs)
        decoded_sections, accounting, decoded = packet.unpack(blob)
        assert decoded_sections['base'] == base and decoded_sections['keypoints'] == keypoints
        assert packet.coverage(decoded, order) == [order] * 240
        assert sum(accounting.values()) == len(blob) and accounting['match'] == 0
        for name in order:
            run = decoded[name][0]
            target = folder / 'publish' / f'{name}.npz'
            assert digest(target.read_bytes()) == rows[name]['new_npz_sha256']
            original = args.inputs / f'{name}.npz'
            assert digest(original.read_bytes()) == rows[name]['source_npz_sha256']
            with np.load(target, allow_pickle=False) as golden, np.load(original, allow_pickle=False) as source:
                same = {'reference': np.array_equal(run['reference'], golden['reference']),
                        'reference_rgb_sha256': array_sha(run['reference']) == rows[name]['decoded_rgb_sha256'],
                        'reference_box': np.array_equal(run['reference_box'], golden['reference_box']),
                        'ticks': np.array_equal(run['t'], golden['t']),
                        'boxes': np.array_equal(np.asarray(run['boxes']), golden['box']),
                        'run_ids': np.array_equal(np.zeros(240, np.int64), golden['run']),
                        'seed': run['seed'] == int(hashlib.sha256(name.encode()).hexdigest()[:7], 16) * 1009,
                        'poses': all(np.array_equal(packet.pose_for_tick(run, i), golden['pose'][i]) for i in range(240)),
                        'non_reference_npz_arrays': set(source.files) == set(golden.files) and all(
                            np.array_equal(source[k], golden[k]) for k in source.files if k != 'reference')}
            assert all(same.values()), (quality, name, same)
            checks.append({'unit': name, 'golden_npz_sha256': rows[name]['new_npz_sha256'], 'checks': same})
        report = packet.inspect(blob, jpeg_profile['resources'])
        assert report['role_bytes'] == accounting and not report['learned_receiver']
        out = args.output / f'q{quality}'
        out.mkdir()
        (out / 'packet.pscp').write_bytes(blob)
        (out / 'profile.json').write_text(json.dumps(jpeg_profile, indent=2, sort_keys=True) + '\n')
        variants.append({'quality': quality, 'packet_path': str(out / 'packet.pscp'),
                         'packet_sha256': digest(blob), 'packet_bytes': len(blob),
                         'profile_sha256': digest((out / 'profile.json').read_bytes()),
                         'role_bytes': accounting, 'component_bytes': report['component_bytes'],
                         'diagnostic_kbps_over_4_8_seconds': len(blob) * 8 / 4.8 / 1000,
                         'comparisons': checks})
    (args.output / 'base.ivf').write_bytes(base)
    command = [str(native), '-v', 'error', '-nostdin', '-threads', '1', '-i', str(args.output / 'base.ivf'),
               '-map', '0:v:0', '-f', 'framemd5', str(args.output / 'base-decoded.framemd5')]
    subprocess.run(command, check=True, timeout=60)
    frame_md5 = (args.output / 'base-decoded.framemd5').read_bytes()
    count = sum(bool(line) and not line.startswith(b'#') for line in frame_md5.splitlines())
    assert count == 240
    record = {'status': 'PASS', 'scope': 'CPU packet/conditioning byte audit, no learned generation or quality scoring',
              'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'implementation_sha256': digest(Path('experiments/tennis/common_player_packet.py').read_bytes()),
              'audit_script_sha256': digest(Path(__file__).read_bytes()),
              'prepared_manifest_sha256': digest(manifest_bytes), 'old_packet_sha256': digest(old_blob),
              'old_profile_sha256': digest(old_profile_bytes), 'png_packet_unchanged_sha256': digest(png),
              'png_packet_unchanged_bytes': len(png), 'variants': variants,
              'opencv': {'version': cv2.__version__, 'build_configuration_sha256': build_sha},
              'native_decode': {'command': command, 'binary_sha256': digest(native.read_bytes()),
                                'frames': count, 'framemd5_sha256': digest(frame_md5)},
              'wall_seconds': time.monotonic() - wall, 'parent_cpu_seconds': time.process_time() - cpu,
              'child_cpu_seconds': resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
              'limitations': ['This does not execute a learned receiver, masks, full-canvas composition or ball coding.',
                             'JPEG quality and longer-motion behavior remain untested. No GPU job or new quality metric is included.',
                             'All240conditioning ticks are checked; the approved appearance pilot renders only24ticks per player.',
                             'OpenCV build information is a configuration fingerprint, not a binary/runtime identity. Linux parity is unproven.',
                             'Pinned hashes authenticate this comparison to frozen inputs; self-declared packet hashes do not establish source truth.',
                             'Model resource hashes are reused declarations, not newly loaded or rehashed model weights.',
                             'Reportedkbps is short-window accounting only, not matched-quality compression evidence.']}
    (args.output / 'audit.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'png_unchanged_bytes': len(png),
                      'jpeg_packet_bytes': {v['quality']: v['packet_bytes'] for v in variants},
                      'native_frames': count, 'wall_seconds': record['wall_seconds'],
                      'cpu_seconds': record['parent_cpu_seconds'] + record['child_cpu_seconds'],
                      'evidence': str(args.output / 'audit.json')}))


if __name__ == '__main__':
    main()
